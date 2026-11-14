#include "simulation_coupled_step.h"

#include "core/thermal/thermal_solver_linear_direct.h"
#include "core/ventilation/flow_calculation.h"
#include "hrv/hrv_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_error.h"
#include "utils/utils.h"

#include <algorithm>
#include <chrono>
#include <memory>

CoupledStepData performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                                              ThermalNetwork& thermalNetwork,
                                              const SimulationConstants& constants,
                                              std::ostream& logs,
                                              TimingList& timings,
                                              const std::string& meta,
                                              simulation::TimestepSolveMetrics* metrics) {
    const bool logEnabled = (constants.logVerbosity > 0);
    CoupledStepData step;

    // 圧力・熱の前に HRV 給気境界を更新（還気/外気の現在値から η で合成）
    if (constants.temperatureCalc || constants.humidityCalc) {
        hrv::updateSupplyBoundaries(thermalNetwork, logs, constants.logVerbosity);
    }

    // 換気計算
    if (constants.pressureCalc) {
      for (int pressureAttempt = 0; pressureAttempt < 2; ++pressureAttempt) {
        auto pressureConstants = constants;
        if (pressureAttempt != 0) pressureConstants.ventilationTolerance *= 0.1;
        std::unique_ptr<ScopedLogSection> pressureScope;
        if (logEnabled) pressureScope = std::make_unique<ScopedLogSection>(logs, "圧力計算");
        PressureSolveResult pressureResult;
        {
            ScopedTimer timer(timings, "pressure_solve_iteration", meta);
            const auto t0 = std::chrono::steady_clock::now();
            pressureResult = ventNetwork.solvePressureDetailed(pressureConstants, logs);
            if (metrics) {
                const auto t1 = std::chrono::steady_clock::now();
                metrics->pressureMs +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                ++metrics->pressureSolveCalls;
                metrics->pressureCeresIterations +=
                    static_cast<std::size_t>(std::max(0, pressureResult.ceresIterations));
                if (pressureResult.usedFallback) ++metrics->pressureFallbackCount;
            }
        }
        step.pressureMap = std::move(pressureResult.pressures);
        step.flowRates = std::move(pressureResult.flows);
        step.flowBalance = std::move(pressureResult.balances);

        // 不採用解はグラフへ反映せず、熱計算にも進まない（ThermalNotConverged で隠さない）
        if (!pressureResult.accepted) {
            throw simulation::Error(
                simulation::ErrorCode::PressureNotConverged,
                std::string("Pressure solver did not accept solution during coupled step (method=") +
                    (pressureResult.method.empty() ? "none" : pressureResult.method) + ")");
        }
        ventNetwork.applySolveResults(step.pressureMap, step.flowRates);
        auto& graph = ventNetwork.getGraph();
        bool snapped = false;
        // A pressure-unknown dead end with one remaining passive branch has
        // exactly zero flow by conservation (e.g. a stopped fan's discharge).
        // Ceres' absolute stopping tolerance can otherwise leave a tiny
        // one-way inflow and make a zero-capacity moisture row singular.
        std::vector<Edge> activeEdges;
        std::vector<std::vector<std::size_t>> incident(boost::num_vertices(graph));
        std::vector<std::size_t> degree(incident.size(), 0);
        std::vector<Vertex> pending;
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const auto& edge = graph[e];
            if (!edge.current_enabled ||
                (FlowCalculation::isFixedVolumeFlowEdge(edge) && edge.current_vol == 0.0) ||
                (edge.type == "fan" && edge.q_max == 0.0 && edge.q1 == 0.0)) continue;
            const auto index = activeEdges.size();
            activeEdges.push_back(e);
            for (auto v : {boost::source(e, graph), boost::target(e, graph)}) {
                incident[v].push_back(index);
                ++degree[v];
            }
        }
        std::vector<bool> removed(activeEdges.size(), false);
        for (auto v : boost::make_iterator_range(boost::vertices(graph)))
            if (degree[v] == 1) pending.push_back(v);
        for (std::size_t i = 0; i < pending.size(); ++i) {
            const auto v = pending[i];
            if (!graph[v].calc_p || degree[v] != 1) continue;
            for (auto index : incident[v]) {
                if (removed[index]) continue;
                const auto e = activeEdges[index];
                auto& edge = graph[e];
                if (edge.type == "fan" || FlowCalculation::isFixedVolumeFlowEdge(edge) ||
                    std::abs(edge.flow_rate) > constants.ventilationTolerance) continue;
                snapped = snapped || edge.flow_rate != 0.0;
                edge.flow_rate = 0.0;
                removed[index] = true;
                for (auto endpoint : {boost::source(e, graph), boost::target(e, graph)})
                    if (--degree[endpoint] == 1) pending.push_back(endpoint);
            }
        }
        if (snapped) {
            // Per-edge values are authoritative (parallel branches are distinct).
            // Recheck conservation rather than accepting a modified pressure solution blindly.
            step.flowRates = ventNetwork.collectFlowRateMap();
            for (auto& balance : step.flowBalance) balance.second = 0.0;
            for (auto e : boost::make_iterator_range(boost::edges(graph))) {
                step.flowBalance[graph[boost::source(e, graph)].key] -= graph[e].flow_rate;
                step.flowBalance[graph[boost::target(e, graph)].key] += graph[e].flow_rate;
            }
            std::string failedNode;
            for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
                if (graph[v].calc_p && std::abs(step.flowBalance[graph[v].key]) > constants.ventilationTolerance) {
                    failedNode = graph[v].key;
                    break;
                }
            }
            if (!failedNode.empty()) {
                if (pressureAttempt == 0) {
                    writeDomainLog(logs, "圧力", "停止流路の収支再検査により高精度再求解: " + failedNode);
                    continue;
                }
                throw simulation::Error(simulation::ErrorCode::PressureNotConverged,
                    "Flow balance failed after normalizing numerical zero flows: node=" + failedNode +
                    ", balance_m3s=" + std::to_string(step.flowBalance[failedNode]));
            }
        }
        break;
      }
    }

    // 熱計算
    if (constants.temperatureCalc) {
        // pressureCalc=false の場合でも fixed_flow 等で flow_rate が入るため、移流用に同期する
        // pressureCalc=true の場合も換気計算結果を熱回路網に同期する
        thermalNetwork.syncFlowRatesFromVentilationNetwork(ventNetwork);
        std::unique_ptr<ScopedLogSection> thermalScope;
        if (logEnabled) thermalScope = std::make_unique<ScopedLogSection>(logs, "熱計算");
        {
            ScopedTimer timer(timings, "thermal_solve_iteration", meta);
            const auto statsBefore =
                ThermalSolverLinearDirect::getDirectTCacheStats(thermalNetwork.directTContext());
            const auto t0 = std::chrono::steady_clock::now();
            thermalNetwork.solveTemperature(constants, logs);
            if (metrics) {
                const auto t1 = std::chrono::steady_clock::now();
                metrics->thermalMs +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                const auto statsAfter =
                    ThermalSolverLinearDirect::getDirectTCacheStats(thermalNetwork.directTContext());
                metrics->thermalRhsOnlyBuilds +=
                    static_cast<std::size_t>(statsAfter.rhsOnlyBuild - statsBefore.rhsOnlyBuild);
                metrics->thermalFullBuilds +=
                    static_cast<std::size_t>(statsAfter.fullBuild - statsBefore.fullBuild);
            }
        }
        // 内側連成を古い温度で続けないよう、熱ソルバ失敗/未収束はここで打ち切る。
        if (!thermalNetwork.getLastThermalConverged()) {
            throw simulation::Error(
                simulation::ErrorCode::ThermalNotConverged,
                std::string("Thermal solver did not converge during coupled step (method=") +
                    thermalNetwork.getLastThermalMethod() + ")");
        }

        // pressureCalc=false の場合、換気側で温度（密度）を参照する計算が走らないため更新不要
        if (constants.pressureCalc) {
            ventNetwork.syncTemperaturesFromThermalNetwork(thermalNetwork);
        }
    }

    return step;
}
