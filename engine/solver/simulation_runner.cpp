#include "simulation_runner.h"

#include "aircon/aircon_controller.h"
#include "core/thermal/thermal_linear_utils.h"
#include "core/thermal/thermal_solver_linear_direct.h"
#include "core/humidity/humidity_coupling.h"
#include "network/contaminant_network.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_aircon_iteration.h"
#include "simulation_context.h"
#include "simulation_coupled_step.h"
#include "simulation_coupling_control.h"
#include "simulation_error.h"
#include "simulation_inner_coupling.h"
#include "simulation_metrics.h"
#include "simulation_runner_helpers.h"
#include "simulation_timestep_result.h"
#include "simulation_timestep_state.h"
#include "transport/concentration_solver.h"
#include "utils/utils.h"

#include <boost/range/iterator_range.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace simulation::detail;

// 空調ノードの ON/OFF・mode 署名（外側ウォームスタート無効化判定）
std::uint64_t airconStateSignature(const ThermalNetwork& thermal) {
    using thermal_linear_utils::fnv1a64_update;
    std::uint64_t h = 0;
    const auto& g = thermal.getGraph();
    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        const auto& n = g[v];
        if (n.getTypeCode() != VertexProperties::TypeCode::Aircon) continue;
        h = fnv1a64_update(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
        h = fnv1a64_update(h, n.on ? 1u : 0u);
        // mode 文字列の簡易ハッシュ
        for (unsigned char c : n.current_mode) {
            h = fnv1a64_update(h, static_cast<std::uint64_t>(c));
        }
    }
    return h;
}

// One acceptance gate for the state that will be exported. A small update or
// an earlier successful sub-solve is not a substitute for these residuals.
bool acceptedStateConverged(simulation::Context& ctx,
                             const TimestepInitialState& initial,
                             const SeparatedHeatSources& heatSources,
                             const FlowRateMap& flows) {
    if (ctx.constants.pressureCalc && !ctx.ventilation.getLastPressureConverged()) return false;
    auto& graph = ctx.thermal.getGraph();
    if (ctx.constants.temperatureCalc) {
        if (anyUnphysicalCalcTemperature(graph)) return false;
        if (!ThermalSolverLinearDirect::validateCurrentState(ctx.thermal, ctx.constants, ctx.logs)) return false;
    }
    double waterDefect = 0.0;
    double waterRelativeResidual = 0.0;
    MoistureBalanceTerms water;
    if (ctx.constants.humidityCalc) {
        HumidityNetworkTerms terms;
        ctx.humidity.buildTerms(static_cast<const ThermalNetwork&>(ctx.thermal).nodeStateView(),
                                ctx.ventilation, terms);
        core::humidity::evaluateMoistureBalanceTerms(
            graph, terms, initial.humidityX, ctx.constants.timestep, water);
        const auto& linear = ctx.humidity.humiditySolverContext();
        if (!terms.updateVertices.empty() &&
            (!linear.factorized || terms.updateVertices != linear.updateVertices)) return false;
        double residualSquared = 0.0;
        for (auto v : terms.updateVertices) {
            if (couplingHumidityUnphysical(graph[v].current_x)) return false;
            const size_t i = static_cast<size_t>(v);
            const double capacity = graph[v].moisture_capacity > 0.0
                ? graph[v].moisture_capacity : PhysicalConstants::DENSITY_DRY_AIR * graph[v].v;
            double diagonal = capacity / ctx.constants.timestep + terms.outSum[i];
            for (const auto& link : terms.moistureLinks[i]) diagonal += link.second;
            const double defect = std::abs(water.residual[i]) / std::max(diagonal, 1e-30);
            if (!std::isfinite(defect)) return false;
            waterDefect = std::max(waterDefect, defect);
            // Match the humidity solver's row scaling: inventory rows use
            // dt/C, algebraic rows remain in kg/s. Evaluate the current state,
            // not the pre-relaxation vector that the linear solver checked.
            const double scaled = water.residual[i] *
                (capacity > 0.0 ? ctx.constants.timestep / capacity : 1.0);
            residualSquared += scaled * scaled;
        }
        ctx.humidity.setLastMoistureBalance(water);
        // This is an absolute humidity defect; the linear solver tolerance is
        // a dimensionless relative residual and must not be mixed with it.
        const double tolerance = couplingHumidityTol(ctx.constants);
        if (waterDefect > tolerance) {
            writeDomainLog(ctx.logs, "受入検査", "水分収支未達: normalized=" + std::to_string(waterDefect));
            return false;
        }
        if (!terms.updateVertices.empty()) {
            const double rhsNorm = linear.rhs.norm();
            waterRelativeResidual = std::sqrt(residualSquared);
            if (rhsNorm > 0.0) waterRelativeResidual /= rhsNorm;
            const double linearTolerance = ctx.constants.humiditySolverTolerance > 0.0
                ? ctx.constants.humiditySolverTolerance : 1e-9;
            if (!std::isfinite(waterRelativeResidual) || waterRelativeResidual > linearTolerance) {
                std::ostringstream message;
                message << "最終水分収支未達: relative=" << waterRelativeResidual
                        << ", tolerance=" << linearTolerance;
                writeDomainLog(ctx.logs, "受入検査", message.str());
                return false;
            }
        }
    }
    double latentDefect = 0.0;
    if (simulation::latentCouplingActive(ctx.constants)) {
        auto mixed = heatSources.humidityLatent;
        std::vector<double> raw;
        if (simulation::latentCouplingModeFromConstants(ctx.constants) == simulation::LatentCouplingMode::FromPhaseChange) {
            updateLatentFromPhaseChange(graph, water, 1.0, mixed, &raw);
        } else {
            updateLatentFromHumidityChange(graph, initial.humidityX, ctx.constants.timestep, 1.0, mixed, &raw);
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            const double defect = std::abs(raw[i] - heatSources.humidityLatent[i]);
            const double tolerance = ctx.constants.couplingLatentAbsoluteToleranceW +
                ctx.constants.couplingLatentRelativeTolerance *
                std::max(std::abs(raw[i]), std::abs(heatSources.humidityLatent[i]));
            if (!std::isfinite(defect) || defect > tolerance) return false;
            latentDefect = std::max(latentDefect, defect);
        }
    }
    if (ctx.constants.temperatureCalc && ctx.constants.humidityCalc &&
        !ctx.aircon.acceptedHumidityConverged(ctx.thermal, flows, couplingHumidityTol(ctx.constants), ctx.logs)) return false;
    if (ctx.constants.temperatureCalc) ctx.aircon.validateAcceptedCapacity(ctx.thermal, flows);
    if (ctx.constants.logVerbosity > 0) {
        std::ostringstream message;
        message << "最終状態整合: thermal_max_W=" << ctx.thermal.getLastThermalMaxBalance()
                << ", humidity_defect_kgkg=" << waterDefect << ", latent_defect_W=" << latentDefect
                << ", humidity_relative_residual=" << waterRelativeResidual;
        writeDomainLog(ctx.logs, "受入検査", message.str());
    }
    return true;
}
} // namespace

void runSimulation(VentilationNetwork& ventNetwork,
                   ThermalNetwork& thermalNetwork,
                   HumidityNetwork& humidityNetwork,
                   ContaminantNetwork& contaminantNetwork,
                   AirconController& airconController,
                   const SimulationConstants& constants,
                   TimestepResult& timestepResultOut,
                   std::ostream& logs,
                   TimingList& timings,
                   const std::string& meta,
                   simulation::TimestepSolveMetrics* metricsIn) {
    simulation::TimestepSolveMetrics localMetrics;
    simulation::TimestepSolveMetrics* metrics = metricsIn ? metricsIn : &localMetrics;
    metrics->reset();

    simulation::Context ctx{
        ventNetwork,
        thermalNetwork,
        humidityNetwork,
        contaminantNetwork,
        airconController,
        constants,
        logs,
        timings,
        meta,
        metrics,
    };

    const bool logEnabled = (ctx.constants.logVerbosity > 0);
    int totalIterations = 0;
    CoupledStepData step;

    const TimestepInitialState initial =
        captureTimestepInitialState(ctx.thermal, ctx.constants.humidityCalc);

    if (ctx.constants.moistEnthalpyEnabled && ctx.constants.humidityCalc) {
        ctx.thermal.setMoistEnthalpyHumidityXn(initial.humidityX);
    } else {
        ctx.thermal.clearMoistEnthalpyHumidityXn();
    }
    ctx.aircon.setMoistEnthalpyEnabled(ctx.constants.moistEnthalpyEnabled);

    SeparatedHeatSources heatSources;
    ensureHeatSourceVectors(heatSources,
                            static_cast<size_t>(boost::num_vertices(ctx.thermal.getGraph())));
    // タイムステップ開始時の heat_source を scheduled として保持
    captureScheduledHeatSources(ctx.thermal.getGraph(), heatSources);
    std::fill(heatSources.airconSensible.begin(), heatSources.airconSensible.end(), 0.0);

    // 同一 ThermalNetwork に保持した前ステップ潜熱を latent(0) として引き継ぐ
    if (simulation::latentCouplingActive(ctx.constants) &&
        ctx.thermal.carriedHumidityLatent().size() == heatSources.humidityLatent.size()) {
        heatSources.humidityLatent = ctx.thermal.carriedHumidityLatent();
    } else {
        std::fill(heatSources.humidityLatent.begin(), heatSources.humidityLatent.end(), 0.0);
    }

    auto innerCtx = simulation::makeInnerCouplingContext(ctx);

    const std::uint64_t airconSigBefore = airconStateSignature(ctx.thermal);
    const bool forceMinTwo = ctx.aircon.shouldForceMinTwoCouplingIters(airconSigBefore);
    // 能力ブラケットは室温・風量・潜熱などに依存するため毎タイムステップ初期化
    ctx.aircon.clearCapacityLimitBracket();

    const std::size_t maxOuter = effectiveMaxAirconControlIterations(ctx.constants);
    bool outerLoopConverged = false;
    bool stateAcceptanceRejected = false;
    for (std::size_t iteration = 0; iteration < maxOuter; ++iteration) {
        stateAcceptanceRejected = false;
        if (metrics) metrics->outerIterations = iteration + 1;
        // 外側反復開始時: 空調顕熱をクリア（scheduled / latent は維持）
        std::fill(heatSources.airconSensible.begin(), heatSources.airconSensible.end(), 0.0);
        composeHeatSourcesIntoGraph(ctx.thermal.getGraph(), heatSources);

        const int loopIndex1Based = simulation::toLogIndex1Based(iteration);
        const std::string loopLabel =
            "圧力-温度連成計算-エアコン制御ループ " + std::to_string(loopIndex1Based) + ":";
        {
            ScopedLogSection coupledScope(ctx.logs, loopLabel);

            // 湿度連成の前に空調吹出境界を整える（current_x=0 固定境界による室内乾燥を防ぐ）
            ctx.aircon.syncHumidityBoundariesBeforeSolve(ctx.thermal);

            runInnerCoupling(innerCtx, logEnabled, iteration, initial, step, totalIterations,
                             heatSources, forceMinTwo && iteration == 0);

            if (!ctx.constants.pressureCalc) {
                step.flowRates = ctx.ventilation.collectFlowRateMap();
            }

            runDecoupledHumidityStep(innerCtx, initial, step, iteration, &heatSources);

            const std::string airconMeta =
                simulation::appendLoopMeta(ctx.meta, loopIndex1Based);
            auto airconCtx = simulation::makeAirconIterationContext(ctx, airconMeta);
            const auto airconAction =
                runAirconIteration(airconCtx, step.flowRates, totalIterations);

            if (airconAction != simulation::AirconIterationAction::Accept) {
                logAirconRecompute(ctx.logs, logEnabled);
                continue;
            }

            stateAcceptanceRejected = !acceptedStateConverged(ctx, initial, heatSources, step.flowRates);
            if (stateAcceptanceRejected) continue;

            if (!ctx.thermal.getLastThermalConverged()) {
                logThermalNotConverged(ctx.logs,
                                       logEnabled,
                                       ctx.thermal.getLastThermalMethod(),
                                       ctx.thermal.getLastThermalRmseBalance(),
                                       ctx.thermal.getLastThermalMaxBalance(),
                                       loopIndex1Based);
                throw simulation::Error(
                    simulation::ErrorCode::ThermalNotConverged,
                    "Thermal solver did not converge: stopping to avoid infinite loop");
            }

            outerLoopConverged = true;
            logOuterLoopConverged(ctx.logs, logEnabled, loopIndex1Based);
            break;
        }
    }

    if (!outerLoopConverged && stateAcceptanceRejected) {
        throw simulation::Error(simulation::ErrorCode::CouplingMaxIterations,
            "Final coupled state did not satisfy thermal/moisture/latent acceptance criteria within the iteration limit");
    }
    ensureOuterAirconLoopConverged(outerLoopConverged);

    if (simulation::latentCouplingActive(ctx.constants)) {
        ctx.thermal.setCarriedHumidityLatent(heatSources.humidityLatent);
    } else {
        ctx.thermal.clearCarriedHumidityLatent();
    }

    ctx.aircon.observeAirconStateSignature(airconStateSignature(ctx.thermal));

    if (ctx.constants.temperatureCalc) {
        ctx.thermal.commitResponseConductionHistory();
    }

    const auto sharedNodeState = makeSharedNodeStateArgs(ctx.thermal);
    const auto concStats = transport::updateConcentrationIfEnabled(ctx.constants,
                                            ctx.ventilation,
                                            sharedNodeState.nodeGraph,
                                            sharedNodeState.nodeState,
                                            ctx.contaminant,
                                            ctx.logs,
                                            ctx.timings,
                                            std::string(ctx.meta));
    if (!concStats.converged) {
        throw simulation::Error(
            simulation::ErrorCode::ConcentrationNotConverged,
            "Concentration transport solve failed (singular, non-finite or negative solution)");
    }

    buildTimestepResult(ctx.constants,
                        ctx.ventilation,
                        ctx.thermal,
                        ctx.humidity,
                        ctx.contaminant,
                        ctx.aircon,
                        step.flowRates,
                        ctx.logs,
                        timestepResultOut);

    logTimestepFinished(ctx.logs, logEnabled, totalIterations);
    if (metrics) {
        metrics->finalizeTimestepOuterStats();
    }
}
