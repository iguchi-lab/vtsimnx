#include "simulation_inner_coupling.h"

#include "core/humidity/humidity_coupling.h"
#include "core/humidity/humidity_solver.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_coupled_step.h"
#include "simulation_coupling_control.h"
#include "simulation_error.h"
#include "simulation_metrics.h"
#include "simulation_runner_helpers.h"
#include "utils/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/range/iterator_range.hpp>

namespace simulation {
namespace {

using detail::CouplingSnapshot;
using detail::InnerCouplingAction;
using detail::InnerCouplingEval;
using detail::SeparatedHeatSources;
using detail::calculateHumidityChangeByVertex;
using detail::calculateMaxAbsDiff;
using detail::calculateTemperatureChangeByVertex;
using detail::captureCouplingPrevState;
using detail::composeHeatSourcesIntoGraph;
using detail::evaluateInnerCoupling;
using detail::humidityCouplingActive;
using detail::logHumiditySolverNotConverged;
using detail::logInnerCouplingConverged;
using detail::logInnerCouplingDelta;
using detail::logInnerCouplingMaxIteration;
using detail::logInnerCouplingNotNeeded;
using detail::logPressureFallbackStop;
using detail::makeSharedNodeStateArgs;
using detail::maxAbsLatentHeatChange;
using detail::relaxHumidityByVertex;
using detail::relaxTemperatureByVertex;
using detail::anyUnphysicalCalcTemperature;
using detail::anyUnphysicalCalcHumidity;
using detail::couplingTemperatureRelaxation;
using detail::couplingTemperatureTol;
using detail::needsInnerCoupledIteration;
using detail::restoreWPrevToGraph;
using detail::updateLatentFromHumidityChange;
using detail::updateLatentFromPhaseChange;
using detail::CoupledDelta;

std::string invalidStateDetail(const Graph& graph, bool temperature) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const auto& node = graph[v];
        const double value = temperature ? node.current_t : node.current_x;
        if ((temperature ? node.calc_t : node.calc_x) &&
            (temperature ? detail::couplingTemperatureUnphysical(value)
                         : detail::couplingHumidityUnphysical(value))) {
            std::ostringstream message;
            message << "node=" << node.key << ", "
                    << (temperature ? "T_C=" : "x_kg_kg=") << value;
            return message.str();
        }
    }
    return "unknown node";
}

double maxAbsVector(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

double sumAbs(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += std::abs(x);
    return s;
}

double applyLatentHeatSources(LatentCouplingMode mode,
                              InnerCouplingContext& ctx,
                              SeparatedHeatSources& heatSources,
                              const std::vector<double>& xN,
                              double dt,
                              double latentRelaxation) {
    if (mode == LatentCouplingMode::Disabled) {
        std::fill(heatSources.humidityLatent.begin(), heatSources.humidityLatent.end(), 0.0);
        composeHeatSourcesIntoGraph(ctx.thermal.getGraph(), heatSources);
        return 0.0;
    }
    if (mode == LatentCouplingMode::FromPhaseChange) {
        HumidityNetworkTerms terms;
        MoistureBalanceTerms bal;
        const auto shared = makeSharedNodeStateArgs(ctx.thermal);
        ctx.humidity.buildTerms(shared.nodeState, ctx.ventilation, terms);
        core::humidity::evaluateMoistureBalanceTerms(
            ctx.thermal.getGraph(), terms, xN, dt, bal);
        ctx.humidity.setLastMoistureBalance(bal);
        updateLatentFromPhaseChange(
            ctx.thermal.getGraph(), bal, latentRelaxation, heatSources.humidityLatent);
    } else {
        updateLatentFromHumidityChange(
            ctx.thermal.getGraph(), xN, dt, latentRelaxation, heatSources.humidityLatent);
    }
    composeHeatSourcesIntoGraph(ctx.thermal.getGraph(), heatSources);
    return sumAbs(heatSources.humidityLatent);
}

} // namespace

void runDecoupledHumidityStep(InnerCouplingContext& ctx,
                              const detail::TimestepInitialState& initial,
                              CoupledStepData& step,
                              std::size_t outerIteration,
                              detail::SeparatedHeatSources* heatSources) {
    if (!(ctx.constants.humidityCalc && !ctx.constants.moistureCouplingEnabled)) {
        return;
    }
    const auto sharedNodeState = makeSharedNodeStateArgs(ctx.thermal);
    // 非連成: 未知ノードだけを x_n に戻してから解く（xN=nullptr → ソルバが現グラフを x_n に採用）。
    // calc_x=false の固定境界（空調吹出 supplyX など）は外側反復の更新を保持する。
    detail::restoreUnknownHumidityStateToGraph(
        sharedNodeState.nodeGraph, ctx.ventilation, initial.humidityX);
    restoreWPrevToGraph(sharedNodeState.nodeGraph, initial.moistureW);
    const int outerLog = toLogIndex1Based(outerIteration);
    const std::string humMeta = appendLoopMeta(ctx.meta, outerLog);
    const auto t0 = std::chrono::steady_clock::now();
    const auto humStats = core::humidity::updateHumidityIfEnabled(
        ctx.constants,
        ctx.ventilation,
        sharedNodeState.nodeGraph,
        sharedNodeState.nodeState,
        ctx.humidity,
        step.flowRates,
        ctx.logs,
        ctx.timings,
        humMeta,
        /*xN=*/nullptr,
        ctx.metrics);
    if (ctx.metrics) {
        ctx.metrics->humidityMs +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (!humStats.converged) {
        throw Error(
            ErrorCode::HumidityNotConverged,
            "Humidity solver did not converge during decoupled humidity step");
    }

    if (anyUnphysicalCalcHumidity(ctx.thermal.getGraph())) {
        throw Error(ErrorCode::HumidityNotConverged,
                    "Unphysical humidity candidate rejected; " +
                    invalidStateDetail(ctx.thermal.getGraph(), false));
    }

    if (heatSources != nullptr && latentCouplingActive(ctx.constants)) {
        const size_t nV = static_cast<size_t>(boost::num_vertices(ctx.thermal.getGraph()));
        detail::ensureHeatSourceVectors(*heatSources, nV);
        applyLatentHeatSources(
            latentCouplingModeFromConstants(ctx.constants),
            ctx,
            *heatSources,
            initial.humidityX,
            static_cast<double>(ctx.constants.timestep),
            ctx.constants.latentRelaxation);
    }
}

void runInnerCoupling(InnerCouplingContext& ctx,
                      bool logEnabled,
                      std::size_t outerIteration,
                      const detail::TimestepInitialState& initial,
                      CoupledStepData& step,
                      int& totalIterations,
                      SeparatedHeatSources& heatSources,
                      bool forceMinTwoCouplingIters) {
    CouplingSnapshot snap;
    double lastLatentAppliedW = 0.0;
    core::humidity::HumiditySolveStats lastHumiditySolveStats{};
    std::size_t coupledIter = 0;
    const std::string meta(ctx.meta);
    const int outerLogIndex = toLogIndex1Based(outerIteration);
    auto* metrics = ctx.metrics;
    const bool latentActive = latentCouplingActive(ctx.constants);
    const LatentCouplingMode latentMode = latentCouplingModeFromConstants(ctx.constants);
    const std::size_t minCouplingIters = forceMinTwoCouplingIters ? 2 : 1;
    const double dt = static_cast<double>(ctx.constants.timestep);

    const size_t nV = static_cast<size_t>(boost::num_vertices(ctx.thermal.getGraph()));
    detail::ensureHeatSourceVectors(heatSources, nV);
    double prevRawTemperatureChange = 0.0;

    while (true) {
        ++coupledIter;
        ++totalIterations;
        if (metrics) ++metrics->coupledIterations;
        const bool humidityActive = humidityCouplingActive(ctx.constants);

        // 熱計算前に分離熱源を合成（潜熱は前反復値 = latent(k)）
        composeHeatSourcesIntoGraph(ctx.thermal.getGraph(), heatSources);
        if (coupledIter == 1) {
            // restore 用に scheduled+sensible をキャプチャ（潜熱は別管理）
            snap.heatSource = heatSources.scheduled;
            for (size_t i = 0; i < nV; ++i) {
                if (i < heatSources.airconSensible.size()) {
                    snap.heatSource[i] += heatSources.airconSensible[i];
                }
            }
            snap.latentHeatSource = heatSources.humidityLatent;
        }

        captureCouplingPrevState(snap, ctx.ventilation, ctx.thermal, ctx.constants, humidityActive);
        snap.latentHeatSource = heatSources.humidityLatent;

        std::unique_ptr<ScopedLogSection> iterScope;
        if (logEnabled) {
            iterScope = std::make_unique<ScopedLogSection>(
                ctx.logs,
                "空気-熱-湿気 連成反復 " + std::to_string(coupledIter) + ":");
        }

        const std::string loopMeta =
            appendLoopMeta(meta, outerLogIndex, static_cast<int>(coupledIter));
        {
            ScopedTimer timer(ctx.timings, "performCoupledCalculation", loopMeta);
            step = performCoupledStepCalculation(ctx.ventilation, ctx.thermal, ctx.constants,
                                                 ctx.logs, ctx.timings, loopMeta, metrics);
        }
        if (!ctx.constants.pressureCalc) {
            step.flowRates = ctx.ventilation.collectFlowRateMap();
        }

        double rawTemperatureChange = 0.0;
        double rawHumidityChange = 0.0;
        if (ctx.constants.temperatureCalc) {
            rawTemperatureChange =
                calculateTemperatureChangeByVertex(ctx.thermal.getGraph(), snap.temperature);
            if (anyUnphysicalCalcTemperature(ctx.thermal.getGraph())) {
                throw Error(ErrorCode::ThermalNotConverged,
                            "Unphysical thermal candidate rejected; " +
                            invalidStateDetail(ctx.thermal.getGraph(), true));
            } else if (needsInnerCoupledIteration(ctx.constants)) {
                // 20K 混合は Picard 再反復があるときだけ。温度だけの 1 回解に
                // 掛けると空調 ON 直後のコイル ΔT が切れ、次ステップで除熱になる。
                const double temperatureAlpha = couplingTemperatureRelaxation(
                    ctx.constants.temperatureRelaxation,
                    coupledIter,
                    rawTemperatureChange,
                    prevRawTemperatureChange,
                    couplingTemperatureTol(ctx.constants));
                relaxTemperatureByVertex(ctx.thermal.getGraph(), ctx.ventilation, snap.temperature,
                                         temperatureAlpha);
                if (ctx.constants.pressureCalc) {
                    ctx.ventilation.syncTemperaturesFromThermalNetwork(ctx.thermal);
                }
                if (logEnabled && temperatureAlpha < 1.0) {
                    writeDomainLog(
                        ctx.logs,
                        "連成",
                        "温度緩和: alpha=" + std::to_string(temperatureAlpha) +
                            ", 生温度変化量: " + std::to_string(rawTemperatureChange) + " K");
                }
                prevRawTemperatureChange = rawTemperatureChange;
            } else {
                prevRawTemperatureChange = rawTemperatureChange;
            }
        }

        if (humidityActive) {
            const auto sharedNodeState = makeSharedNodeStateArgs(ctx.thermal);
            restoreWPrevToGraph(sharedNodeState.nodeGraph, initial.moistureW);
            const auto t0 = std::chrono::steady_clock::now();
            lastHumiditySolveStats = core::humidity::updateHumidityIfEnabled(
                ctx.constants,
                ctx.ventilation,
                sharedNodeState.nodeGraph,
                sharedNodeState.nodeState,
                ctx.humidity,
                step.flowRates, ctx.logs, ctx.timings,
                loopMeta,
                &initial.humidityX,
                metrics);
            if (metrics) {
                ctx.metrics->humidityMs +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
            }
            logHumiditySolverNotConverged(ctx.logs, logEnabled, lastHumiditySolveStats);
            if (!lastHumiditySolveStats.converged) {
                throw Error(
                    ErrorCode::HumidityNotConverged,
                    "Humidity solver did not converge during inner coupling");
            }
            rawHumidityChange = calculateHumidityChangeByVertex(ctx.thermal.getGraph(), snap.humidity);
            if (anyUnphysicalCalcHumidity(ctx.thermal.getGraph())) {
                throw Error(ErrorCode::HumidityNotConverged,
                            "Unphysical humidity candidate rejected; " +
                            invalidStateDetail(ctx.thermal.getGraph(), false));
            } else {
                relaxHumidityByVertex(ctx.thermal.getGraph(), ctx.ventilation, snap.humidity,
                                      ctx.constants.humidityRelaxation);
            }
        }

        // 湿気更新＋緩和の直後に latent を更新。湿度未更新の内側パスでは持ち越し値を維持。
        const std::vector<double> latentPrev = heatSources.humidityLatent;
        if (!latentActive) {
            std::fill(heatSources.humidityLatent.begin(), heatSources.humidityLatent.end(), 0.0);
            composeHeatSourcesIntoGraph(ctx.thermal.getGraph(), heatSources);
            lastLatentAppliedW = 0.0;
        } else if (humidityActive) {
            lastLatentAppliedW = applyLatentHeatSources(
                latentMode,
                ctx,
                heatSources,
                initial.humidityX,
                dt,
                ctx.constants.latentRelaxation);
        } else {
            composeHeatSourcesIntoGraph(ctx.thermal.getGraph(), heatSources);
            lastLatentAppliedW = sumAbs(heatSources.humidityLatent);
        }

        CoupledDelta delta{};
        if (ctx.constants.pressureCalc) {
            delta.pressureChange =
                calculateMaxAbsDiff(snap.pressure, ctx.ventilation.collectPressureValues());
        }
        if (ctx.constants.temperatureCalc) {
            delta.temperatureChange =
                std::max(rawTemperatureChange,
                         calculateTemperatureChangeByVertex(ctx.thermal.getGraph(), snap.temperature));
        }
        if (humidityActive) {
            delta.humidityChange =
                std::max(rawHumidityChange,
                         calculateHumidityChangeByVertex(ctx.thermal.getGraph(), snap.humidity));
        }
        if (latentActive) {
            delta.latentHeatChange = maxAbsLatentHeatChange(latentPrev, heatSources.humidityLatent);
            delta.latentHeatScale =
                std::max(maxAbsVector(latentPrev), maxAbsVector(heatSources.humidityLatent));
        }

        const InnerCouplingEval eval = evaluateInnerCoupling(
            ctx.constants,
            humidityActive,
            latentActive,
            coupledIter,
            minCouplingIters,
            delta,
            ctx.ventilation.getLastPressureConverged());

        if (eval.action == InnerCouplingAction::ThrowPressureNonConvergence) {
            logPressureFallbackStop(ctx.logs, logEnabled);
            throw Error(ErrorCode::PressureNotConverged,
                        "Disabled final normal solve: stopping after fallback non-convergence");
        }
        if (eval.action == InnerCouplingAction::BreakNoNeed) {
            logInnerCouplingNotNeeded(ctx.logs, logEnabled);
            break;
        }

        logInnerCouplingDelta(ctx.logs, logEnabled, delta, lastLatentAppliedW, lastHumiditySolveStats);

        if (eval.action == InnerCouplingAction::BreakConverged) {
            logInnerCouplingConverged(ctx.logs, logEnabled, coupledIter);
            break;
        }
        if (eval.action == InnerCouplingAction::ThrowMaxIteration) {
            logInnerCouplingMaxIteration(ctx.logs, logEnabled, coupledIter, eval,
                                         lastLatentAppliedW, lastHumiditySolveStats);
            throw Error(ErrorCode::CouplingMaxIterations,
                        "Maximum iteration count reached: stopping after maximum iteration count");
        }
    }
}

} // namespace simulation
