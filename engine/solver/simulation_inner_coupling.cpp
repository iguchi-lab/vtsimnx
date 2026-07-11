#include "simulation_inner_coupling.h"

#include "core/humidity/humidity_solver.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_coupled_step.h"
#include "simulation_coupling_control.h"
#include "simulation_error.h"
#include "simulation_runner_helpers.h"
#include "utils/utils.h"

#include <memory>
#include <string>

namespace simulation {
namespace {

using detail::CouplingSnapshot;
using detail::InnerCouplingAction;
using detail::InnerCouplingEval;
using detail::calculateHumidityChangeByVertex;
using detail::calculateMaxAbsDiff;
using detail::calculateTemperatureChangeByVertex;
using detail::captureCouplingPrevState;
using detail::captureHeatSourceByVertex;
using detail::evaluateInnerCoupling;
using detail::humidityCouplingActive;
using detail::logHumiditySolverNotConverged;
using detail::logInnerCouplingConverged;
using detail::logInnerCouplingDelta;
using detail::logInnerCouplingMaxIteration;
using detail::logInnerCouplingNotNeeded;
using detail::logPressureFallbackStop;
using detail::makeSharedNodeStateArgs;
using detail::relaxHumidityByVertex;
using detail::restoreHeatSourceByVertex;
using detail::restoreWPrevToGraph;
using detail::restoreXPrevToGraph;
using detail::CoupledDelta;

// 現状は Disabled 固定（方針B）。FeedbackToThermal は未実装。
constexpr LatentCouplingMode kLatentCouplingMode = LatentCouplingMode::Disabled;

} // namespace

void runDecoupledHumidityStep(InnerCouplingContext& ctx,
                              const detail::TimestepInitialState& initial,
                              CoupledStepData& step,
                              int outerIteration) {
    if (!(ctx.constants.humidityCalc && !ctx.constants.moistureCouplingEnabled)) {
        return;
    }
    const auto sharedNodeState = makeSharedNodeStateArgs(ctx.thermal);
    restoreXPrevToGraph(sharedNodeState.nodeGraph, ctx.ventilation, initial.humidityX);
    restoreWPrevToGraph(sharedNodeState.nodeGraph, initial.moistureW);
    (void)core::humidity::updateHumidityIfEnabled(
        ctx.constants,
        ctx.ventilation,
        sharedNodeState.nodeGraph,
        sharedNodeState.nodeState,
        ctx.humidity,
        step.flowRates,
        ctx.logs,
        ctx.timings,
        std::string(ctx.meta) + ",iteration=" + std::to_string(outerIteration + 1));
}

void runInnerCoupling(InnerCouplingContext& ctx,
                      bool logEnabled,
                      int outerIteration,
                      const detail::TimestepInitialState& initial,
                      CoupledStepData& step,
                      int& totalIterations) {
    CouplingSnapshot snap;
    double lastLatentAppliedW = 0.0;
    core::humidity::HumiditySolveStats lastHumiditySolveStats{};
    int coupledIter = 0;
    const std::string meta(ctx.meta);

    while (true) {
        ++coupledIter;
        ++totalIterations;
        const bool humidityActive = humidityCouplingActive(ctx.constants);
        if (coupledIter == 1) {
            captureHeatSourceByVertex(ctx.thermal.getGraph(), snap.heatSource);
        }

        captureCouplingPrevState(snap, ctx.ventilation, ctx.thermal, ctx.constants, humidityActive);

        std::unique_ptr<ScopedLogSection> iterScope;
        if (logEnabled) {
            iterScope = std::make_unique<ScopedLogSection>(
                ctx.logs,
                "空気-熱-湿気 連成反復 " + std::to_string(coupledIter) + ":");
        }

        {
            ScopedTimer timer(ctx.timings, "performCoupledCalculation",
                              meta + ",iteration=" + std::to_string(outerIteration + 1));
            step = performCoupledStepCalculation(ctx.ventilation, ctx.thermal, ctx.constants,
                                                 ctx.logs, ctx.timings,
                                                 meta + ",iteration=" + std::to_string(outerIteration + 1));
        }
        if (!ctx.constants.pressureCalc) {
            step.flowRates = ctx.ventilation.collectFlowRateMap();
        }

        if (humidityActive) {
            const auto sharedNodeState = makeSharedNodeStateArgs(ctx.thermal);
            restoreXPrevToGraph(sharedNodeState.nodeGraph, ctx.ventilation, initial.humidityX);
            restoreWPrevToGraph(sharedNodeState.nodeGraph, initial.moistureW);
            lastHumiditySolveStats = core::humidity::updateHumidityIfEnabled(
                ctx.constants,
                ctx.ventilation,
                sharedNodeState.nodeGraph,
                sharedNodeState.nodeState,
                ctx.humidity,
                step.flowRates, ctx.logs, ctx.timings,
                meta + ",iteration=" + std::to_string(outerIteration + 1) +
                    ",coupledIter=" + std::to_string(coupledIter));
            logHumiditySolverNotConverged(ctx.logs, logEnabled, lastHumiditySolveStats);
            relaxHumidityByVertex(ctx.thermal.getGraph(), ctx.ventilation, snap.humidity,
                                  ctx.constants.humidityRelaxation);
        }

        restoreHeatSourceByVertex(ctx.thermal.getGraph(), snap.heatSource);
        const double latentAppliedThisIter = resolveLatentAppliedThisIter(kLatentCouplingMode);
        lastLatentAppliedW = latentAppliedThisIter;

        CoupledDelta delta{};
        if (ctx.constants.pressureCalc) {
            delta.pressureChange =
                calculateMaxAbsDiff(snap.pressure, ctx.ventilation.collectPressureValues());
        }
        if (ctx.constants.temperatureCalc) {
            delta.temperatureChange =
                calculateTemperatureChangeByVertex(ctx.thermal.getGraph(), snap.temperature);
        }
        if (humidityActive) {
            delta.humidityChange =
                calculateHumidityChangeByVertex(ctx.thermal.getGraph(), snap.humidity);
        }

        const InnerCouplingEval eval = evaluateInnerCoupling(
            ctx.constants,
            humidityActive,
            coupledIter,
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

        logInnerCouplingDelta(ctx.logs, logEnabled, delta, latentAppliedThisIter, lastHumiditySolveStats);

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
