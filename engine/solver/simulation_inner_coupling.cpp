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

namespace {

using namespace simulation::detail;

constexpr LatentCouplingMode kLatentCouplingMode = LatentCouplingMode::Disabled;

struct SharedNodeStateArgs {
    Graph& nodeGraph;
    ConstNodeStateView nodeState;
};

SharedNodeStateArgs makeSharedNodeStateArgs(ThermalNetwork& thermalNetwork) {
    return SharedNodeStateArgs{
        thermalNetwork.getGraph(),
        static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView(),
    };
}

CoupledDelta computeCoupledDelta(const SimulationConstants& constants,
                                 VentilationNetwork& ventNetwork,
                                 ThermalNetwork& thermalNetwork,
                                 const CouplingSnapshot& snap) {
    CoupledDelta d{};
    if (constants.pressureCalc) {
        d.pressureChange = calculateMaxAbsDiff(snap.pressure, ventNetwork.collectPressureValues());
    }
    if (constants.temperatureCalc) {
        d.temperatureChange = calculateTemperatureChangeByVertex(thermalNetwork.getGraph(), snap.temperature);
    }
    return d;
}

} // namespace

void runInnerCoupling(VentilationNetwork& ventNetwork,
                      ThermalNetwork& thermalNetwork,
                      HumidityNetwork& humidityNetwork,
                      const SimulationConstants& constants,
                      std::ostream& logs,
                      TimingList& timings,
                      const std::string& meta,
                      bool logEnabled,
                      int outerIteration,
                      const TimestepInitialState& initial,
                      CoupledStepData& step,
                      int& totalIterations) {
    CouplingSnapshot snap;
    double lastLatentAppliedW = 0.0;
    core::humidity::HumiditySolveStats lastHumiditySolveStats{};
    int coupledIter = 0;

    while (true) {
        ++coupledIter;
        ++totalIterations;
        const bool humidityActive = humidityCouplingActive(constants);
        if (coupledIter == 1) {
            captureHeatSourceByVertex(thermalNetwork.getGraph(), snap.heatSource);
        }

        captureCouplingPrevState(snap, ventNetwork, thermalNetwork, constants, humidityActive);

        std::unique_ptr<ScopedLogSection> iterScope;
        if (logEnabled) {
            iterScope = std::make_unique<ScopedLogSection>(
                logs,
                "空気-熱-湿気 連成反復 " + std::to_string(coupledIter) + ":");
        }

        {
            ScopedTimer timer(timings, "performCoupledCalculation",
                              meta + ",iteration=" + std::to_string(outerIteration + 1));
            step = performCoupledStepCalculation(ventNetwork, thermalNetwork, constants, logs, timings,
                                                 meta + ",iteration=" + std::to_string(outerIteration + 1));
        }
        if (!constants.pressureCalc) {
            step.flowRates = ventNetwork.collectFlowRateMap();
        }

        if (humidityActive) {
            const auto sharedNodeState = makeSharedNodeStateArgs(thermalNetwork);
            // 同一タイムステップ内反復なので、毎回 x_prev / w_prev に戻して再評価する。
            restoreXPrevToGraph(sharedNodeState.nodeGraph, ventNetwork, initial.humidityX);
            restoreWPrevToGraph(sharedNodeState.nodeGraph, initial.moistureW);
            lastHumiditySolveStats = core::humidity::updateHumidityIfEnabled(
                constants,
                ventNetwork,
                sharedNodeState.nodeGraph,
                sharedNodeState.nodeState,
                humidityNetwork,
                step.flowRates, logs, timings,
                meta + ",iteration=" + std::to_string(outerIteration + 1) +
                           ",coupledIter=" + std::to_string(coupledIter));
            logHumiditySolverNotConverged(logs, logEnabled, lastHumiditySolveStats);
            relaxHumidityByVertex(thermalNetwork.getGraph(), ventNetwork, snap.humidity, constants.humidityRelaxation);
        }

        // 潜熱フィードバック方針（現状は Disabled = 方針B）
        restoreHeatSourceByVertex(thermalNetwork.getGraph(), snap.heatSource);
        const double latentAppliedThisIter =
            (kLatentCouplingMode == LatentCouplingMode::FeedbackToThermal) ? 0.0 : 0.0;
        lastLatentAppliedW = latentAppliedThisIter;

        auto delta = computeCoupledDelta(constants, ventNetwork, thermalNetwork, snap);
        if (humidityActive) {
            delta.humidityChange = calculateHumidityChangeByVertex(thermalNetwork.getGraph(), snap.humidity);
        }

        // evaluateInnerCoupling は maxCouplingIterations（未設定時 maxInnerIteration）を参照
        const InnerCouplingEval eval = evaluateInnerCoupling(
            constants,
            humidityActive,
            coupledIter,
            delta,
            ventNetwork.getLastPressureConverged());

        if (eval.action == InnerCouplingAction::ThrowPressureNonConvergence) {
            logPressureFallbackStop(logs, logEnabled);
            throw SimulationError(
                SimulationErrorCode::PressureNotConverged,
                "Disabled final normal solve: stopping after fallback non-convergence");
        }
        if (eval.action == InnerCouplingAction::BreakNoNeed) {
            logInnerCouplingNotNeeded(logs, logEnabled);
            break;
        }

        logInnerCouplingDelta(logs, logEnabled, delta, latentAppliedThisIter, lastHumiditySolveStats);

        if (eval.action == InnerCouplingAction::BreakConverged) {
            logInnerCouplingConverged(logs, logEnabled, coupledIter);
            break;
        }
        if (eval.action == InnerCouplingAction::ThrowMaxIteration) {
            logInnerCouplingMaxIteration(logs, logEnabled, coupledIter, eval,
                                         lastLatentAppliedW, lastHumiditySolveStats);
            throw SimulationError(
                SimulationErrorCode::CouplingMaxIterations,
                "Maximum iteration count reached: stopping after maximum iteration count");
        }
    }
}
