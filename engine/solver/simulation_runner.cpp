#include "simulation_runner.h"

#include "aircon/aircon_controller.h"
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
#include "simulation_runner_helpers.h"
#include "simulation_timestep_result.h"
#include "simulation_timestep_state.h"
#include "transport/concentration_solver.h"
#include "utils/utils.h"

#include <cstddef>
#include <string>

namespace {
using namespace simulation::detail;
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
                   const std::string& meta) {
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
    };

    const bool logEnabled = (ctx.constants.logVerbosity > 0);
    // 内側連成の累積回数（空調容量調整 API 互換のため int のまま保持）
    int totalIterations = 0;
    CoupledStepData step;

    const TimestepInitialState initial =
        captureTimestepInitialState(ctx.thermal, ctx.constants.humidityCalc);

    auto innerCtx = simulation::makeInnerCouplingContext(ctx);

    const std::size_t maxOuter = effectiveMaxAirconControlIterations(ctx.constants);
    bool outerLoopConverged = false;
    for (std::size_t iteration = 0; iteration < maxOuter; ++iteration) {
        if (iteration == 0) {
            ctx.aircon.clearCapacityLimitBracket();
        }
        // 外側反復開始時: 前反復の空調 heat_source をクリアしてから連成を始める。
        resetNodeHeatSources(ctx.thermal.getGraph());

        const int loopIndex1Based = simulation::toLogIndex1Based(iteration);
        const std::string loopLabel =
            "圧力-温度連成計算-エアコン制御ループ " + std::to_string(loopIndex1Based) + ":";
        {
            ScopedLogSection coupledScope(ctx.logs, loopLabel);

            runInnerCoupling(innerCtx, logEnabled, iteration, initial, step, totalIterations);

            // pressureCalc=false でも aircon が流量を参照できるよう FlowRateMap を同期
            if (!ctx.constants.pressureCalc) {
                step.flowRates = ctx.ventilation.collectFlowRateMap();
            }

            runDecoupledHumidityStep(innerCtx, initial, step, iteration);

            const std::string airconMeta =
                std::string(ctx.meta) + ",iteration=" + std::to_string(loopIndex1Based);
            auto airconCtx = simulation::makeAirconIterationContext(ctx, airconMeta);
            const auto airconAction =
                runAirconIteration(airconCtx, step.flowRates, totalIterations);

            if (airconAction != simulation::AirconIterationAction::Accept) {
                logAirconRecompute(ctx.logs, logEnabled);
                continue;
            }

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

    ensureOuterAirconLoopConverged(outerLoopConverged);

    // 応答係数履歴はタイムステップ受理後に一度だけ進める（内側連成・空調反復では進めない）
    if (ctx.constants.temperatureCalc) {
        ctx.thermal.commitResponseConductionHistory();
    }

    // 濃度（c）更新：外側空調ループ収束後のみ（エアコン制御には影響しない想定）
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
            "Concentration solver produced non-finite or negative values");
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
}
