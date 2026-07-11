#include "simulation_runner.h"

#include "aircon/aircon_controller.h"
#include "core/humidity/humidity_solver.h"
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

#include <boost/range/iterator_range.hpp>
#include <string>

namespace {
using namespace simulation::detail;

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

int maxAirconControlIterations(const SimulationConstants& constants) {
    return static_cast<int>(
        constants.maxAirconControlIterations > 0 ? constants.maxAirconControlIterations
                                                 : constants.maxInnerIteration);
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
                   const std::string& meta) {
    SimulationContext ctx{
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
    int totalIterations = 0;
    CoupledStepData step;

    const TimestepInitialState initial =
        captureTimestepInitialState(ctx.thermal, ctx.constants.humidityCalc);

    const int maxOuter = maxAirconControlIterations(ctx.constants);
    for (int iteration = 0; iteration < maxOuter; ++iteration) {
        if (iteration == 0) {
            ctx.aircon.clearCapacityLimitBracket();
        }
        // タイムステップ内の各外側反復は、熱源を初期化してから開始する。
        for (auto v : boost::make_iterator_range(boost::vertices(ctx.thermal.getGraph()))) {
            ctx.thermal.getGraph()[v].heat_source = 0.0;
        }

        const std::string loopLabel =
            "圧力-温度連成計算-エアコン制御ループ " + std::to_string(iteration + 1) + ":";
        bool loopConverged = false;
        {
            ScopedLogSection coupledScope(ctx.logs, loopLabel);

            runInnerCoupling(ctx.ventilation,
                             ctx.thermal,
                             ctx.humidity,
                             ctx.constants,
                             ctx.logs,
                             ctx.timings,
                             std::string(ctx.meta),
                             logEnabled,
                             iteration,
                             initial,
                             step,
                             totalIterations);

            // pressureCalc=false でも aircon が流量を参照できるよう FlowRateMap を同期
            if (!ctx.constants.pressureCalc) {
                step.flowRates = ctx.ventilation.collectFlowRateMap();
            }

            // 連成OFF時は従来互換: 外側ループごとに1回のみ湿気更新
            if (ctx.constants.humidityCalc && !ctx.constants.moistureCouplingEnabled) {
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
                    std::string(ctx.meta) + ",iteration=" + std::to_string(iteration + 1));
            }

            const auto airconRes = runAirconIteration(
                ctx.aircon,
                ctx.thermal,
                ctx.ventilation,
                ctx.constants,
                step.flowRates,
                ctx.logs,
                totalIterations,
                ctx.timings,
                std::string(ctx.meta) + ",iteration=" + std::to_string(iteration + 1));

            if (airconRes.action != AirconIterationAction::Accept) {
                logAirconRecompute(ctx.logs, logEnabled);
                continue;
            }

            if (!ctx.thermal.getLastThermalConverged()) {
                logThermalNotConverged(ctx.logs,
                                       logEnabled,
                                       ctx.thermal.getLastThermalMethod(),
                                       ctx.thermal.getLastThermalRmseBalance(),
                                       ctx.thermal.getLastThermalMaxBalance(),
                                       iteration + 1);
                throw SimulationError(
                    SimulationErrorCode::ThermalNotConverged,
                    "Thermal solver did not converge: stopping to avoid infinite loop");
            }

            loopConverged = true;
        }
        if (loopConverged) {
            logOuterLoopConverged(ctx.logs, logEnabled, iteration + 1);
            break;
        }
    }

    // 濃度（c）更新：エアコン制御完了後（エアコン制御には影響しない想定）
    const auto sharedNodeState = makeSharedNodeStateArgs(ctx.thermal);
    transport::updateConcentrationIfEnabled(ctx.constants,
                                            ctx.ventilation,
                                            sharedNodeState.nodeGraph,
                                            sharedNodeState.nodeState,
                                            ctx.contaminant,
                                            ctx.logs,
                                            ctx.timings,
                                            std::string(ctx.meta));

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
