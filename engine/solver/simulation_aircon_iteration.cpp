#include "simulation_aircon_iteration.h"

#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_runner_helpers.h"
#include "utils/utils.h"

#include <string>

namespace simulation {

AirconIterationResult runAirconIteration(AirconIterationContext& ctx,
                                         const FlowRateMap& flowRates,
                                         int& totalIterations) {
    AirconIterationResult r;
    const std::string meta(ctx.meta);

    // 0. DUCT_CENTRAL の処理熱量連動風量を補正（変更が入ったら外側ループをやり直し）
    bool ductFlowAdjusted = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_duct_flow_adjust", meta);
        ductFlowAdjusted = ctx.aircon.checkAndAdjustDuctCentralAirflow(
            ctx.thermal, ctx.ventilation, flowRates, ctx.logs);
    }
    if (ductFlowAdjusted) {
        r.action = decideAirconIterationAction(true, false, false);
        return r;
    }

    // 1. 現在の温度でエアコン出力を決定する。
    //    制御前に heat_source をゼロ化してから再設定する（外側ループ開始時の初期化とは別意図）。
    bool allAirconControlled = false;
    {
        detail::resetNodeHeatSources(ctx.thermal.getGraph());

        ScopedTimer timer(ctx.timings, "aircon_control", meta);
        allAirconControlled =
            ctx.aircon.controlAllAircons(ctx.thermal, ctx.constants.thermalTolerance, ctx.logs);
    }

    if (!allAirconControlled) {
        r.action = decideAirconIterationAction(false, false, false);
        return r;
    }

    bool adjustmentMade = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_capacity_adjust", meta);
        adjustmentMade = ctx.aircon.checkAndAdjustCapacity(
            ctx.thermal, ctx.ventilation, ctx.constants, flowRates, ctx.logs, totalIterations);
    }

    r.action = decideAirconIterationAction(false, true, adjustmentMade);
    return r;
}

} // namespace simulation
