#include "simulation_aircon_iteration.h"

#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_runner_helpers.h"
#include "utils/utils.h"

#include <chrono>
#include <string>

namespace simulation {

AirconIterationAction runAirconIteration(AirconIterationContext& ctx,
                                         const FlowRateMap& flowRates,
                                         int& totalIterations) {
    if (auto* overrideFn = test_hooks::airconIterationOverride()) {
        if (auto forced = overrideFn()) {
            return *forced;
        }
    }

    const std::string meta(ctx.meta);
    auto* metrics = ctx.metrics;

    bool ductFlowAdjusted = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_duct_flow_adjust", meta);
        const auto t0 = std::chrono::steady_clock::now();
        ductFlowAdjusted = ctx.aircon.checkAndAdjustDuctCentralAirflow(
            ctx.thermal, ctx.ventilation, flowRates, ctx.logs);
        if (metrics) {
            metrics->airconMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    }
    if (ductFlowAdjusted) {
        if (metrics) ++metrics->airconFlowAdjustRecalc;
        return decideAirconIterationAction(true, false, false);
    }

    // 制御前に heat_source をゼロ化してから再設定する（外側ループ開始時の初期化とは別意図）。
    bool allAirconControlled = false;
    {
        detail::resetNodeHeatSources(ctx.thermal.getGraph());

        ScopedTimer timer(ctx.timings, "aircon_control", meta);
        const auto t0 = std::chrono::steady_clock::now();
        allAirconControlled =
            ctx.aircon.controlAllAircons(
                ctx.thermal, effectiveAirconTemperatureToleranceK(ctx.constants), ctx.logs);
        if (metrics) {
            metrics->airconMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    }

    if (!allAirconControlled) {
        if (metrics) ++metrics->airconOnOffRecalc;
        return decideAirconIterationAction(false, false, false);
    }

    bool adjustmentMade = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_capacity_adjust", meta);
        const auto t0 = std::chrono::steady_clock::now();
        adjustmentMade = ctx.aircon.checkAndAdjustCapacity(
            ctx.thermal, ctx.ventilation, ctx.constants, flowRates, ctx.logs, totalIterations);
        if (metrics) {
            metrics->airconMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    }

    if (adjustmentMade && metrics) ++metrics->airconCapacityRecalc;
    return decideAirconIterationAction(false, true, adjustmentMade);
}

} // namespace simulation
