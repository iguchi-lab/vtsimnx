#include "simulation_aircon_iteration.h"

#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_runner_helpers.h"
#include "utils/utils.h"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

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
    bool supplyHumidityChanged = false;
    const double humidityAbsTol = detail::couplingHumidityTol(ctx.constants);
    std::vector<AirconStateProposal> proposals;

    bool ductFlowAdjusted = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_duct_flow_adjust", meta);
        const auto t0 = std::chrono::steady_clock::now();
        ductFlowAdjusted = ctx.aircon.checkAndAdjustDuctCentralAirflow(
            ctx.thermal, ctx.ventilation, flowRates, ctx.logs, &supplyHumidityChanged,
            humidityAbsTol, &proposals);
        if (metrics) {
            metrics->airconMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    }
    if (ductFlowAdjusted) {
        const auto reasons = aggregateProposalReasons(proposals) |
                             reasonsFromAirconFlags(true, true, false, supplyHumidityChanged);
        recordAirconRecomputeMetrics(metrics, reasons);
        return decideAirconIterationAction(reasons);
    }

    // 制御評価（heat_source のゼロ化は行わない。熱源の正本は SeparatedHeatSources）
    bool allAirconControlled = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_control", meta);
        const auto t0 = std::chrono::steady_clock::now();
        allAirconControlled =
            ctx.aircon.controlAllAircons(
                ctx.thermal, effectiveAirconTemperatureToleranceK(ctx.constants), ctx.logs,
                &supplyHumidityChanged, humidityAbsTol, &proposals);
        if (metrics) {
            metrics->airconMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    }

    if (!allAirconControlled) {
        const auto reasons = aggregateProposalReasons(proposals) |
                             reasonsFromAirconFlags(false, false, false, supplyHumidityChanged);
        recordAirconRecomputeMetrics(metrics, reasons);
        return decideAirconIterationAction(reasons);
    }

    bool adjustmentMade = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_capacity_adjust", meta);
        const auto t0 = std::chrono::steady_clock::now();
        adjustmentMade = ctx.aircon.checkAndAdjustCapacity(
            ctx.thermal, ctx.ventilation, ctx.constants, flowRates, ctx.logs, totalIterations,
            &supplyHumidityChanged, humidityAbsTol, &proposals);
        if (metrics) {
            metrics->airconMs +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    }

    const auto reasons = aggregateProposalReasons(proposals) |
                         reasonsFromAirconFlags(false, true, adjustmentMade, supplyHumidityChanged);
    recordAirconRecomputeMetrics(metrics, reasons);
    return decideAirconIterationAction(reasons);
}

} // namespace simulation
