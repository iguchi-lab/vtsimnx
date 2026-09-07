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

    // 1) ON/OFF（符号付き必要負荷）。OFF 変更があれば能力・風量より先に再計算へ戻る。
    bool allAirconControlled = false;
    {
        ScopedTimer timer(ctx.timings, "aircon_control", meta);
        const auto t0 = std::chrono::steady_clock::now();
        allAirconControlled =
            ctx.aircon.controlAllAircons(
                ctx.thermal, effectiveAirconTemperatureToleranceK(ctx.constants), ctx.logs,
                &supplyHumidityChanged, humidityAbsTol, &proposals, &flowRates);
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

    // 2) 処理熱（能力制限）を先に確定する。探索中は風量を動かさない。
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
    if (adjustmentMade) {
        const auto reasons = aggregateProposalReasons(proposals) |
                             reasonsFromAirconFlags(false, true, true, supplyHumidityChanged);
        recordAirconRecomputeMetrics(metrics, reasons);
        return decideAirconIterationAction(reasons);
    }

    // 3) 処理熱が安定したあと、最後に DUCT_CENTRAL 風量を処理熱へ合わせる。
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

    const auto reasons = aggregateProposalReasons(proposals) |
                         reasonsFromAirconFlags(ductFlowAdjusted, true, false, supplyHumidityChanged);
    recordAirconRecomputeMetrics(metrics, reasons);
    return decideAirconIterationAction(reasons);
}

} // namespace simulation
