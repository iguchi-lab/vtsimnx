#pragma once

#include "simulation_context.h"
#include "simulation_metrics.h"
#include "types/aircon_control_state.h"
#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace simulation {

enum class AirconIterationAction {
    Accept,
    RecomputeForFlow,
    RecomputeForControl,
    RecomputeForCapacity,
    RecomputeForSupplyHumidity,
};

struct AirconIterationResult {
    AirconIterationAction action = AirconIterationAction::Accept;
    AirconRecomputeReason reasons = AirconRecomputeReason::None;
    std::vector<AirconStateProposal> proposals;
};

/** ビットフラグから再計算アクションを決める（優先順位: ON/OFF > Capacity > Flow > Humidity）。 */
inline AirconIterationAction decideAirconIterationAction(AirconRecomputeReason reasons) {
    if (hasReason(reasons, AirconRecomputeReason::OnOffChanged)) {
        return AirconIterationAction::RecomputeForControl;
    }
    if (hasReason(reasons, AirconRecomputeReason::CapacitySetpointChanged)) {
        return AirconIterationAction::RecomputeForCapacity;
    }
    if (hasReason(reasons, AirconRecomputeReason::AirflowChanged)) {
        return AirconIterationAction::RecomputeForFlow;
    }
    if (hasReason(reasons, AirconRecomputeReason::SupplyHumidityChanged)) {
        return AirconIterationAction::RecomputeForSupplyHumidity;
    }
    return AirconIterationAction::Accept;
}

inline AirconRecomputeReason reasonsFromAirconFlags(bool ductFlowAdjusted,
                                                    bool allAirconControlled,
                                                    bool capacityAdjusted,
                                                    bool supplyHumidityChanged = false) {
    AirconRecomputeReason reasons = AirconRecomputeReason::None;
    if (ductFlowAdjusted) {
        reasons |= AirconRecomputeReason::AirflowChanged;
    }
    if (!allAirconControlled) {
        reasons |= AirconRecomputeReason::OnOffChanged;
    }
    if (capacityAdjusted) {
        reasons |= AirconRecomputeReason::CapacitySetpointChanged;
    }
    if (supplyHumidityChanged) {
        reasons |= AirconRecomputeReason::SupplyHumidityChanged;
    }
    return reasons;
}

// 空調反復の分岐判定（互換: 旧 bool API。内部はビットフラグへ変換）
inline AirconIterationAction decideAirconIterationAction(bool ductFlowAdjusted,
                                                         bool allAirconControlled,
                                                         bool capacityAdjusted,
                                                         bool supplyHumidityChanged = false) {
    return decideAirconIterationAction(reasonsFromAirconFlags(
        ductFlowAdjusted, allAirconControlled, capacityAdjusted, supplyHumidityChanged));
}

/** 外側1回分の再計算理由をメトリクスへ反映（主因のみカウント、マスクは OR）。 */
inline void recordAirconRecomputeMetrics(TimestepSolveMetrics* metrics,
                                         AirconRecomputeReason reasons) {
    if (!metrics || reasons == AirconRecomputeReason::None) {
        return;
    }
    metrics->airconRecomputeReasonsMask |= static_cast<std::uint32_t>(reasons);
    if (hasReason(reasons, AirconRecomputeReason::OnOffChanged)) {
        ++metrics->airconOnOffRecalc;
    } else if (hasReason(reasons, AirconRecomputeReason::CapacitySetpointChanged)) {
        ++metrics->airconCapacityRecalc;
    } else if (hasReason(reasons, AirconRecomputeReason::AirflowChanged)) {
        ++metrics->airconFlowAdjustRecalc;
    } else if (hasReason(reasons, AirconRecomputeReason::SupplyHumidityChanged)) {
        ++metrics->airconSupplyHumidityRecalc;
    }
}

namespace test_hooks {

// 統合テスト用。設定時は本番の空調分岐をバイパスする。
using AirconIterationOverride = std::optional<AirconIterationAction> (*)();
inline AirconIterationOverride& airconIterationOverride() {
    static AirconIterationOverride fn = nullptr;
    return fn;
}

struct ScopedAirconIterationOverride {
    AirconIterationOverride previous;
    explicit ScopedAirconIterationOverride(AirconIterationOverride fn)
        : previous(airconIterationOverride()) {
        airconIterationOverride() = fn;
    }
    ~ScopedAirconIterationOverride() { airconIterationOverride() = previous; }
    ScopedAirconIterationOverride(const ScopedAirconIterationOverride&) = delete;
    ScopedAirconIterationOverride& operator=(const ScopedAirconIterationOverride&) = delete;
};

} // namespace test_hooks

// totalIterations: 内側連成の累積回数（空調容量調整 API 互換のため参照渡し）。
AirconIterationAction runAirconIteration(AirconIterationContext& ctx,
                                         const FlowRateMap& flowRates,
                                         int& totalIterations);

} // namespace simulation
