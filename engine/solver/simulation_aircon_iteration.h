#pragma once

#include "simulation_context.h"
#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <optional>

namespace simulation {

enum class AirconIterationAction {
    Accept,
    RecomputeForFlow,
    RecomputeForControl,
    RecomputeForCapacity,
    RecomputeForSupplyHumidity,
};

// 空調反復の分岐判定（ユニットテスト用に純粋関数として公開）
inline AirconIterationAction decideAirconIterationAction(bool ductFlowAdjusted,
                                                         bool allAirconControlled,
                                                         bool capacityAdjusted,
                                                         bool supplyHumidityChanged = false) {
    if (ductFlowAdjusted) {
        return AirconIterationAction::RecomputeForFlow;
    }
    if (!allAirconControlled) {
        return AirconIterationAction::RecomputeForControl;
    }
    if (capacityAdjusted) {
        return AirconIterationAction::RecomputeForCapacity;
    }
    if (supplyHumidityChanged) {
        return AirconIterationAction::RecomputeForSupplyHumidity;
    }
    return AirconIterationAction::Accept;
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
