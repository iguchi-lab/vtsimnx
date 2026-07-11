#pragma once

#include "simulation_context.h"
#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <ostream>
#include <string>

namespace simulation {

enum class AirconIterationAction {
    Accept,
    RecomputeForFlow,
    RecomputeForControl,
    RecomputeForCapacity,
};

struct AirconIterationResult {
    AirconIterationAction action = AirconIterationAction::Accept;
};

// 空調反復の分岐判定（ユニットテスト用に純粋関数として公開）
inline AirconIterationAction decideAirconIterationAction(bool ductFlowAdjusted,
                                                         bool allAirconControlled,
                                                         bool capacityAdjusted) {
    if (ductFlowAdjusted) {
        return AirconIterationAction::RecomputeForFlow;
    }
    if (!allAirconControlled) {
        return AirconIterationAction::RecomputeForControl;
    }
    if (capacityAdjusted) {
        return AirconIterationAction::RecomputeForCapacity;
    }
    return AirconIterationAction::Accept;
}

AirconIterationResult runAirconIteration(AirconIterationContext& ctx,
                                         const FlowRateMap& flowRates,
                                         int& totalIterations);

} // namespace simulation

// 移行期の別名
using AirconIterationAction = simulation::AirconIterationAction;
using AirconIterationResult = simulation::AirconIterationResult;
using simulation::decideAirconIterationAction;
using simulation::runAirconIteration;
