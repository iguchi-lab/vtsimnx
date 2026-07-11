#pragma once

#include "simulation_context.h"
#include "simulation_coupled_step.h"
#include "simulation_timestep_state.h"
#include "vtsimnx_solver_timing.h"

#include <stdexcept>
#include <string>

namespace simulation {

// 方針B: 除湿潜熱は熱ネットワークへフィードバックしない。
enum class LatentCouplingMode {
    Disabled,
    FeedbackToThermal,
};

// Disabled は 0、FeedbackToThermal は未実装のため logic_error。
inline double resolveLatentAppliedThisIter(LatentCouplingMode mode) {
    switch (mode) {
    case LatentCouplingMode::Disabled:
        return 0.0;
    case LatentCouplingMode::FeedbackToThermal:
        throw std::logic_error("latent feedback is not implemented");
    }
    throw std::logic_error("unknown LatentCouplingMode");
}

void runInnerCoupling(InnerCouplingContext& ctx,
                      bool logEnabled,
                      int outerIteration,
                      const detail::TimestepInitialState& initial,
                      CoupledStepData& step,
                      int& totalIterations);

// moistureCouplingEnabled=false 時: 外側ループごとに1回だけ湿気を更新する。
void runDecoupledHumidityStep(InnerCouplingContext& ctx,
                              const detail::TimestepInitialState& initial,
                              CoupledStepData& step,
                              int outerIteration);

} // namespace simulation

using LatentCouplingMode = simulation::LatentCouplingMode;
using simulation::resolveLatentAppliedThisIter;
using simulation::runInnerCoupling;
using simulation::runDecoupledHumidityStep;
