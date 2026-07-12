#pragma once

#include "simulation_context.h"
#include "simulation_coupled_step.h"
#include "simulation_timestep_state.h"
#include "vtsimnx_solver_timing.h"

#include <cstddef>
#include <stdexcept>

namespace simulation {

// 方針B既定: Disabled。SimulationConstants::latentCouplingMode と対応。
enum class LatentCouplingMode {
    Disabled = 0,
    FeedbackToThermal = 1,
};

inline LatentCouplingMode latentCouplingModeFromConstants(const SimulationConstants& c) {
    return (c.latentCouplingMode == static_cast<int>(LatentCouplingMode::FeedbackToThermal))
               ? LatentCouplingMode::FeedbackToThermal
               : LatentCouplingMode::Disabled;
}

// Disabled は 0。FeedbackToThermal は適用量の集計用（実装は apply 側）。
inline double resolveLatentAppliedThisIter(LatentCouplingMode mode,
                                          double appliedW = 0.0) {
    switch (mode) {
    case LatentCouplingMode::Disabled:
        return 0.0;
    case LatentCouplingMode::FeedbackToThermal:
        return appliedW;
    }
    throw std::logic_error("unknown LatentCouplingMode");
}

void runInnerCoupling(InnerCouplingContext& ctx,
                      bool logEnabled,
                      std::size_t outerIteration,
                      const detail::TimestepInitialState& initial,
                      CoupledStepData& step,
                      int& totalIterations,
                      detail::SeparatedHeatSources& heatSources,
                      bool forceMinTwoCouplingIters);

// moistureCouplingEnabled=false 時: 外側ループごとに1回だけ湿気を更新する。
void runDecoupledHumidityStep(InnerCouplingContext& ctx,
                              const detail::TimestepInitialState& initial,
                              CoupledStepData& step,
                              std::size_t outerIteration);

} // namespace simulation
