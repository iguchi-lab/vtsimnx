#pragma once

#include "simulation_context.h"
#include "simulation_coupled_step.h"
#include "simulation_timestep_state.h"
#include "vtsimnx_solver_timing.h"

#include <cstddef>
#include <stdexcept>

namespace simulation {

// 0=Disabled, 1=FromHumidityChange（ノード絶対湿度変化→同ノード潜熱）
enum class LatentCouplingMode {
    Disabled = 0,
    FromHumidityChange = 1,
};

inline LatentCouplingMode latentCouplingModeFromConstants(const SimulationConstants& c) {
    return (c.latentCouplingMode == static_cast<int>(LatentCouplingMode::FromHumidityChange))
               ? LatentCouplingMode::FromHumidityChange
               : LatentCouplingMode::Disabled;
}

inline bool latentCouplingActive(const SimulationConstants& c) {
    return c.humidityCalc && c.temperatureCalc && c.moistureCouplingEnabled &&
           latentCouplingModeFromConstants(c) == LatentCouplingMode::FromHumidityChange;
}

// Disabled は 0。FromHumidityChange は適用量の集計用。
inline double resolveLatentAppliedThisIter(LatentCouplingMode mode,
                                          double appliedW = 0.0) {
    switch (mode) {
    case LatentCouplingMode::Disabled:
        return 0.0;
    case LatentCouplingMode::FromHumidityChange:
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
                              std::size_t outerIteration,
                              detail::SeparatedHeatSources* heatSources = nullptr);

} // namespace simulation
