#pragma once

#include "simulation_context.h"
#include "simulation_coupled_step.h"
#include "simulation_timestep_state.h"
#include "vtsimnx_solver_timing.h"

#include <cstddef>
#include <stdexcept>

namespace simulation {

// 0=Disabled, 1=FromHumidityChange（実験・非推奨）, 2=FromPhaseChange（材料相変化）
enum class LatentCouplingMode {
    Disabled = 0,
    FromHumidityChange = 1,
    FromPhaseChange = 2,
};

inline LatentCouplingMode latentCouplingModeFromConstants(const SimulationConstants& c) {
    switch (c.latentCouplingMode) {
    case static_cast<int>(LatentCouplingMode::FromHumidityChange):
        return LatentCouplingMode::FromHumidityChange;
    case static_cast<int>(LatentCouplingMode::FromPhaseChange):
        return LatentCouplingMode::FromPhaseChange;
    default:
        return LatentCouplingMode::Disabled;
    }
}

inline bool latentCouplingActive(const SimulationConstants& c) {
    if (!(c.humidityCalc && c.temperatureCalc && c.moistureCouplingEnabled)) {
        return false;
    }
    const auto mode = latentCouplingModeFromConstants(c);
    return mode == LatentCouplingMode::FromHumidityChange ||
           mode == LatentCouplingMode::FromPhaseChange;
}

// Disabled は 0。有効モードは適用量の集計用。
inline double resolveLatentAppliedThisIter(LatentCouplingMode mode,
                                          double appliedW = 0.0) {
    switch (mode) {
    case LatentCouplingMode::Disabled:
        return 0.0;
    case LatentCouplingMode::FromHumidityChange:
    case LatentCouplingMode::FromPhaseChange:
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
