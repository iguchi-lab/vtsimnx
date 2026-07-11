#pragma once

#include "simulation_coupled_step.h"
#include "simulation_timestep_state.h"
#include "vtsimnx_solver_timing.h"

#include <ostream>
#include <string>

class VentilationNetwork;
class ThermalNetwork;
class HumidityNetwork;

// 方針B: 除湿潜熱は熱ネットワークへフィードバックしない。
enum class LatentCouplingMode {
    Disabled,
    FeedbackToThermal,
};

void runInnerCoupling(VentilationNetwork& ventNetwork,
                      ThermalNetwork& thermalNetwork,
                      HumidityNetwork& humidityNetwork,
                      const SimulationConstants& constants,
                      std::ostream& logs,
                      TimingList& timings,
                      const std::string& meta,
                      bool logEnabled,
                      int outerIteration,
                      const simulation::detail::TimestepInitialState& initial,
                      CoupledStepData& step,
                      int& totalIterations);
