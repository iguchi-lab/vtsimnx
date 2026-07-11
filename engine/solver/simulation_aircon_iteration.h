#pragma once

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <ostream>
#include <string>

class AirconController;
class ThermalNetwork;
class VentilationNetwork;

enum class AirconIterationAction {
    Accept,
    RecomputeForFlow,
    RecomputeForControl,
    RecomputeForCapacity,
};

struct AirconIterationResult {
    AirconIterationAction action = AirconIterationAction::Accept;
};

AirconIterationResult runAirconIteration(AirconController& airconController,
                                         ThermalNetwork& thermalNetwork,
                                         VentilationNetwork& ventNetwork,
                                         const SimulationConstants& constants,
                                         const FlowRateMap& flowRates,
                                         std::ostream& logs,
                                         int& totalIterations,
                                         TimingList& timings,
                                         const std::string& meta);
