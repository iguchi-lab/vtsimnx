#pragma once

#include "vtsim_solver.h"

#include <ostream>

class VentilationNetwork;
class ThermalNetwork;
class HumidityNetwork;
class ContaminantNetwork;
class AirconController;

void buildTimestepResult(const SimulationConstants& constants,
                         VentilationNetwork& ventNetwork,
                         ThermalNetwork& thermalNetwork,
                         HumidityNetwork& humidityNetwork,
                         ContaminantNetwork& contaminantNetwork,
                         AirconController& airconController,
                         const FlowRateMap& flowRates,
                         std::ostream& logs,
                         TimestepResult& timestepResultOut);
