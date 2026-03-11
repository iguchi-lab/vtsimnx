#pragma once

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <ostream>

class VentilationNetwork;
class ThermalNetwork;
class AirconController;

// シミュレーションのタイムステップループを実行
void runSimulation(VentilationNetwork& ventNetwork,
                   ThermalNetwork& thermalNetwork,
                   AirconController& airconController,
                   const SimulationConstants& constants,
                   TimestepResult& timestepResultOut,
                   std::ostream& logs,
                   TimingList& timings,
                   const std::string& meta);


