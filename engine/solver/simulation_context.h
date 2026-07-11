#pragma once

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <ostream>
#include <string>
#include <string_view>

class VentilationNetwork;
class ThermalNetwork;
class HumidityNetwork;
class ContaminantNetwork;
class AirconController;

// 非所有参照の束ね。公開 runSimulation シグネチャは維持し、内部で組み立てる。
struct SimulationContext {
    VentilationNetwork& ventilation;
    ThermalNetwork& thermal;
    HumidityNetwork& humidity;
    ContaminantNetwork& contaminant;
    AirconController& aircon;
    const SimulationConstants& constants;
    std::ostream& logs;
    TimingList& timings;
    std::string_view meta;
};
