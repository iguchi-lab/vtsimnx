#pragma once

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <limits>
#include <ostream>
#include <string_view>

class VentilationNetwork;
class ThermalNetwork;
class HumidityNetwork;
class ContaminantNetwork;
class AirconController;

namespace simulation {

// 非所有参照の束ね。公開 runSimulation シグネチャは維持し、内部で組み立てる。
struct Context {
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

struct InnerCouplingContext {
    VentilationNetwork& ventilation;
    ThermalNetwork& thermal;
    HumidityNetwork& humidity;
    const SimulationConstants& constants;
    std::ostream& logs;
    TimingList& timings;
    std::string_view meta;
};

struct AirconIterationContext {
    AirconController& aircon;
    ThermalNetwork& thermal;
    VentilationNetwork& ventilation;
    const SimulationConstants& constants;
    std::ostream& logs;
    TimingList& timings;
    std::string_view meta;
};

inline int toLogIndex1Based(std::size_t zeroBased) noexcept {
    if (zeroBased >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(zeroBased + 1);
}

inline InnerCouplingContext makeInnerCouplingContext(Context& ctx) {
    return InnerCouplingContext{
        ctx.ventilation,
        ctx.thermal,
        ctx.humidity,
        ctx.constants,
        ctx.logs,
        ctx.timings,
        ctx.meta,
    };
}

inline AirconIterationContext makeAirconIterationContext(Context& ctx, std::string_view meta) {
    return AirconIterationContext{
        ctx.aircon,
        ctx.thermal,
        ctx.ventilation,
        ctx.constants,
        ctx.logs,
        ctx.timings,
        meta,
    };
}

} // namespace simulation
