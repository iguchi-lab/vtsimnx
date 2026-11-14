#pragma once

#include "../../vtsim_solver.h"

namespace FlowCalculation {

// Effective flow law for one pressure problem. Classify after timestep updates
// and fallback mutations; never cache this on the mutable EdgeProperties.
enum class FlowLaw {
    Disabled,
    FixedVolume,
    SimpleOpening,
    Gap,
    Fan,
    PressureLoss,
    Unknown,
};

inline bool isFixedVolumeFlowEdge(const EdgeProperties& edge) {
    return edge.type == "fixed_flow" || edge.has_prescribed_vol || !edge.vol.empty();
}

inline FlowLaw resolveFlowLaw(const EdgeProperties& edge) {
    if (!edge.current_enabled) return FlowLaw::Disabled;
    if (isFixedVolumeFlowEdge(edge)) return FlowLaw::FixedVolume;
    if (edge.type == "simple_opening") return FlowLaw::SimpleOpening;
    if (edge.type == "gap") return FlowLaw::Gap;
    if (edge.type == "fan") return FlowLaw::Fan;
    if (edge.type == "pressure_loss") return FlowLaw::PressureLoss;
    return FlowLaw::Unknown;
}

} // namespace FlowCalculation
