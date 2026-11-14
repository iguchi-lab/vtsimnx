#pragma once

#include "types/graph_types.h"
#include "aircon/aircon_saturation.h"
#include <cmath>

namespace aircon::humidity_control {

inline bool enabled(const VertexProperties& n) {
    return n.control_mode != "temperature_priority" &&
           n.current_mode == "COOLING" && std::isfinite(n.current_pre_rh);
}

inline bool simultaneous(const VertexProperties& n) {
    return n.control_mode == "temperature_humidity";
}

// Invert the SAME saturation bound used for serialized outlet acceptance.
inline double dewPoint(double x) {
    double lo = -80.0, hi = 80.0;
    for (int i = 0; i < 60; ++i) {
        const double mid = (lo + hi) / 2.0;
        if (latent::saturationHumidityUpperBound(mid) < x) lo = mid;
        else hi = mid;
    }
    return hi;
}

} // namespace aircon::humidity_control
