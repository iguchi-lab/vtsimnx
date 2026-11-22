#pragma once

#include "types/graph_types.h"
#include "aircon/aircon_saturation.h"
#include <cmath>

namespace aircon::humidity_control {

inline bool humidityControlMode(const VertexProperties& n) {
    return n.control_mode == "humidity_priority" || n.control_mode == "temperature_humidity";
}

inline bool idealRoomLoad(const VertexProperties& n) {
    return n.control_mode == "ideal_room_load";
}

inline bool enabled(const VertexProperties& n) {
    return humidityControlMode(n) &&
           n.current_mode == "COOLING" && std::isfinite(n.current_pre_rh);
}

inline bool simultaneous(const VertexProperties& n) {
    return n.control_mode == "temperature_humidity";
}

// ON cooling with a finite RH setpoint: hold set-node x at the ideal target.
inline bool idealRoomLoadHumidityActive(const VertexProperties& n) {
    return idealRoomLoad(n) && n.on && n.current_mode == "COOLING" &&
           std::isfinite(n.current_pre_rh);
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
