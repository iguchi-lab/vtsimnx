#pragma once

#include "archenv/include/archenv.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace aircon::latent {

// Conservative boundary target, including float32 artifact rounding protection.
inline double saturationHumidityUpperBound(double temperatureC) {
    return std::max(0.0, archenv::absolute_humidity(temperatureC, 100.0)) *
        (1.0 - 8.0 * std::numeric_limits<float>::epsilon());
}

// A solved passive outlet can exceed that conservative target by roundoff
// while still being unsaturated. Check the physical limit in both the solver
// state and the actual f32 artifact representation, without an RH tolerance.
inline bool unsaturatedBeforeAndAfterSerialization(double temperatureC, double x) {
    const double savedT = static_cast<float>(temperatureC);
    const double savedX = static_cast<float>(x);
    if (!std::isfinite(temperatureC) || !std::isfinite(x) || x < 0.0 ||
        !std::isfinite(savedT) || !std::isfinite(savedX)) return false;
    const double limit = archenv::absolute_humidity(temperatureC, 100.0);
    const double savedLimit = archenv::absolute_humidity(savedT, 100.0);
    return std::isfinite(limit) && std::isfinite(savedLimit) &&
        x <= limit && savedX <= savedLimit;
}

} // namespace aircon::latent
