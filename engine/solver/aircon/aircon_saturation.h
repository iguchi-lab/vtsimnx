#pragma once

#include "archenv/include/archenv.h"
#include <algorithm>
#include <limits>

namespace aircon::latent {

// Shared physical limit, including float32 artifact rounding protection.
inline double saturationHumidityUpperBound(double temperatureC) {
    return std::max(0.0, archenv::absolute_humidity(temperatureC, 100.0)) *
        (1.0 - 8.0 * std::numeric_limits<float>::epsilon());
}

} // namespace aircon::latent
