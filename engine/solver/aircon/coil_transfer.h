#pragma once
#include <algorithm>
#include <cmath>
namespace aircon::latent {
// Committee report (2021), 4.2.1 step 6. Retain the original natural-log
// convention (the report does not specify its base). Use actual face velocity.
inline double coolingCoilMassTransferCoefficient(double flow, double faceArea) {
    if (!(flow > 0.0) || !(faceArea > 0.0)) return 0.0;
    return std::max(0.0, 0.037 * std::log(flow / faceArea) + 0.0637);
}
} // namespace aircon::latent
