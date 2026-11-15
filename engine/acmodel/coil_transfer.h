#pragma once

#include <algorithm>
#include <cmath>

namespace acmodel {
// Hosoi (2018), Eq. 24. Retain the DUCT_CENTRAL correlation's lower
// evaluation velocity (400 m3/h at Af=0.133 m2). This affects only the
// empirical coefficient, NEVER the actual air/water/energy mass flow.
inline double coolingCoilMassTransferCoefficient(double actualFlowM3s, double faceAreaM2) {
    if (!(actualFlowM3s > 0.0) || !(faceAreaM2 > 0.0)) return 0.0;
    constexpr double minimumCorrelationVelocity = (400.0 / 3600.0) / 0.133;
    const double velocity = std::max(actualFlowM3s / faceAreaM2, minimumCorrelationVelocity);
    return 0.050 * std::log(velocity) + 0.073; // kg/(m2 s)
}
} // namespace acmodel
