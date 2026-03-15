#include "aircon/aircon_latent.h"

#include "aircon/aircon_operation_mode.h"
#include "archenv/include/archenv.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {
constexpr double kAirDensity = archenv::DENSITY_DRY_AIR;         // [kg/m^3]
constexpr double kAirSpecificHeat = archenv::SPECIFIC_HEAT_AIR;   // [J/(kg·K)]
constexpr double kDefaultOuterFlowRate = 25.5 / 60.0;             // m^3/s
constexpr double kDefaultBypassFactor = 0.2;                      // [-]
constexpr double kDefaultSupplyRhPercent = 95.0;                  // [%]
constexpr double kDefaultCoilFaceArea = 0.133;                    // [m^2]
constexpr double kDefaultCoilSurfaceArea = 4.84;                  // [m^2]

inline std::optional<double> readFinitePositiveSpecNumber(const nlohmann::json& spec, const char* key) {
    if (!spec.is_object() || !spec.contains(key) || !spec[key].is_number()) return std::nullopt;
    const double v = spec[key].get<double>();
    if (!std::isfinite(v) || !(v > 0.0)) return std::nullopt;
    return v;
}

inline double readBypassFactor(const VertexProperties& nodeProps) {
    const auto& spec = nodeProps.ac_spec;
    auto read = [&](const char* key, double& out) -> bool {
        if (!spec.is_object() || !spec.contains(key) || !spec[key].is_number()) return false;
        out = spec[key].get<double>();
        return std::isfinite(out);
    };
    double bf = kDefaultBypassFactor;
    if (!read("bf", bf) && !read("BF", bf) && !read("bypass_factor", bf)) {
        return kDefaultBypassFactor;
    }
    return std::clamp(bf, 0.0, 0.99);
}

inline std::string readLatentMethod(const VertexProperties& nodeProps) {
    const auto& spec = nodeProps.ac_spec;
    if (!spec.is_object() || !spec.contains("latent_method") || !spec["latent_method"].is_string()) {
        return "rh95";
    }
    return toLowerCopy(spec["latent_method"].get<std::string>());
}

inline double readCoilFaceArea(const VertexProperties& nodeProps) {
    const auto& spec = nodeProps.ac_spec;
    if (auto v = readFinitePositiveSpecNumber(spec, "Af")) return *v;
    if (auto v = readFinitePositiveSpecNumber(spec, "coil_face_area")) return *v;
    return kDefaultCoilFaceArea;
}

inline double readCoilSurfaceArea(const VertexProperties& nodeProps) {
    const auto& spec = nodeProps.ac_spec;
    if (auto v = readFinitePositiveSpecNumber(spec, "Ao")) return *v;
    if (auto v = readFinitePositiveSpecNumber(spec, "coil_surface_area")) return *v;
    return kDefaultCoilSurfaceArea;
}

inline double dewPointFromAbsoluteHumidity(double x) {
    const double xx = std::max(0.0, x);
    double lo = -40.0;
    double hi = 80.0;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double xSat = std::max(0.0, archenv::absolute_humidity_from_vapor_pressure(
                                              archenv::saturation_vapor_pressure(mid)));
        if (xSat < xx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

} // namespace

namespace aircon::latent {

double totalHeatCapacity(const LatentProcessResult& loads) {
    return std::max(0.0, loads.sensibleHeatCapacity) + std::max(0.0, loads.latentHeatCapacity);
}

acmodel::InputData buildAcmodelInput(const AirconValidationData& validData,
                                     double sensibleHeatCapacity,
                                     double latentHeatCapacity,
                                     double airFlowRate) {
    acmodel::InputData input;
    input.T_ex = validData.outdoorTemp;
    input.T_in = validData.indoorTemp;
    input.X_ex = validData.outdoorX;
    input.X_in = validData.indoorX;
    input.Q_S = sensibleHeatCapacity;
    input.Q_L = latentHeatCapacity;
    input.Q = sensibleHeatCapacity + latentHeatCapacity;
    input.V_inner = airFlowRate;
    input.V_outer = kDefaultOuterFlowRate;
    input.V_vent = 0.0; // 未入力時は各モデル側の既定値を使用
    return input;
}

LatentProcessResult estimateLatentProcess(const AirconValidationData& validData,
                                          OperationMode operationMode,
                                          double sensibleHeatCapacity,
                                          double airFlowRate,
                                          const VertexProperties& nodeProps) {
    LatentProcessResult result;
    result.sensibleHeatCapacity = std::max(0.0, sensibleHeatCapacity);
    result.supplyX = std::max(0.0, validData.indoorX);

    if (operationMode != OperationMode::Cooling) return result;
    if (!(airFlowRate > std::numeric_limits<double>::epsilon())) return result;

    const double tIn = validData.indoorTemp;
    const double tOut = validData.airconTemp;
    const double xIn = std::max(0.0, validData.indoorX);
    if (!(tIn > tOut)) {
        result.supplyX = xIn;
        return result;
    }

    const std::string latentMethod = readLatentMethod(nodeProps);
    if (latentMethod == "none") {
        result.supplyX = xIn;
        return result;
    }

    auto applyRh95 = [&]() {
        const double x95 = std::max(0.0, archenv::absolute_humidity(tOut, kDefaultSupplyRhPercent));
        result.supplyX = std::min(xIn, x95);
        const double xSatOut = std::max(0.0, archenv::absolute_humidity(tOut, 100.0));
        if (xSatOut > std::numeric_limits<double>::epsilon()) {
            result.supplyRhPercent = 100.0 * result.supplyX / xSatOut;
        }
    };

    if (latentMethod == "bf") {
        const double bf = readBypassFactor(nodeProps);
        if (!(bf < 1.0)) {
            result.supplyX = xIn;
        } else {
            const double tCoil = tIn - (tIn - tOut) / std::max(1e-9, (1.0 - bf));
            const double xCoil = std::max(0.0, archenv::absolute_humidity(tCoil, 100.0));
            result.coilTemp = tCoil;
            result.coilX = xCoil;

            const double denom = (tIn - tCoil);
            double xOut = xIn;
            if (std::abs(denom) > 1e-9) {
                const double ratio = (tIn - tOut) / denom;
                xOut = xIn + (xCoil - xIn) * ratio;
            }
            if (!std::isfinite(xOut)) xOut = xIn;
            xOut = std::max(0.0, xOut);
            result.supplyX = std::min(xOut, xIn);
        }

        const double xSatOut = std::max(0.0, archenv::absolute_humidity(tOut, 100.0));
        if (xSatOut > std::numeric_limits<double>::epsilon()) {
            result.supplyRhPercent = 100.0 * result.supplyX / xSatOut;
            result.bfRhPercentBeforeFallback = result.supplyRhPercent;
            result.rhExceeded = (result.supplyRhPercent > 100.0 + 1e-6);
        }
        if (result.rhExceeded) {
            applyRh95();
            result.usedRh95Fallback = true;
            result.rhExceeded = false;
        }
    } else {
        if (latentMethod == "coil_aoaf" || latentMethod == "aoaf" || latentMethod == "literature") {
            const double V = std::abs(airFlowRate);
            const double Af = readCoilFaceArea(nodeProps);
            const double Ao = readCoilSurfaceArea(nodeProps);
            const double HsW = std::max(0.0, result.sensibleHeatCapacity);
            if (V <= std::numeric_limits<double>::epsilon() ||
                Af <= std::numeric_limits<double>::epsilon() ||
                Ao <= std::numeric_limits<double>::epsilon() ||
                HsW <= std::numeric_limits<double>::epsilon()) {
                result.supplyX = xIn;
            } else {
                const double tr = tIn;
                const double xr = xIn;
                const double trDp = dewPointFromAbsoluteHumidity(xr);

                const double te = tr - HsW / (kAirSpecificHeat * kAirDensity * V);
                const double xe = std::max(0.0, archenv::absolute_humidity(te, 100.0));

                const double tStar = 0.5 * (tr + te);
                const double xStar = 0.5 * (xr + xe);

                const double vx = V / Af;
                if (vx > 0.0) {
                    const double kx = std::max(0.0, 0.037 * std::log(vx) + 0.0637);
                    const double alphaC = kx * (archenv::SPECIFIC_HEAT_AIR +
                                                archenv::SPECIFIC_HEAT_WATER_VAPOR * xStar);
                    if (alphaC > std::numeric_limits<double>::epsilon()) {
                        const double td = tStar - HsW / (alphaC * Ao);
                        const double xd = std::max(0.0, archenv::absolute_humidity(td, 100.0));

                        double hrW = 0.0;
                        if (trDp > td) {
                            const double dx = std::max(0.0, xStar - xd);
                            hrW = std::max(0.0, (archenv::LATENT_HEAT_VAPORIZATION +
                                                 archenv::SPECIFIC_HEAT_WATER_VAPOR * td) *
                                                kx * dx * Ao);
                        }
                        result.latentHeatCapacity = hrW;

                        const double denom = (archenv::LATENT_HEAT_VAPORIZATION +
                                              archenv::SPECIFIC_HEAT_WATER_VAPOR * td);
                        if (denom > std::numeric_limits<double>::epsilon()) {
                            const double mWater = hrW / denom;
                            const double deltaX = mWater / (kAirDensity * V);
                            result.supplyX = std::clamp(xr - std::max(0.0, deltaX), 0.0, xr);
                        } else {
                            result.supplyX = xr;
                        }
                    } else {
                        result.supplyX = xr;
                    }
                } else {
                    result.supplyX = xr;
                }
            }
        } else {
            applyRh95();
        }
    }

    if (!(result.latentHeatCapacity > 0.0)) {
        const double deltaX = std::max(0.0, xIn - result.supplyX);
        result.latentHeatCapacity =
            std::max(0.0,
                     kAirDensity * std::abs(airFlowRate) *
                         archenv::vapor_latent_heat(tOut) * deltaX);
    }
    return result;
}

} // namespace aircon::latent
