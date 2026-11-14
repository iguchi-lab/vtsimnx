#include "network/humidity_network.h"
#include "aircon/aircon_latent.h"

#include "aircon/aircon_network_utils.h"
#include "aircon/aircon_humidity_control.h"
#include "aircon/aircon_operation_mode.h"
#include "archenv/include/archenv.h"
#include "core/thermal/thermal_moist_air.h"
#include "network/thermal_network.h"

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
                                          const VertexProperties& nodeProps,
                                          bool moistEnthalpyEnabled) {
    LatentProcessResult result;
    result.sensibleHeatCapacity = std::max(0.0, sensibleHeatCapacity);
    result.supplyX = std::max(0.0, validData.indoorX);

    if (operationMode != OperationMode::Cooling) {
        result.regime = "non_cooling_passthrough";
        if (moistEnthalpyEnabled && airFlowRate > std::numeric_limits<double>::epsilon()) {
            const double qTotal = thermal_moist_air::processedEnthalpyHeatW(
                validData.indoorTemp,
                std::max(0.0, validData.indoorX),
                validData.airconTemp,
                result.supplyX,
                airFlowRate,
                /*heating=*/true);
            result.sensibleHeatCapacity = qTotal;
            result.latentHeatCapacity = 0.0;
        }
        return result;
    }
    if (!(airFlowRate > std::numeric_limits<double>::epsilon())) {
        result.regime = "no_flow";
        return result;
    }

    const double tIn = validData.indoorTemp;
    const double tOut = validData.airconTemp;
    const double xIn = std::max(0.0, validData.indoorX);

    // All cooling models share the same physical outlet constraint. The
    // sensible solution fixes tOut; water removal is derived from the final x.
    // Leave a sub-ppm margin for float32 temperature/humidity artifact rounding.
    result.supplyXUpperBound = std::min(
        saturationHumidityUpperBound(tOut), validData.additionalSupplyXUpperBound);
    auto rhPercent = [&](double x) {
        const double vaporPressure = archenv::STANDARD_ATMOSPHERIC_PRESSURE * x / (0.622 + x);
        return 100.0 * vaporPressure / archenv::saturation_vapor_pressure(tOut);
    };
    auto finishCooling = [&]() {
        result.unboundedSupplyRhPercent = rhPercent(std::max(0.0, result.supplyX));
        result.saturationLimited = result.supplyX > result.supplyXUpperBound;
        result.supplyX = std::clamp(result.supplyX, 0.0, std::min(xIn, result.supplyXUpperBound));
        result.supplyRhPercent = rhPercent(result.supplyX);
        result.condensationRateKgPerS = kAirDensity * std::abs(airFlowRate) * (xIn - result.supplyX);
        result.latentHeatCapacity = result.condensationRateKgPerS * archenv::vapor_latent_heat(tOut);
        if (moistEnthalpyEnabled) {
            const double qTotal = thermal_moist_air::processedEnthalpyHeatW(
                tIn, xIn, tOut, result.supplyX, airFlowRate, /*heating=*/false);
            const double cpEff = 0.5 * (thermal_moist_air::moistAirCp(xIn) +
                                       thermal_moist_air::moistAirCp(result.supplyX));
            result.sensibleHeatCapacity = std::min(
                qTotal, kAirDensity * std::abs(airFlowRate) * cpEff * std::max(0.0, tIn - tOut));
            result.latentHeatCapacity = std::max(0.0, qTotal - result.sensibleHeatCapacity);
        }
        return result;
    };

    if (aircon::humidity_control::enabled(nodeProps)) {
        const bool reheat = aircon::humidity_control::simultaneous(nodeProps);
        const double request = std::isfinite(nodeProps.humidity_supply_request)
            ? nodeProps.humidity_supply_request : xIn;
        const double wetCoilFloor = saturationHumidityUpperBound(nodeProps.min_coil_temperature);
        result.regime = reheat ? "room_humidistat_reheat" : "room_humidistat_overcool";
        // Humidity priority has a saturated ideal coil, with no reheat. The
        // controller searches the room temperature within its overcool bound.
        const double desiredX = reheat ? std::max(wetCoilFloor, request) : xIn;
        auto evaluate = [&](double x) {
            result.supplyX = std::clamp(x, 0.0, std::min(xIn, result.supplyXUpperBound));
            finishCooling();
            result.coilTemp = tOut;
            if (reheat && result.supplyX < xIn - 1e-12)
                result.coilTemp = std::min(tOut, aircon::humidity_control::dewPoint(result.supplyX));
            result.coilX = result.supplyX;
            const double cp = moistEnthalpyEnabled
                ? thermal_moist_air::moistAirCp(result.supplyX) : kAirSpecificHeat;
            result.reheatCapacity = reheat
                ? kAirDensity * std::abs(airFlowRate) * cp * std::max(0.0, tOut - result.coilTemp) : 0.0;
            // Also account for sensible heating after a dry/pass-through coil.
            if (reheat && result.coilTemp > tIn) {
                result.reheatCapacity += kAirDensity * std::abs(airFlowRate) * cp * (result.coilTemp - tIn);
                result.coilTemp = tIn;
            }
            result.sensibleHeatCapacity = kAirDensity * std::abs(airFlowRate) *
                (moistEnthalpyEnabled ? 0.5 * (thermal_moist_air::moistAirCp(xIn) + cp) : cp) *
                std::max(0.0, tIn - result.coilTemp);
            result.latentHeatCapacity = result.condensationRateKgPerS * archenv::vapor_latent_heat(result.coilTemp);
            if (moistEnthalpyEnabled) {
                const double total = thermal_moist_air::processedEnthalpyHeatW(
                    tIn, xIn, result.coilTemp, result.supplyX, airFlowRate, false);
                result.sensibleHeatCapacity = std::min(total, result.sensibleHeatCapacity);
                result.latentHeatCapacity = std::max(0.0, total - result.sensibleHeatCapacity);
            }
        };
        evaluate(desiredX);
        if (reheat) {
            std::string source;
            const auto limit = aircon::capacity::resolveMaxHeatCapacity(nodeProps, operationMode, source);
            if (limit && totalHeatCapacity(result) > *limit) {
                double lo = result.supplyX, hi = std::min(xIn, result.supplyXUpperBound);
                evaluate(hi);
                if (totalHeatCapacity(result) <= *limit) {
                    // Meet the temperature load first when the combined
                    // request is infeasible; keep the RH shortfall visible.
                    for (int i = 0; i < 50; ++i) {
                        const double mid = 0.5 * (lo + hi);
                        evaluate(mid);
                        if (totalHeatCapacity(result) > *limit) lo = mid;
                        else hi = mid;
                    }
                    evaluate(hi);
                    result.humidityCapacityLimited = true;
                }
                // If even the warmest/driest-feasible state exceeds capacity,
                // the common temperature/capacity search handles it.
            }
        }
        return result;
    }

    // Temperature priority has no independent dehumidifier or reheat coil.
    // pre_rh is an unmet-humidity threshold only: water removal must follow
    // the temperature-driven cooling process and the selected latent method.
    // Active room humidity requests are handled by the two modes above.

    if (!(tIn > tOut)) {
        result.regime = "no_temperature_drop";
        return finishCooling();
    }

    const std::string latentMethod = readLatentMethod(nodeProps);
    if (latentMethod == "none") {
        result.regime = "latent_disabled";
        return finishCooling();
    }

    auto applyRh95 = [&]() {
        result.regime = "rh95";
        const double x95 = std::max(0.0, archenv::absolute_humidity(tOut, kDefaultSupplyRhPercent));
        result.supplyX = std::min(xIn, x95);
    };

    if (latentMethod == "bf") {
        result.regime = "bypass_factor";
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

    } else {
        if (latentMethod == "coil_aoaf" || latentMethod == "aoaf" || latentMethod == "literature") {
            result.regime = "coil_dry_or_inactive";
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
                // A cooling coil cannot humidify its tentative outlet state.
                // At the dew-point boundary xr == x_sat(td), this makes
                // xStar <= xr, so the positive latent driving difference
                // approaches zero instead of jumping to a finite load.
                const double xe = std::min(xr, std::max(0.0, archenv::absolute_humidity(te, 100.0)));

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
                            result.regime = "coil_wet";
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

    return finishCooling();
}

namespace {
double supplyHumidityRecomputeTol(double humidityAbsTol) {
    return std::min((humidityAbsTol > 0.0) ? humidityAbsTol : 1e-9, 1e-7);
}
} // namespace

bool supplyHumidityNeedsUpdate(const VertexProperties& node,
                                const LatentProcessResult& loads,
                                double humidityAbsTol) {
    return !std::isfinite(node.current_x) ||
        node.current_x > loads.supplyXUpperBound ||
        std::abs(node.current_x - loads.supplyX) > supplyHumidityRecomputeTol(humidityAbsTol);
}

bool applySupplyHumidityToAirconNode(ThermalNetwork& thermalNetwork,
                                     const std::string& airconKey,
                                     const LatentProcessResult& loads,
                                     double humidityAbsTol) {
    auto& node = thermalNetwork.getNode(airconKey);
    const double newRemoval = std::max(0.0, loads.condensationRateKgPerS);
    const bool wasPassthrough = airconHumidityPassthrough(node);
    node.aircon_moisture_removal_kg_s = newRemoval;
    const bool passthrough = airconHumidityPassthrough(node);
    if (passthrough) return !wasPassthrough;
    const bool changed = wasPassthrough || supplyHumidityNeedsUpdate(node, loads, humidityAbsTol);
    // Relax the boundary iteration, but test its unrelaxed defect above.
    // The hard saturation bound is applied immediately and triggers a re-solve.
    if (changed) {
        const double relaxation = aircon::humidity_control::enabled(node) &&
            !aircon::humidity_control::simultaneous(node) ? 1.0 : 0.5;
        const double mixed = std::isfinite(node.current_x)
            ? node.current_x + relaxation * (loads.supplyX - node.current_x) : loads.supplyX;
        node.current_x = std::min(mixed, loads.supplyXUpperBound);
    }
    return changed;
}

bool applyPassthroughHumidityToAirconNode(ThermalNetwork& thermalNetwork,
                                          const std::string& airconKey,
                                          double humidityAbsTol) {
    auto& node = thermalNetwork.getNode(airconKey);
    (void)humidityAbsTol;
    const bool changed = !airconHumidityPassthrough(node);
    node.aircon_moisture_removal_kg_s = 0.0;
    return changed;
}

} // namespace aircon::latent
