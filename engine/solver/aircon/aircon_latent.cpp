#include "network/humidity_network.h"
#include "aircon/aircon_latent.h"

#include "aircon/aircon_network_utils.h"
#include "aircon/aircon_humidity_control.h"
#include "aircon/aircon_operation_mode.h"
#include "archenv/include/archenv.h"
#include "core/thermal/thermal_moist_air.h"
#include "../../acmodel/coil_transfer.h"
#include "network/thermal_network.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {
constexpr double kAirDensity = archenv::DENSITY_DRY_AIR;         // [kg/m^3]
constexpr double kAirSpecificHeat = archenv::SPECIFIC_HEAT_AIR;   // [J/(kg·K)]
constexpr double kDefaultOuterFlowRate = 25.5 / 60.0;             // m^3/s
constexpr double kDefaultBypassFactor = 0.1;                      // [-]
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
        return nodeProps.control_mode == "temperature_priority" ? "rh95" : "ideal_coil";
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

inline std::optional<double> referenceFlow(const VertexProperties& node) {
    if (auto v = readFinitePositiveSpecNumber(node.ac_spec, "bf_reference_flow_rate")) return v;
    const auto& spec = node.ac_spec;
    if (spec.is_object() && spec.contains("V_inner") && spec["V_inner"].is_object() &&
        spec["V_inner"].contains("cooling"))
        return readFinitePositiveSpecNumber(spec["V_inner"]["cooling"], "rtd");
    return std::nullopt;
}

} // namespace

namespace aircon::latent {

bool usesPhysicalCoil(const VertexProperties& node) {
    const std::string method = readLatentMethod(node);
    return method == "bf" || method == "coil_aoaf" || method == "aoaf" || method == "literature";
}

double effectiveBypassFactor(const VertexProperties& node, double actualFlowM3s) {
    const double referenceBF = readBypassFactor(node);
    if (!(actualFlowM3s > 0.0) || referenceBF == 0.0) return 0.0;
    if (node.ac_spec.is_object() && !node.ac_spec.value("bf_flow_correction", true)) return referenceBF;
    const auto reference = referenceFlow(node);
    // Older/minimal models without a rated/reference airflow retain a fixed BF.
    if (!reference) return referenceBF;
    return std::exp(std::log(referenceBF) * *reference / actualFlowM3s);
}

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
    result.coilTemp = tOut;
    result.coilX = xIn;

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

    if (aircon::humidity_control::simultaneous(nodeProps)) {
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
    if (latentMethod == "none" || latentMethod == "rh100" || latentMethod == "ideal_coil") {
        result.regime = "saturation_only";
        return finishCooling();
    }

    auto applyRh95 = [&]() {
        result.regime = "rh95";
        const double x95 = std::max(0.0, archenv::absolute_humidity(tOut, kDefaultSupplyRhPercent));
        result.supplyX = std::min(xIn, x95);
    };

    if (usesPhysicalCoil(nodeProps)) {
        const double flow = std::abs(airFlowRate);
        double contact = 0.0;
        if (latentMethod == "bf") {
            result.effectiveBypassFactor = effectiveBypassFactor(nodeProps, flow);
            contact = 1.0 - result.effectiveBypassFactor;
        } else {
            const double coefficient = acmodel::coolingCoilMassTransferCoefficient(
                flow, readCoilFaceArea(nodeProps));
            const double ntu = coefficient * readCoilSurfaceArea(nodeProps) / (kAirDensity * flow);
            result.effectiveBypassFactor = std::exp(-ntu);
            contact = -std::expm1(-ntu);
        }
        // Integrate heat and water transfer through ONE uniform-temperature
        // surface. Lewis relation gives the same contact effectiveness for T
        // and x; no guessed outlet humidity or arithmetic-mean overshoot.
        const double effectiveness = std::max(contact, 1e-12);
        result.coilTemp = tIn - (tIn-tOut) / effectiveness;
        // Capacity trials can request physically unreachable coil temperatures.
        // Keep psychrometrics finite; the shared coil-floor search bounds the
        // accepted solution, rather than hiding the infeasible temperature.
        result.coilX = std::max(0.0, archenv::absolute_humidity(
            std::max(-80.0, result.coilTemp), 100.0));
        result.supplyX = std::min(xIn, xIn + contact * (result.coilX-xIn));
        result.regime = latentMethod == "bf" ? "bypass_factor" :
            (result.coilX < xIn ? "coil_wet" : "coil_dry_or_inactive");
    } else {
        applyRh95();
    }

    return finishCooling();
}

namespace {
LatentProcessResult coolingAtTemperature(const AirconValidationData& data,
        const VertexProperties& node, double flow, bool moist, double temperature) {
    auto probe = data;
    probe.airconTemp = temperature;
    probe.additionalSupplyXUpperBound = std::numeric_limits<double>::infinity();
    auto probeNode = node;
    probeNode.control_mode = "humidity_priority"; // evaluate the selected coil, without reheat
    const double sensible = kAirDensity * flow * kAirSpecificHeat * std::max(0.0, data.indoorTemp-temperature);
    return estimateLatentProcess(probe, OperationMode::Cooling, sensible, flow, probeNode, moist);
}
}

double minimumCoolingOutletTemperature(const AirconValidationData& data,
        const VertexProperties& node, double flow, bool moist) {
    const std::string method = readLatentMethod(node);
    if (aircon::humidity_control::simultaneous(node) ||
        (method != "bf" && method != "coil_aoaf" && method != "aoaf" && method != "literature"))
        return node.min_coil_temperature;
    if (data.indoorTemp <= node.min_coil_temperature) return node.min_coil_temperature;
    const auto probe = coolingAtTemperature(data, node, flow, moist, node.min_coil_temperature);
    const double bf = probe.effectiveBypassFactor;
    // Both physical coils have a flow-dependent, temperature-independent
    // contact factor: their surface limit has an exact inverse.
    return bf*data.indoorTemp + (1.0-bf)*node.min_coil_temperature;
}

double humidityRequestedOutletTemperature(const AirconValidationData& data,
        const VertexProperties& node, double flow, bool moist) {
    const std::string method = readLatentMethod(node);
    if (method == "ideal_coil" || method == "none" || method == "rh100")
        return std::min(data.indoorTemp, std::max(node.min_coil_temperature,
            aircon::humidity_control::dewPoint(node.humidity_supply_request)));
    double lo = minimumCoolingOutletTemperature(data, node, flow, moist);
    double hi = std::max(lo, data.indoorTemp);
    if (coolingAtTemperature(data, node, flow, moist, lo).supplyX > node.humidity_supply_request)
        return std::min(data.indoorTemp, lo); // infeasible: leave the room shortfall visible
    for (int i = 0; i < 35; ++i) {
        const double mid = (lo+hi)/2.0;
        if (coolingAtTemperature(data, node, flow, moist, mid).supplyX > node.humidity_supply_request) hi = mid;
        else lo = mid;
    }
    return std::min(data.indoorTemp, lo);
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
                                     double humidityAbsTol,
                                     double relaxation) {
    auto& node = thermalNetwork.getNode(airconKey);
    const bool limitChanged = node.humidity_capacity_limited != loads.humidityCapacityLimited;
    node.humidity_capacity_limited = loads.humidityCapacityLimited;
    const double newRemoval = std::max(0.0, loads.condensationRateKgPerS);
    const bool wasPassthrough = airconHumidityPassthrough(node);
    node.aircon_humidity_transition = !wasPassthrough && newRemoval == 0.0 &&
        supplyHumidityNeedsUpdate(node, loads, humidityAbsTol);
    node.aircon_moisture_removal_kg_s = newRemoval;
    const bool passthrough = airconHumidityPassthrough(node);
    if (passthrough) return !wasPassthrough || limitChanged;
    const bool changed = wasPassthrough || supplyHumidityNeedsUpdate(node, loads, humidityAbsTol);
    // Relax the boundary iteration, but test its unrelaxed defect above.
    // The hard saturation bound is applied immediately and triggers a re-solve.
    if (changed) {
        const double mixed = std::isfinite(node.current_x)
            ? node.current_x + relaxation * (loads.supplyX - node.current_x) : loads.supplyX;
        node.current_x = std::clamp(mixed, 0.0, loads.supplyXUpperBound);
    }
    return changed || limitChanged;
}

bool applyPassthroughHumidityToAirconNode(ThermalNetwork& thermalNetwork,
                                          const std::string& airconKey,
                                          double humidityAbsTol) {
    auto& node = thermalNetwork.getNode(airconKey);
    (void)humidityAbsTol;
    const bool changed = !airconHumidityPassthrough(node) || node.humidity_capacity_limited;
    node.humidity_capacity_limited = false;
    node.aircon_moisture_removal_kg_s = 0.0;
    node.aircon_humidity_transition = false;
    return changed;
}

} // namespace aircon::latent
