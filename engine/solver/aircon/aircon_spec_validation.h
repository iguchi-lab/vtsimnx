#pragma once
#include "aircon/aircon_operation_mode.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace aircon {
inline void validateSpec(const nlohmann::json& spec, const std::string& model,
                         const std::string& control, const std::vector<std::string>& modes, double offFlow = 0.0) {
    if (!spec.is_object()) throw std::runtime_error("ac_spec must be an object");
    const std::string method = toLowerCopy(spec.value("latent_method",
        control == "temperature_priority" ? std::string("rh95") : std::string("ideal_coil")));
    const std::set<std::string> methods{"rh95", "rh100", "none", "bf", "coil_aoaf", "aoaf", "literature", "ideal_coil"};
    if (!methods.count(method)) throw std::runtime_error("invalid ac_spec.latent_method: " + method);
    if (control != "temperature_priority" && method != "none" && method != "rh100" && method != "ideal_coil")
        throw std::runtime_error("humidistat latent_method requires ideal_coil (none/rh100 aliases)");
    if (control == "temperature_priority" && method == "ideal_coil")
        throw std::runtime_error("latent_method=ideal_coil requires a humidity control mode");
    for (const char* key : {"bf", "BF", "bypass_factor", "Af", "Ao", "coil_face_area", "coil_surface_area"}) {
        if (!spec.contains(key)) continue;
        if (!spec[key].is_number()) throw std::runtime_error(std::string("invalid ac_spec.") + key);
        const double v = spec[key].get<double>();
        const bool bf = std::string(key) == "bf" || std::string(key) == "BF" || std::string(key) == "bypass_factor";
        if (!std::isfinite(v) || (bf ? !(v >= 0 && v <= 0.99) : !(v > 0)))
            throw std::runtime_error(std::string("invalid ac_spec.") + key);
    }
    if (model != "DUCT_CENTRAL") return;
    std::set<std::string> active;
    for (const auto& mode : modes) {
        if (mode == "HEATING" || mode == "AUTO") active.insert("heating");
        if (mode == "COOLING" || mode == "AUTO") active.insert("cooling");
    }
    for (const auto& mode : active) {
        auto positive = [&](const char* group, const char* key) {
            const std::string path = std::string("ac_spec.") + group + "." + mode + "." + key;
            if (!spec.contains(group) || !spec[group].contains(mode) || !spec[group][mode].contains(key) ||
                !spec[group][mode][key].is_number()) throw std::runtime_error("missing positive " + path);
            const double v = spec[group][mode][key].get<double>();
            const bool nonnegative = std::string(group) == "P_fan" ||
                (std::string(group) == "Q" && std::string(key) == "min");
            if ((nonnegative ? v < 0 : !(v > 0)) || !std::isfinite(v)) throw std::runtime_error("invalid " + path);
            return v;
        };
        const double qmin = positive("Q", "min"), qmid = positive("Q", "mid"), qrtd = positive("Q", "rtd");
        if (!(qmin < qmid && qmid < qrtd)) throw std::runtime_error("DUCT_CENTRAL requires Q.min < mid < rtd");
        if (!(positive("P", "rtd") > positive("P_fan", "rtd")))
            throw std::runtime_error("DUCT_CENTRAL requires P.rtd > P_fan.rtd");
        for (const char* key : {"mid", "rtd", "dsgn"}) positive("V_inner", key);
    }
    if (offFlow > 0.0 && std::find(modes.begin(), modes.end(), "OFF") != modes.end()) {
        const std::string mode = spec.contains("V_inner") && spec["V_inner"].contains("cooling") ? "cooling" : "heating";
        const double power = spec.at("P_fan").at(mode).at("rtd").get<double>();
        const double volume = spec.at("V_inner").at(mode).at("dsgn").get<double>();
        if (!std::isfinite(power) || power < 0 || !std::isfinite(volume) || !(volume > 0))
            throw std::runtime_error("OFF ventilation requires valid P_fan.rtd and V_inner.dsgn");
    }
}
} // namespace aircon
