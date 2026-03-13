#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "archenv.h"
#include "nlohmann/json.hpp"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectNear(double actual, double expected, double tol, const std::string& name) {
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        fail(name + " mismatch (actual=" + std::to_string(actual) + ", expected=" +
             std::to_string(expected) + ", diff=" + std::to_string(diff) + ", tol=" +
             std::to_string(tol) + ")");
    }
}

double cppValue(const nlohmann::json& constants, const std::string& key) {
    if (!constants.contains(key)) {
        fail("missing key in constants registry: " + key);
        return 0.0;
    }
    const auto& node = constants.at(key);
    if (node.at("cpp_value").is_null()) {
        fail("cpp_value is null for key: " + key);
        return 0.0;
    }
    return node.at("cpp_value").get<double>();
}

} // namespace

int main() {
    std::ifstream ifs(ARCHENV_CONSTANTS_JSON_PATH);
    if (!ifs) {
        std::cerr << "[FAIL] cannot open constants registry: " << ARCHENV_CONSTANTS_JSON_PATH << "\n";
        return 1;
    }

    nlohmann::json j;
    ifs >> j;
    const auto& constants = j.at("constants");

    expectNear(
        archenv::STANDARD_ATMOSPHERIC_PRESSURE,
        cppValue(constants, "STANDARD_ATMOSPHERIC_PRESSURE"),
        1e-12,
        "STANDARD_ATMOSPHERIC_PRESSURE");
    expectNear(
        archenv::SPECIFIC_HEAT_AIR,
        cppValue(constants, "SPECIFIC_HEAT_AIR"),
        1e-12,
        "SPECIFIC_HEAT_AIR");
    expectNear(
        archenv::SPECIFIC_HEAT_WATER_VAPOR,
        cppValue(constants, "SPECIFIC_HEAT_WATER_VAPOR"),
        1e-12,
        "SPECIFIC_HEAT_WATER_VAPOR");
    expectNear(
        archenv::LATENT_HEAT_VAPORIZATION,
        cppValue(constants, "LATENT_HEAT_VAPORIZATION"),
        1e-12,
        "LATENT_HEAT_VAPORIZATION");
    expectNear(
        archenv::DENSITY_DRY_AIR,
        cppValue(constants, "DENSITY_DRY_AIR"),
        1e-12,
        "DENSITY_DRY_AIR");
    expectNear(
        archenv::KELVIN_OFFSET,
        cppValue(constants, "KELVIN_OFFSET"),
        1e-12,
        "KELVIN_OFFSET");

    if (g_failures == 0) {
        std::cout << "[OK] archenv constants match registry\n";
        return 0;
    }
    return 1;
}
