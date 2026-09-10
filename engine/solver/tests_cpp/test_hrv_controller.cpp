#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "hrv/hrv_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "types/graph_types.h"

#include <boost/graph/adjacency_list.hpp>

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (!(std::abs(actual - expected) <= tol)) {
        fail(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
    }
}

VertexProperties makeNode(const std::string& key, const std::string& type, double t, double x) {
    VertexProperties v{};
    v.key = key;
    v.type = type;
    v.current_t = t;
    v.current_x = x;
    return v;
}

} // namespace

int main() {
    ThermalNetwork thermal;
    thermal.addNode(makeNode("OA", "normal", 0.0, 0.002));
    thermal.addNode(makeNode("ROOM", "normal", 20.0, 0.008));
    {
        auto sa = makeNode("HRV1", "hrv", 0.0, 0.0);
        sa.outside_node = "OA";
        sa.in_node = "ROOM";
        sa.set_node = "ROOM";
        sa.model = "total";
        sa.ac_spec = nlohmann::json{
            {"eta_t", 0.5},
            {"eta_x", 0.4},
            {"recovery", "total"},
            {"exhaust_node", "HRV1_exhaust"},
        };
        thermal.addNode(sa);
        thermal.addNode(makeNode("HRV1_exhaust", "normal", 0.0, 0.0));
    }

    std::ostringstream logs;
    hrv::updateSupplyBoundaries(thermal, logs, 0);
    const auto& g = thermal.getGraph();
    const auto& keyToV = thermal.getKeyToVertex();
    const auto& sa = g[keyToV.at("HRV1")];
    expectNear(sa.current_t, 10.0, 1e-9, "T_sa = 0 + 0.5*(20-0)");
    expectNear(sa.current_x, 0.0044, 1e-12, "x_sa = 0.002 + 0.4*(0.008-0.002)");
    const auto& ea = g[keyToV.at("HRV1_exhaust")];
    expectNear(ea.current_t, 20.0, 1e-9, "exhaust mirrors room T");
    expectNear(ea.current_x, 0.008, 1e-12, "exhaust mirrors room x");

    FlowRateMap flows;
    flows[{"OA", "HRV1"}] = 0.1;
    flows[{"ROOM", "HRV1_exhaust"}] = 0.1;
    auto sens = hrv::collectSensibleRecoveredW(thermal, flows);
    auto lat = hrv::collectLatentRecoveredW(thermal, flows);
    expectTrue(sens.size() == 1 && lat.size() == 1, "one HRV key");
    expectTrue(sens[0] > 0.0, "sensible recovered > 0");
    expectTrue(lat[0] > 0.0, "latent recovered > 0 for total");

    // 顕熱のみは潜熱回収 0
    {
        auto& node = thermal.getNode("HRV1");
        node.model = "sensible";
        node.ac_spec["recovery"] = "sensible";
        node.ac_spec["eta_x"] = 0.4;
        hrv::updateSupplyBoundaries(thermal, logs, 0);
        expectNear(thermal.getNode("HRV1").current_x, 0.002, 1e-12, "sensible: x_sa = x_oa");
        lat = hrv::collectLatentRecoveredW(thermal, flows);
        expectNear(lat[0], 0.0, 1e-9, "sensible: latent recovered = 0");
    }

    if (g_failures == 0) {
        std::cout << "OK\n";
        return 0;
    }
    std::cerr << g_failures << " failure(s)\n";
    return 1;
}
