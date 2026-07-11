#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "network/ventilation_network.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

template <class Fn>
void expectThrows(Fn fn, const std::string& msg) {
    try {
        fn();
        fail(msg + " (expected throw)");
    } catch (const std::exception&) {
        // ok
    }
}

VertexProperties makeNode(const std::string& key) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.v = 1.0;
    return v;
}

EdgeProperties makeEdge(const std::string& uniqueId, const std::string& s, const std::string& t) {
    EdgeProperties e{};
    e.key = uniqueId;
    e.unique_id = uniqueId;
    e.type = "fixed_flow";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.current_vol = 0.1;
    e.vol = {0.1};
    return e;
}

} // namespace

int main() {
    SimulationConstants constants{};
    constants.logVerbosity = 0;
    constants.pressureCalc = true;
    std::ostringstream logs;

    {
        std::vector<VertexProperties> nodes = {makeNode("A"), makeNode("B")};
        std::vector<EdgeProperties> edges = {
            makeEdge("duct", "A", "B"),
            makeEdge("duct_000", "A", "B"),
        };
        VentilationNetwork vent;
        vent.buildFromData(nodes, edges, constants, logs);
        expectThrows([&]() { (void)vent.getFlowRateKeys(); },
                     "normalized flow_rate key collision should throw");
    }

    {
        std::vector<VertexProperties> nodes = {makeNode("A"), makeNode("B")};
        std::vector<EdgeProperties> edges = {
            makeEdge("duct_a", "A", "B"),
            makeEdge("duct_b_000", "A", "B"),
        };
        VentilationNetwork vent;
        vent.buildFromData(nodes, edges, constants, logs);
        const auto keys = vent.getFlowRateKeys();
        expectTrue(keys.size() == 2, "distinct normalized keys should succeed");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}
