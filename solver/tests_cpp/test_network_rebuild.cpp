#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "network/thermal_network.h"
#include "network/ventilation_network.h"

namespace {

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

VertexProperties makeNode(const std::string& key) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.calc_p = true;
    v.calc_t = true;
    v.current_p = 0.0;
    v.current_t = 20.0;
    return v;
}

EdgeProperties makeVentEdge(const std::string& key, const std::string& s, const std::string& t) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "fixed_flow";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.current_vol = 1.0;
    e.vol = {1.0};
    return e;
}

EdgeProperties makeThermalEdge(const std::string& key, const std::string& s, const std::string& t) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "conductance";
    e.subtype = "conduction";
    e.source = s;
    e.target = t;
    e.conductance = 1.0;
    e.area = 1.0;
    e.current_enabled = true;
    return e;
}

SimulationConstants makeConstants() {
    SimulationConstants c{};
    c.logVerbosity = 0;
    c.pressureCalc = false;     // ここでは network 構築の安定性だけを見る
    c.temperatureCalc = true;   // ThermalNetwork を作るため
    return c;
}

} // namespace

int main() {
    const auto constants = makeConstants();
    std::ostringstream logs;

    const std::vector<VertexProperties> nodes = {makeNode("A"), makeNode("B")};
    const std::vector<EdgeProperties> ventEdges = {makeVentEdge("A->B", "A", "B")};
    const std::vector<EdgeProperties> thEdges = {makeThermalEdge("T_A->B", "A", "B")};

    // VentilationNetwork: build twice should not accumulate vertices/edges
    {
        VentilationNetwork net;
        net.buildFromData(nodes, ventEdges, constants, logs);
        const int n1 = net.getNodeCount();
        const int e1 = net.getEdgeCount();
        net.buildFromData(nodes, ventEdges, constants, logs);
        const int n2 = net.getNodeCount();
        const int e2 = net.getEdgeCount();
        expectTrue(n1 == 2 && e1 == 1, "vent: first build expected 2 nodes, 1 edge");
        expectTrue(n2 == 2 && e2 == 1, "vent: rebuild should not accumulate");
    }

    // ThermalNetwork: build twice should not accumulate vertices/edges (includes advection edges from ventEdges)
    {
        ThermalNetwork net;
        net.buildFromData(nodes, thEdges, ventEdges, constants, logs);
        const int n1 = net.getNodeCount();
        const int e1 = net.getEdgeCount();
        net.buildFromData(nodes, thEdges, ventEdges, constants, logs);
        const int n2 = net.getNodeCount();
        const int e2 = net.getEdgeCount();
        expectTrue(n1 == 2, "thermal: first build expected 2 nodes");
        expectTrue(e1 == 2, "thermal: first build expected 2 edges (conduction + advection)");
        expectTrue(n2 == 2 && e2 == 2, "thermal: rebuild should not accumulate");
    }

    std::cout << "[OK] all tests passed\n";
    return 0;
}


