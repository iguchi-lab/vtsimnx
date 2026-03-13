#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
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

EdgeProperties makeVentEdgeWithVol(const std::string& key,
                                   const std::string& uniqueId,
                                   const std::string& s,
                                   const std::string& t,
                                   double vol) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = uniqueId;
    e.type = "fixed_flow";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.current_vol = vol;
    e.vol = {vol};
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

    // ThermalNetwork: duplicate source/target advection edges should each receive flow_rate
    {
        const std::vector<EdgeProperties> ventEdgesDup = {
            makeVentEdgeWithVol("A->B(1)", "A->B(1)", "A", "B", 1.0),
            makeVentEdgeWithVol("A->B(2)", "A->B(2)", "A", "B", 2.0),
        };
        const std::vector<EdgeProperties> thEdgesEmpty = {};

        VentilationNetwork ventNet;
        ThermalNetwork thermalNet;
        ventNet.buildFromData(nodes, ventEdgesDup, constants, logs);
        thermalNet.buildFromData(nodes, thEdgesEmpty, ventEdgesDup, constants, logs);
        thermalNet.syncFlowRatesFromVentilationNetwork(ventNet);

        std::vector<double> duplicatedPairFlows;
        for (auto e : boost::make_iterator_range(boost::edges(thermalNet.getGraph()))) {
            const auto& ep = thermalNet.getGraph()[e];
            if (ep.getTypeCode() != EdgeProperties::TypeCode::Advection) continue;
            if (ep.source == "A" && ep.target == "B") {
                duplicatedPairFlows.push_back(ep.flow_rate);
            }
        }

        expectTrue(duplicatedPairFlows.size() == 2, "thermal: duplicate A->B advection edges should exist");
        std::sort(duplicatedPairFlows.begin(), duplicatedPairFlows.end());
        expectTrue(std::abs(duplicatedPairFlows[0] - 1.0) < 1e-12,
                   "thermal: first duplicate A->B advection edge should keep its own flow_rate");
        expectTrue(std::abs(duplicatedPairFlows[1] - 2.0) < 1e-12,
                   "thermal: second duplicate A->B advection edge should keep its own flow_rate");
    }

    std::cout << "[OK] all tests passed\n";
    return 0;
}


