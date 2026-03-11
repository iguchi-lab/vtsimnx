#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/humidity/humidity_solver.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "transport/humidity_solver.h"

namespace {

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        std::ostringstream oss;
        oss << msg << " actual=" << actual << " expected=" << expected
            << " diff=" << diff << " tol=" << tol;
        throw std::runtime_error(oss.str());
    }
}

VertexProperties makeNode(const std::string& key) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.current_t = 20.0;
    v.current_x = 0.0;
    v.v = 100.0;
    return v;
}

EdgeProperties makeFixedFlowEdge(const std::string& key, const std::string& s, const std::string& t, double vol_m3s) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "fixed_flow";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.current_vol = vol_m3s;
    e.vol = {vol_m3s};
    e.eta = 0.0;
    return e;
}

SimulationConstants makeConstants() {
    SimulationConstants c{};
    c.logVerbosity = 0;
    c.pressureCalc = false;
    c.temperatureCalc = false;
    c.humidityCalc = true;
    c.concentrationCalc = false;
    c.timestep = 3600;
    return c;
}

} // namespace

int main() {
    std::ostringstream logs;
    TimingList timings;
    const auto constants = makeConstants();

    auto A = makeNode("A");
    auto B = makeNode("B");
    A.calc_x = false;
    B.calc_x = true;
    A.current_x = 0.010;
    B.current_x = 0.004;
    B.v = 60.0;

    std::vector<VertexProperties> nodes = {A, B};
    std::vector<EdgeProperties> ventEdges = {
        makeFixedFlowEdge("A->B", "A", "B", 0.08),
    };
    std::vector<EdgeProperties> thEdges = {};

    VentilationNetwork ventCore;
    ThermalNetwork thermalCore;
    ventCore.buildFromData(nodes, ventEdges, constants, logs);
    thermalCore.buildFromData(nodes, thEdges, ventEdges, constants, logs);
    ventCore.updatePropertiesForTimestep(nodes, ventEdges, 0);
    const FlowRateMap flowCore = ventCore.collectFlowRateMap();

    VentilationNetwork ventTransport;
    ThermalNetwork thermalTransport;
    ventTransport.buildFromData(nodes, ventEdges, constants, logs);
    thermalTransport.buildFromData(nodes, thEdges, ventEdges, constants, logs);
    ventTransport.updatePropertiesForTimestep(nodes, ventEdges, 0);
    const FlowRateMap flowTransport = ventTransport.collectFlowRateMap();

    core::humidity::updateHumidityIfEnabled(constants, ventCore, thermalCore, flowCore, logs, timings, "core");
    transport::updateHumidityIfEnabled(constants, ventTransport, thermalTransport, flowTransport, logs, timings, "transport");

    const auto& coreMap = thermalCore.getKeyToVertex();
    const auto& trMap = thermalTransport.getKeyToVertex();
    const double coreX = thermalCore.getGraph()[coreMap.at("B")].current_x;
    const double trX = thermalTransport.getGraph()[trMap.at("B")].current_x;
    const double coreW = thermalCore.getGraph()[coreMap.at("B")].current_w;
    const double trW = thermalTransport.getGraph()[trMap.at("B")].current_w;

    expectNear(coreX, trX, 1e-12, "core and transport humidity entry should be equivalent");
    expectNear(coreW, trW, 1e-12, "core and transport moisture state should be equivalent");

    std::cout << "[OK] all tests passed\n";
    return 0;
}

