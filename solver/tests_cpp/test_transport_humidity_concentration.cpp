#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "transport/humidity_solver.h"
#include "transport/concentration_solver.h"

namespace {

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (!(std::isfinite(actual) && std::isfinite(expected))) {
        throw std::runtime_error(msg + " (non-finite)");
    }
    const double diff = std::abs(actual - expected);
    if (diff > tol) {
        std::ostringstream oss;
        oss << msg << " actual=" << actual << " expected=" << expected << " diff=" << diff << " tol=" << tol;
        throw std::runtime_error(oss.str());
    }
}

VertexProperties makeNode(const std::string& key) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.current_p = 0.0;
    v.current_t = 20.0;
    v.current_x = 0.0;
    v.current_c = 0.0;
    v.current_beta = 0.0;
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
    e.current_humidity_generation = 0.0;
    e.current_dust_generation = 0.0;
    return e;
}

SimulationConstants makeConstants() {
    SimulationConstants c{};
    c.logVerbosity = 0;
    c.pressureCalc = false;
    c.temperatureCalc = false;
    c.humidityCalc = true;
    c.concentrationCalc = true;
    c.timestep = 3600; // 1h
    return c;
}

} // namespace

int main() {
    std::ostringstream logs;
    TimingList timings;
    const auto constants = makeConstants();

    // ------------------------------------------------------------------
    // 1) humidity: 1-step implicit update (m3/s -> kg/s conversion included)
    // ------------------------------------------------------------------
    {
        auto V0 = makeNode("void");
        V0.v = 0.0;
        auto A = makeNode("A");
        auto B = makeNode("B");
        A.calc_x = false;  // 境界（固定値想定）
        B.calc_x = true;   // 更新対象
        A.current_x = 0.010; // kg/kg(DA)
        B.current_x = 0.005;
        B.v = 100.0;

        std::vector<VertexProperties> nodes = {V0, A, B};
        std::vector<EdgeProperties> ventEdges = {
            makeFixedFlowEdge("A->B", "A", "B", 0.1), // 0.1 m3/s
            makeFixedFlowEdge("B->void", "B", "void", 0.1), // mass balance
        };
        std::vector<EdgeProperties> thEdges = {};

        VentilationNetwork vent;
        ThermalNetwork thermal;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);

        // fixed_flow の flow_rate を反映（collectFlowRateMap が使えるように）
        vent.updatePropertiesForTimestep(nodes, ventEdges, 0);

        FlowRateMap flowRates = vent.collectFlowRateMap(); // (A,B)->0.1
        transport::updateHumidityIfEnabled(constants, vent, thermal, flowRates, logs, timings, "test");

        // balanced flow (A->B and B->void):
        // dx/dt = (q/V)*(x_src - x), implicit Euler -> x_{n+1} = (x_n + a*x_src)/(1+a)
        const double dt = static_cast<double>(constants.timestep);
        const double q = 0.1;      // m3/s
        const double V = 100.0;    // m3
        const double a = dt * (q / V);
        const double expected = (0.005 + a * 0.010) / (1.0 + a);

        const auto& tG = thermal.getGraph();
        const auto& tMap = thermal.getKeyToVertex();
        const auto itB = tMap.find("B");
        if (itB == tMap.end()) {
            std::ostringstream oss;
            oss << "missing node B in thermal. keys={";
            bool first = true;
            for (const auto& kv : tMap) {
                if (!first) oss << ",";
                first = false;
                oss << kv.first;
            }
            oss << "}";
            throw std::runtime_error(oss.str());
        }
        const double actual = tG[itB->second].current_x;
        expectNear(actual, expected, 1e-12, "humidity implicit update");
    }

    // ------------------------------------------------------------------
    // 2) concentration: beta decay only (no inflow/outflow/generation)
    //    c(t+dt) = c(t) * exp(-beta*dt)
    // ------------------------------------------------------------------
    {
        auto V0 = makeNode("void");
        V0.v = 0.0;
        auto A = makeNode("A");
        auto B = makeNode("B");
        A.calc_c = false;
        B.calc_c = true;
        A.current_c = 0.0;
        B.current_c = 100.0;
        B.v = 100.0;
        B.current_beta = 1e-5; // 1/s

        std::vector<VertexProperties> nodes = {V0, A, B};
        // ノードをネットワークに含めるためのダミー（流量0）
        std::vector<EdgeProperties> ventEdges = {
            makeFixedFlowEdge("B->void", "B", "void", 0.0),
        };
        std::vector<EdgeProperties> thEdges = {};

        VentilationNetwork vent;
        ThermalNetwork thermal;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);

        transport::updateConcentrationIfEnabled(constants, vent, thermal, logs, timings, "test");

        const double dt = static_cast<double>(constants.timestep);
        const double expected = 100.0 * std::exp(-1e-5 * dt);
        const auto& tG = thermal.getGraph();
        const auto& tMap = thermal.getKeyToVertex();
        const auto itB = tMap.find("B");
        if (itB == tMap.end()) {
            std::ostringstream oss;
            oss << "missing node B in thermal. keys={";
            bool first = true;
            for (const auto& kv : tMap) {
                if (!first) oss << ",";
                first = false;
                oss << kv.first;
            }
            oss << "}";
            throw std::runtime_error(oss.str());
        }
        const double actual = tG[itB->second].current_c;
        expectNear(actual, expected, 1e-10, "concentration beta decay");
    }

    // ------------------------------------------------------------------
    // 3) concentration: generation + beta (dust_generation on void->B)
    // ------------------------------------------------------------------
    {
        auto V0 = makeNode("void");
        V0.v = 0.0;
        auto A = makeNode("A");
        auto B = makeNode("B");
        A.calc_c = false;
        B.calc_c = true;
        B.current_c = 100.0;
        B.v = 100.0;
        B.current_beta = 1e-5; // 1/s

        std::vector<VertexProperties> nodes = {V0, A, B};
        EdgeProperties gen = makeFixedFlowEdge("void->B", "void", "B", 0.0);
        gen.current_dust_generation = 10.0; // [count/s]
        gen.dust_generation = {10.0};
        std::vector<EdgeProperties> ventEdges = {gen};
        std::vector<EdgeProperties> thEdges = {};

        VentilationNetwork vent;
        ThermalNetwork thermal;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);
        vent.updatePropertiesForTimestep(nodes, ventEdges, 0);

        transport::updateConcentrationIfEnabled(constants, vent, thermal, logs, timings, "test");

        // k1 = m/V, k2 = beta
        const double dt = static_cast<double>(constants.timestep);
        const double m = 10.0;
        const double V = 100.0;
        const double beta = 1e-5;
        const double k1 = m / V;
        const double k2 = beta;
        const double k = k1 / k2;
        const double expected = (100.0 - k) * std::exp(-k2 * dt) + k;

        const auto& tG = thermal.getGraph();
        const auto& tMap = thermal.getKeyToVertex();
        const auto itB = tMap.find("B");
        if (itB == tMap.end()) {
            std::ostringstream oss;
            oss << "missing node B in thermal. keys={";
            bool first = true;
            for (const auto& kv : tMap) {
                if (!first) oss << ",";
                first = false;
                oss << kv.first;
            }
            oss << "}";
            throw std::runtime_error(oss.str());
        }
        const double actual = tG[itB->second].current_c;
        expectNear(actual, expected, 1e-8, "concentration generation+beta");
    }

    std::cout << "[OK] all tests passed\n";
    return 0;
}


