#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "network/humidity_network.h"
#include "network/contaminant_network.h"
#include "core/humidity/humidity_solver.h"
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
        HumidityNetwork humidity;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);

        // fixed_flow の flow_rate を反映（collectFlowRateMap が使えるように）
        vent.updatePropertiesForTimestep(nodes, ventEdges, 0);

        FlowRateMap flowRates = vent.collectFlowRateMap(); // (A,B)->0.1
        (void)core::humidity::updateHumidityIfEnabled(constants,
                                                      vent,
                                                      thermal.getGraph(),
                                                      static_cast<const ThermalNetwork&>(thermal).nodeStateView(),
                                                      humidity,
                                                      flowRates,
                                                      logs,
                                                      timings,
                                                      "test");

        // humidity_x keys: calc_x=true の B に加え、v<=0 の void も出力対象になるはず
        const auto& humidityKeys = humidity.getOutputKeys(static_cast<const ThermalNetwork&>(thermal).nodeStateView());
        bool hasVoid = false;
        bool hasB = false;
        for (const auto& k : humidityKeys) {
            if (k == "void") hasVoid = true;
            if (k == "B") hasB = true;
        }
        if (!hasVoid || !hasB) {
            std::ostringstream oss;
            oss << "humidity_x keys missing expected entries. hasVoid=" << (hasVoid ? "true" : "false")
                << " hasB=" << (hasB ? "true" : "false") << " keys={";
            bool firstKey = true;
            for (const auto& k : humidityKeys) {
                if (!firstKey) oss << ",";
                firstKey = false;
                oss << k;
            }
            oss << "}";
            throw std::runtime_error(oss.str());
        }

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
        ContaminantNetwork contaminant;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);

        transport::updateConcentrationIfEnabled(constants,
                                                vent,
                                                thermal.getGraph(),
                                                static_cast<const ThermalNetwork&>(thermal).nodeStateView(),
                                                contaminant,
                                                logs,
                                                timings,
                                                "test");

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
        ContaminantNetwork contaminant;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);
        vent.updatePropertiesForTimestep(nodes, ventEdges, 0);

        transport::updateConcentrationIfEnabled(constants,
                                                vent,
                                                thermal.getGraph(),
                                                static_cast<const ThermalNetwork&>(thermal).nodeStateView(),
                                                contaminant,
                                                logs,
                                                timings,
                                                "test");

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

    // ------------------------------------------------------------------
    // 4) humidity: duplicate edges on same (src,dst) pair must be SUMMED
    //    (not overwritten). Scenario: 24h-ventilation branch always-on
    //    plus a schedule-based branch both create SRC->ROOM edges.
    //    When schedule=0, the 24h flow must still reach ROOM.
    // ------------------------------------------------------------------
    {
        // Nodes: SRC (x=0.01, boundary), ROOM (calc_x, v=50), EXT (boundary)
        auto SRC  = makeNode("SRC");
        SRC.v = 0.0;
        SRC.calc_x = false;
        SRC.current_x = 0.010;

        auto ROOM = makeNode("ROOM");
        ROOM.calc_x = true;
        ROOM.current_x = 0.0;
        ROOM.v = 50.0;

        auto EXT = makeNode("EXT");
        EXT.v = 0.0;
        EXT.calc_x = false;
        EXT.current_x = 0.001;

        // edge1: SRC->ROOM (24h ventilation, always-on, q=0.05 m3/s)
        // edge2: SRC->ROOM (schedule-based, q=0.0 at this timestep)
        // edge3: ROOM->EXT (always-on exhaust, q=0.05)
        EdgeProperties e1 = makeFixedFlowEdge("24h_SRC->ROOM", "SRC", "ROOM", 0.05);
        EdgeProperties e2 = makeFixedFlowEdge("sch_SRC->ROOM", "SRC", "ROOM", 0.0); // schedule=0
        EdgeProperties e3 = makeFixedFlowEdge("ROOM->EXT",     "ROOM", "EXT",  0.05);

        std::vector<VertexProperties> nodes = {SRC, ROOM, EXT};
        std::vector<EdgeProperties> ventEdges = {e1, e2, e3};
        std::vector<EdgeProperties> thEdges   = {};

        VentilationNetwork vent;
        ThermalNetwork thermal;
        HumidityNetwork humidity;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);
        vent.updatePropertiesForTimestep(nodes, ventEdges, 0);

        FlowRateMap flowRates = vent.collectFlowRateMap();

        // SRC->ROOM flow must equal 24h flow (0.05), not be zeroed by schedule edge (0.0)
        const double q_src_room = [&]() -> double {
            auto it = flowRates.find({"SRC", "ROOM"});
            return (it != flowRates.end()) ? it->second : 0.0;
        }();
        if (std::abs(q_src_room - 0.05) > 1e-12) {
            std::ostringstream oss;
            oss << "duplicate edge overwrite bug: SRC->ROOM flow=" << q_src_room
                << " expected=0.05 (24h flow must not be overwritten by schedule=0 edge)";
            throw std::runtime_error(oss.str());
        }

        (void)core::humidity::updateHumidityIfEnabled(constants,
                                                      vent,
                                                      thermal.getGraph(),
                                                      static_cast<const ThermalNetwork&>(thermal).nodeStateView(),
                                                      humidity,
                                                      flowRates,
                                                      logs,
                                                      timings,
                                                      "test");

        // ROOM should approach SRC humidity (0.010), not decay toward 0
        const double dt = static_cast<double>(constants.timestep);
        const double q = 0.05;
        const double V = 50.0;
        const double a = dt * (q / V);
        const double expected = (0.0 + a * 0.010) / (1.0 + a);

        const auto& tG = thermal.getGraph();
        const auto& tMap = thermal.getKeyToVertex();
        const auto itR = tMap.find("ROOM");
        if (itR == tMap.end()) throw std::runtime_error("missing ROOM");
        const double actual = tG[itR->second].current_x;
        expectNear(actual, expected, 1e-12, "duplicate edge flow must be summed (not overwritten)");
    }

    // ------------------------------------------------------------------
    // 5) humidity network (Phase1): moisture_conductance + moisture_capacity
    //    空気ノードと材料ノードの間で湿気交換する（換気流なし）
    // ------------------------------------------------------------------
    {
        auto ROOM = makeNode("ROOM");
        ROOM.calc_x = true;
        ROOM.current_x = 0.010;
        ROOM.v = 100.0; // 空気側容量は rho*V

        auto MAT = makeNode("MAT");
        MAT.calc_x = true;
        MAT.current_x = 0.002;
        MAT.v = 0.0;
        MAT.moisture_capacity = 50.0; // 材料側容量

        std::vector<VertexProperties> nodes = {ROOM, MAT};
        std::vector<EdgeProperties> ventEdges = {};
        std::vector<EdgeProperties> thEdges = {};
        EdgeProperties m{};
        m.key = "MAT->ROOM";
        m.unique_id = "MAT->ROOM";
        m.type = "conductance";
        m.source = "MAT";
        m.target = "ROOM";
        m.moisture_conductance = 0.002; // [kg/s]
        thEdges.push_back(m);

        VentilationNetwork vent;
        ThermalNetwork thermal;
        HumidityNetwork humidity;
        vent.buildFromData(nodes, ventEdges, constants, logs);
        thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);

        FlowRateMap emptyFlows;
        (void)core::humidity::updateHumidityIfEnabled(constants,
                                                      vent,
                                                      thermal.getGraph(),
                                                      static_cast<const ThermalNetwork&>(thermal).nodeStateView(),
                                                      humidity,
                                                      emptyFlows,
                                                      logs,
                                                      timings,
                                                      "test");

        const auto& tG = thermal.getGraph();
        const auto& tMap = thermal.getKeyToVertex();
        const auto itRoom = tMap.find("ROOM");
        const auto itMat = tMap.find("MAT");
        if (itRoom == tMap.end() || itMat == tMap.end()) {
            throw std::runtime_error("missing ROOM or MAT");
        }
        const double xRoom1 = tG[itRoom->second].current_x;
        const double xMat1 = tG[itMat->second].current_x;
        // 交換のみなので ROOM は減少、MAT は増加する
        if (!(xRoom1 < ROOM.current_x)) {
            throw std::runtime_error("moisture network: ROOM humidity must decrease");
        }
        if (!(xMat1 > MAT.current_x)) {
            throw std::runtime_error("moisture network: MAT humidity must increase");
        }
        // 質量保存（容量重み平均がほぼ保存）
        const double cRoom = PhysicalConstants::DENSITY_DRY_AIR * ROOM.v;
        const double cMat = MAT.moisture_capacity;
        const double m0 = cRoom * ROOM.current_x + cMat * MAT.current_x;
        const double m1 = cRoom * xRoom1 + cMat * xMat1;
        expectNear(m1, m0, 1e-7, "moisture network mass conservation");
    }

    std::cout << "[OK] all tests passed\n";
    return 0;
}


