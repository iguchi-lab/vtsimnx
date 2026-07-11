#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/humidity/humidity_solver.h"
#include "core/thermal/thermal_solver.h"
#include "core/ventilation/pressure_solver.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"

namespace {

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (!std::isfinite(actual)) {
        throw std::runtime_error(msg + " (actual is not finite)");
    }
    const double diff = std::abs(actual - expected);
    if (diff > tol) {
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
    v.current_p = 0.0;
    v.current_t = 20.0;
    v.current_x = 0.0;
    v.current_c = 0.0;
    v.current_beta = 0.0;
    v.v = 100.0;
    v.calc_p = false;
    v.calc_t = false;
    v.calc_x = false;
    v.calc_c = false;
    v.heat_source = 0.0;
    return v;
}

EdgeProperties makeOpening(const std::string& key,
                           const std::string& s,
                           const std::string& t,
                           double alpha,
                           double area) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "simple_opening";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.alpha = alpha;
    e.area = area;
    e.h_from = 0.0;
    e.h_to = 0.0;
    return e;
}

EdgeProperties makeConductance(const std::string& key,
                               const std::string& s,
                               const std::string& t,
                               double k) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "conductance";
    e.source = s;
    e.target = t;
    e.conductance = k;
    e.current_enabled = true;
    return e;
}

EdgeProperties makeFixedFlow(const std::string& key,
                             const std::string& s,
                             const std::string& t,
                             double q) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "fixed_flow";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.current_vol = q;
    e.vol = {q};
    e.eta = 0.0;
    return e;
}

SimulationConstants makeConstants() {
    SimulationConstants c{};
    c.timestep = 3600;
    c.length = 1;
    // 体積流量収支の物理合否。Ceres 停止条件とは分離（ceresFunctionRelative 等）。
    c.ventilationTolerance = 1e-8;
    c.thermalTolerance = 1e-10;
    c.convergenceTolerance = 1e-10;
    c.maxInnerIterations = 50;
    c.pressureCalc = true;
    c.temperatureCalc = true;
    c.humidityCalc = true;
    c.concentrationCalc = false;
    c.logVerbosity = 0;
    return c;
}

} // namespace

int main() {
    try {
        const auto constants = makeConstants();
        std::ostringstream logs;

        // ------------------------------------------------------------------
        // 1) core/ventilation: 1室2開口の既知解
        //    対称開口・等温・同一高さなら、室圧は境界圧の中点になる。
        // ------------------------------------------------------------------
        {
            VentilationNetwork vent;

            auto outH = makeNode("OUT_H");
            outH.calc_p = false;
            outH.current_p = 100.0;
            auto room = makeNode("ROOM");
            room.calc_p = true;
            room.current_p = 0.0;
            auto outL = makeNode("OUT_L");
            outL.calc_p = false;
            outL.current_p = 0.0;

            vent.addNode(outH);
            vent.addNode(room);
            vent.addNode(outL);
            vent.addEdge(makeOpening("OUT_H->ROOM", "OUT_H", "ROOM", 0.65, 1.0));
            vent.addEdge(makeOpening("ROOM->OUT_L", "ROOM", "OUT_L", 0.65, 1.0));

            PressureSolver solver(vent, logs);
            auto [pressureMap, flowRates, balance] = solver.solvePressures(constants);

            const auto itRoomP = pressureMap.find("ROOM");
            expectTrue(itRoomP != pressureMap.end(), "ventilation known solution: ROOM pressure exists");
            expectNear(itRoomP->second, 50.0, 1e-4,
                       "ventilation known solution: room pressure should be midpoint");

            const auto itQIn = flowRates.find({"OUT_H", "ROOM"});
            const auto itQOut = flowRates.find({"ROOM", "OUT_L"});
            expectTrue(itQIn != flowRates.end(), "ventilation known solution: inflow exists");
            expectTrue(itQOut != flowRates.end(), "ventilation known solution: outflow exists");
            expectNear(itQIn->second, itQOut->second, 1e-6,
                       "ventilation known solution: inflow and outflow should match");
            expectNear(balance.at("ROOM"), 0.0, 1e-6,
                       "ventilation known solution: room volume-flow balance should be zero");

            // disabled 開口は solver 評価・apply 後とも流量ゼロで、既知解を壊さない
            auto disabled = makeOpening("ROOM->OUT_L(disabled)", "ROOM", "OUT_L", 0.65, 1.0);
            disabled.current_enabled = false;
            vent.addEdge(disabled);
            PressureSolver solver2(vent, logs);
            auto result2 = solver2.solveDetailed(constants);
            expectTrue(result2.accepted, "disabled extra opening still accepted");
            expectNear(result2.pressures.at("ROOM"), 50.0, 1e-4, "disabled opening: room pressure unchanged");
            expectTrue(result2.flows.count({"ROOM", "OUT_L"}) > 0, "disabled opening: outflow key exists");
            vent.applySolveResults(result2.pressures, result2.flows);
            bool sawDisabled = false;
            for (auto e : boost::make_iterator_range(boost::edges(vent.getGraph()))) {
                if (vent.getGraph()[e].key == "ROOM->OUT_L(disabled)") {
                    sawDisabled = true;
                    expectNear(vent.getGraph()[e].flow_rate, 0.0, 0.0,
                               "disabled opening: applySolveResults flow_rate == 0");
                }
            }
            expectTrue(sawDisabled, "disabled opening edge present");
        }

        // ------------------------------------------------------------------
        // 1b) 固定圧境界のない閉成分でも soft anchor で安定に解ける
        // ------------------------------------------------------------------
        {
            VentilationNetwork vent;
            auto a = makeNode("A");
            a.calc_p = true;
            a.current_p = 12.0;
            auto b = makeNode("B");
            b.calc_p = true;
            b.current_p = -3.0;
            vent.addNode(a);
            vent.addNode(b);
            vent.addEdge(makeOpening("A->B", "A", "B", 0.65, 1.0));

            PressureSolver solver(vent, logs);
            auto result = solver.solveDetailed(constants);
            expectTrue(result.accepted, "closed component: accepted with soft anchor");
            expectNear(result.pressures.at("A"), result.pressures.at("B"), 1e-4,
                       "closed component: pressures equalize");
            // ゲージ固定により絶対圧は初期値（A）付近に寄る
            expectNear(result.pressures.at("A"), 12.0, 1e-3, "closed component: gauge pinned near A init");
            expectNear(result.balances.at("A"), 0.0, 1e-6, "closed component: A balance");
            expectNear(result.balances.at("B"), 0.0, 1e-6, "closed component: B balance");
        }

        // ------------------------------------------------------------------
        // 1c) 独立な閉成分が複数あっても各成分に gauge を立てて解ける
        // ------------------------------------------------------------------
        {
            VentilationNetwork vent;
            auto a1 = makeNode("A1"); a1.calc_p = true; a1.current_p = 5.0;
            auto b1 = makeNode("B1"); b1.calc_p = true; b1.current_p = 0.0;
            auto a2 = makeNode("A2"); a2.calc_p = true; a2.current_p = -8.0;
            auto b2 = makeNode("B2"); b2.calc_p = true; b2.current_p = 2.0;
            vent.addNode(a1); vent.addNode(b1); vent.addNode(a2); vent.addNode(b2);
            vent.addEdge(makeOpening("A1->B1", "A1", "B1", 0.65, 1.0));
            vent.addEdge(makeOpening("A2->B2", "A2", "B2", 0.65, 1.0));

            PressureSolver solver(vent, logs);
            auto result = solver.solveDetailed(constants);
            expectTrue(result.accepted, "multi closed components accepted");
            expectTrue(result.metrics.maxAbs <= constants.ventilationTolerance,
                       "multi closed components: maxAbs within ventilationTolerance");
            expectNear(result.pressures.at("A1"), result.pressures.at("B1"), 1e-4, "comp1 equalize");
            expectNear(result.pressures.at("A2"), result.pressures.at("B2"), 1e-4, "comp2 equalize");
            expectNear(result.pressures.at("A1"), 5.0, 1e-3, "comp1 gauge near A1 init");
            expectNear(result.pressures.at("A2"), -8.0, 1e-3, "comp2 gauge near A2 init");
        }

        // ------------------------------------------------------------------
        // 2) core/thermal: 1室2境界の定常伝熱既知解
        //    k1*(T1-T)+k2*(T2-T)=0 -> T=(k1*T1+k2*T2)/(k1+k2)
        // ------------------------------------------------------------------
        {
            ThermalNetwork thermal;

            auto hot = makeNode("HOT");
            hot.calc_t = false;
            hot.current_t = 30.0;
            auto cold = makeNode("COLD");
            cold.calc_t = false;
            cold.current_t = 10.0;
            auto room = makeNode("ROOM");
            room.calc_t = true;
            room.current_t = 0.0;

            thermal.addNode(hot);
            thermal.addNode(cold);
            thermal.addNode(room);
            thermal.addEdge(makeConductance("HOT->ROOM", "HOT", "ROOM", 2.0));
            thermal.addEdge(makeConductance("COLD->ROOM", "COLD", "ROOM", 1.0));

            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);

            const auto& g = thermal.getGraph();
            const auto& kv = thermal.getKeyToVertex();
            const double actual = g[kv.at("ROOM")].current_t;
            const double expected = (2.0 * 30.0 + 1.0 * 10.0) / 3.0;
            expectNear(actual, expected, 1e-8,
                       "thermal known solution: weighted-average temperature");
        }

        // ------------------------------------------------------------------
        // 3) core/humidity: 1ステップ implicit 既知解
        //    dx/dt = (q/V)*(x_src-x) -> x_{n+1}=(x_n+a*x_src)/(1+a), a=dt*q/V
        // ------------------------------------------------------------------
        {
            auto src = makeNode("SRC");
            src.v = 0.0;
            src.calc_x = false;
            src.current_x = 0.010;

            auto room = makeNode("ROOM");
            room.v = 100.0;
            room.calc_x = true;
            room.current_x = 0.005;

            auto ext = makeNode("EXT");
            ext.v = 0.0;
            ext.calc_x = false;
            ext.current_x = 0.002;

            std::vector<VertexProperties> nodes = {src, room, ext};
            std::vector<EdgeProperties> ventEdges = {
                makeFixedFlow("SRC->ROOM", "SRC", "ROOM", 0.1),
                makeFixedFlow("ROOM->EXT", "ROOM", "EXT", 0.1),
            };
            std::vector<EdgeProperties> thEdges = {};

            VentilationNetwork vent;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            TimingList timings;
            vent.buildFromData(nodes, ventEdges, constants, logs);
            thermal.buildFromData(nodes, thEdges, ventEdges, constants, logs);
            vent.updatePropertiesForTimestep(nodes, ventEdges, 0);

            FlowRateMap flowRates = vent.collectFlowRateMap();
            (void)core::humidity::updateHumidityIfEnabled(constants,
                                                          vent,
                                                          thermal.getGraph(),
                                                          static_cast<const ThermalNetwork&>(thermal).nodeStateView(),
                                                          humidity,
                                                          flowRates,
                                                          logs,
                                                          timings,
                                                          "known-solution");

            const double dt = static_cast<double>(constants.timestep);
            const double q = 0.1;
            const double V = 100.0;
            const double a = dt * (q / V);
            const double expected = (0.005 + a * 0.010) / (1.0 + a);
            const auto& g = thermal.getGraph();
            const auto& kv = thermal.getKeyToVertex();
            const double actual = g[kv.at("ROOM")].current_x;
            expectNear(actual, expected, 1e-12,
                       "humidity known solution: implicit one-step update");
        }

        std::cout << "[OK] all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}

