#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/range/iterator_range.hpp>

#include "core/thermal/thermal_solver.h"
#include "core/thermal/thermal_solver_linear_direct.h"
#include "core/ventilation/pressure_solver.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_coupled_step.h"
#include "simulation_error.h"
#include "types/common_types.h"
#include "vtsimnx_solver_timing.h"

namespace {

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

VertexProperties makeVentNode(const std::string& key, bool calcP, double p0) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.calc_p = calcP;
    v.current_p = p0;
    v.current_t = 20.0;
    v.v = 100.0;
    return v;
}

VertexProperties makeThermalNode(const std::string& key, bool calcT, double t0) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.calc_t = calcT;
    v.current_t = t0;
    v.v = 100.0;
    return v;
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
    e.flow_rate = q;
    e.current_vol = q;
    e.vol = {q};
    return e;
}

EdgeProperties makeOpening(const std::string& key,
                           const std::string& s,
                           const std::string& t) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "simple_opening";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.alpha = 0.65;
    e.area = 1.0;
    e.h_from = 0.0;
    e.h_to = 0.0;
    return e;
}

SimulationConstants baseConstants() {
    SimulationConstants c{};
    c.logVerbosity = 0;
    c.pressureCalc = true;
    c.temperatureCalc = false;
    c.humidityCalc = false;
    c.ventilationTolerance = 1e-8;
    c.timestep = 3600;
    return c;
}

// 不採用圧力を apply せず PressureNotConverged で止まり、熱も汚染しない
void testRejectedPressureNotAppliedAndNoThermal() {
    std::ostringstream logs;
    TimingList timings;
    auto constants = baseConstants();
    constants.temperatureCalc = true;
    constants.pressureCalc = true;

    // ROOM へ流入のみ・流出なし → 質量収支不能
    auto room = makeVentNode("ROOM", true, 0.0);
    auto ext = makeVentNode("EXT", false, 10.0);
    std::vector<VertexProperties> nodes = {room, ext};
    std::vector<EdgeProperties> vent = {makeFixedFlow("EXT->ROOM", "EXT", "ROOM", 0.1)};

    VentilationNetwork ventNet;
    ThermalNetwork thermal;
    ventNet.buildFromData(nodes, vent, constants, logs);

    auto roomT = makeThermalNode("ROOM", true, 25.0);
    auto extT = makeThermalNode("EXT", false, 0.0);
    std::vector<VertexProperties> tNodes = {roomT, extT};
    std::vector<EdgeProperties> th;
    thermal.buildFromData(tNodes, th, vent, constants, logs);

    const double pRoomBefore = ventNet.getGraph()[ventNet.getKeyToVertex().at("ROOM")].current_p;
    const double tRoomBefore = thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_t;

    bool threw = false;
    try {
        (void)performCoupledStepCalculation(ventNet, thermal, constants, logs, timings, "rej");
    } catch (const simulation::Error& e) {
        threw = true;
        expectTrue(e.code() == simulation::ErrorCode::PressureNotConverged,
                   "coupled: PressureNotConverged");
        expectTrue(std::string(simulation::toErrorCodeString(e.code())) == "pressure_not_converged",
                   "coupled: api string");
    }
    expectTrue(threw, "coupled must throw on rejected pressure");
    expectNear(ventNet.getGraph()[ventNet.getKeyToVertex().at("ROOM")].current_p,
               pRoomBefore, 0.0, "pressure graph unchanged");
    expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_t,
               tRoomBefore, 0.0, "thermal not polluted");
    expectTrue(!ventNet.getLastPressureConverged(), "lastPressureConverged false");
}

// 全固定圧: 圧力求解スキップで流量評価し accepted
void testFixedPressureOnlyAccepted() {
    std::ostringstream logs;
    auto constants = baseConstants();
    auto a = makeVentNode("A", false, 20.0);
    auto b = makeVentNode("B", false, 0.0);
    std::vector<VertexProperties> nodes = {a, b};
    std::vector<EdgeProperties> vent = {makeFixedFlow("A->B", "A", "B", 0.25)};

    VentilationNetwork ventNet;
    ventNet.buildFromData(nodes, vent, constants, logs);
    for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
        ventNet.getGraph()[e].current_vol = 0.25;
        ventNet.getGraph()[e].type = "fixed_flow";
        ventNet.getGraph()[e].current_enabled = true;
    }

    PressureSolver solver(ventNet, logs);
    auto result = solver.solveDetailed(constants);
    expectTrue(result.accepted, "fixed-pressure: accepted");
    expectTrue(result.method == "fixed_pressure", "fixed-pressure: method");
    expectTrue(ventNet.getLastPressureConverged(), "fixed-pressure: flag");
    expectNear(result.flows.at({"A", "B"}), 0.25, 1e-12, "fixed-pressure: flow");
}

// 空の熱問題: 収束状態を明示設定
void testThermalNoActiveNodeState() {
    std::ostringstream logs;
    auto constants = baseConstants();
    constants.pressureCalc = false;
    constants.temperatureCalc = true;

    auto a = makeThermalNode("A", false, 18.0);
    auto b = makeThermalNode("B", false, 20.0);
    std::vector<VertexProperties> nodes = {a, b};
    std::vector<EdgeProperties> th;
    std::vector<EdgeProperties> vent;

    ThermalNetwork thermal;
    thermal.buildFromData(nodes, th, vent, constants, logs);
    // 汚染: 以前の失敗状態を擬似的に残す
    thermal.setLastThermalConvergence(false, 9.0, 9.0, "stale");
    thermal.solveTemperature(constants, logs);
    expectTrue(thermal.getLastThermalConverged(), "no-active: converged");
    expectNear(thermal.getLastThermalRmseBalance(), 0.0, 0.0, "no-active: rmse");
    expectTrue(thermal.getLastThermalMethod() == "DirectT(no-active-node)", "no-active: method");
}

// 通常の開口既知解が accepted のまま連成適用できる
void testAcceptedPressureAppliedInCoupledStep() {
    std::ostringstream logs;
    TimingList timings;
    auto constants = baseConstants();
    constants.temperatureCalc = false;

    auto outH = makeVentNode("OUT_H", false, 100.0);
    auto room = makeVentNode("ROOM", true, 0.0);
    auto outL = makeVentNode("OUT_L", false, 0.0);
    std::vector<VertexProperties> nodes = {outH, room, outL};
    std::vector<EdgeProperties> vent = {
        makeOpening("OUT_H->ROOM", "OUT_H", "ROOM"),
        makeOpening("ROOM->OUT_L", "ROOM", "OUT_L"),
    };

    VentilationNetwork ventNet;
    ThermalNetwork thermal;
    ventNet.buildFromData(nodes, vent, constants, logs);
    thermal.buildFromData(nodes, {}, vent, constants, logs);

    auto step = performCoupledStepCalculation(ventNet, thermal, constants, logs, timings, "ok");
    expectNear(step.pressureMap.at("ROOM"), 50.0, 1e-3, "coupled apply: room P");
    expectNear(ventNet.getGraph()[ventNet.getKeyToVertex().at("ROOM")].current_p,
               50.0, 1e-3, "graph updated");
    expectTrue(ventNet.getLastPressureConverged(), "accepted flag");
}

void testDeadEndRoundoffAndSmallPrescribedFlow() {
    for (bool prescribed : {false, true}) {
        std::ostringstream logs;
        TimingList timings;
        auto constants = baseConstants();
        auto ext = makeVentNode("EXT", false, 0.0);
        auto end = makeVentNode("END", !prescribed, 1e-5);
        auto edge = prescribed ? makeFixedFlow("EXT->END", "EXT", "END", 1e-13)
                               : makeOpening("EXT->END", "EXT", "END");
        // Pressure residual is already below its absolute tolerance, while a
        // massless mixing node still requires an exactly conservative row.
        if (!prescribed) edge.area = 1e-8;
        VentilationNetwork vent;
        ThermalNetwork thermal;
        vent.buildFromData({ext, end}, {edge}, constants, logs);
        thermal.buildFromData({ext, end}, {}, {edge}, constants, logs);
        auto step = performCoupledStepCalculation(vent, thermal, constants, logs, timings, "zero-flow");
        const double expected = prescribed ? 1e-13 : 0.0;
        expectNear(step.flowRates.at({"EXT", "END"}), expected, 0.0, "consistent transport flow");
        for (auto e : boost::make_iterator_range(boost::edges(vent.getGraph())))
            expectNear(vent.getGraph()[e].flow_rate, expected, 0.0, "authoritative branch flow");
    }
}

void testDeadEndTreeHydrostaticsAndFanRestart() {
    std::ostringstream logs;
    TimingList timings;
    auto constants = baseConstants();
    auto high = makeVentNode("HIGH", false, 100.0);
    auto low = makeVentNode("LOW", false, 0.0);
    auto room = makeVentNode("ROOM", true, -1000.0);
    auto duct = makeVentNode("DUCT", true, 3000.0);
    auto end = makeVentNode("END", true, -5000.0);
    duct.current_t = 10.0;
    end.current_t = 35.0;
    auto first = makeOpening("ROOM->DUCT", "ROOM", "DUCT");
    first.h_from = 1.0;
    first.h_to = 2.0;
    auto last = makeOpening("END->DUCT", "END", "DUCT");
    last.h_from = 3.0;
    last.h_to = 4.0;
    auto fan = makeFixedFlow("LOW->END", "LOW", "END", 0.0);
    fan.type = "fan"; // Prescribed flow takes precedence over a zero PQ curve.
    fan.q_max = fan.q1 = 0.0;
    std::vector<VertexProperties> nodes = {high, low, room, duct, end};
    std::vector<EdgeProperties> edges = {
        makeOpening("HIGH->ROOM", "HIGH", "ROOM"),
        makeOpening("ROOM->LOW", "ROOM", "LOW"), first, last,
        fan,
    };
    VentilationNetwork vent;
    ThermalNetwork thermal;
    vent.buildFromData(nodes, edges, constants, logs);
    auto step = performCoupledStepCalculation(vent, thermal, constants, logs, timings, "tree");
    expectNear(step.pressureMap.at("ROOM"), 50.0, 1e-6, "remaining network solution");
    auto& graph = vent.getGraph();
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        if (graph[e].key != first.key && graph[e].key != last.key) continue;
        const auto& s = graph[boost::source(e, graph)];
        const auto& t = graph[boost::target(e, graph)];
        const double ps = s.current_p - calculateDensity(s.current_t) * archenv::GRAVITY * graph[e].h_from;
        const double pt = t.current_p - calculateDensity(t.current_t) * archenv::GRAVITY * graph[e].h_to;
        expectNear(ps, pt, 1e-12, "recovered hydrostatic pressure");
        expectNear(graph[e].flow_rate, 0.0, 0.0, "exact structural zero");
        expectTrue(graph[e].current_enabled && s.calc_p && t.calc_p, "topology restored");
    }
    // The next solve must see a newly active prescribed fan and retain both
    // passive branches. No persistent zero-flow mask may suppress circulation.
    for (auto e : boost::make_iterator_range(boost::edges(graph)))
        if (graph[e].key == "LOW->END") graph[e].current_vol = 0.02;
    step = performCoupledStepCalculation(vent, thermal, constants, logs, timings, "restart");
    expectNear(step.flowRates.at({"END", "DUCT"}), 0.02, constants.ventilationTolerance, "fan restart end");
    expectNear(step.flowRates.at({"ROOM", "DUCT"}), -0.02, constants.ventilationTolerance, "fan restart duct");
}

void testFailedReducedSolveRestoresTopologyAndState() {
    std::ostringstream logs;
    TimingList timings;
    auto constants = baseConstants();
    std::vector<VertexProperties> nodes = {
        makeVentNode("EXT", false, 0.0), makeVentNode("ROOM", true, 1.0),
        makeVentNode("END", true, 123.0),
    };
    std::vector<EdgeProperties> edges = {
        makeFixedFlow("EXT->ROOM", "EXT", "ROOM", 0.1),
        makeOpening("ROOM->END", "ROOM", "END"),
    };
    VentilationNetwork vent;
    ThermalNetwork thermal;
    vent.buildFromData(nodes, edges, constants, logs);
    bool threw = false;
    try {
        (void)performCoupledStepCalculation(vent, thermal, constants, logs, timings, "rejected-tree");
    } catch (const simulation::Error& e) {
        threw = e.code() == simulation::ErrorCode::PressureNotConverged;
    }
    expectTrue(threw, "dead end cannot dispose of a prescribed inflow");
    expectTrue(!vent.getLastPressureConverged(), "rejected solve flag");
    expectNear(vent.getNode("ROOM").current_p, 1.0, 0.0, "rejected room pressure unchanged");
    expectNear(vent.getNode("END").current_p, 123.0, 0.0, "rejected leaf pressure unchanged");
    expectTrue(vent.getNode("END").calc_p, "rejected leaf flag restored");
    for (auto e : boost::make_iterator_range(boost::edges(vent.getGraph())))
        expectTrue(vent.getGraph()[e].current_enabled, "rejected edge flag restored");
}

// 2つの ThermalNetwork で DirectT キャッシュ統計が独立
void testDirectTContextsIndependentAcrossNetworks() {
    std::ostringstream logs;
    auto constants = baseConstants();
    constants.pressureCalc = false;
    constants.temperatureCalc = true;

    auto makeNet = [&](ThermalNetwork& thermal, double wallT) {
        auto wall = makeThermalNode("WALL", false, wallT);
        auto room = makeThermalNode("ROOM", true, 0.0);
        std::vector<VertexProperties> nodes = {wall, room};
        EdgeProperties cond{};
        cond.key = "WALL->ROOM";
        cond.unique_id = "WALL->ROOM";
        cond.type = "conductance";
        cond.source = "WALL";
        cond.target = "ROOM";
        cond.conductance = 2.0;
        cond.current_enabled = true;
        thermal.buildFromData(nodes, {cond}, {}, constants, logs);
    };

    ThermalNetwork netA;
    ThermalNetwork netB;
    makeNet(netA, 20.0);
    makeNet(netB, 10.0);

    ThermalSolver solverA(netA, logs);
    ThermalSolver solverB(netB, logs);

    solverA.solveTemperatures(constants);
    solverA.solveTemperatures(constants);
    solverB.solveTemperatures(constants);

    const auto statsA = ThermalSolverLinearDirect::getDirectTCacheStats(netA.directTContext());
    const auto statsB = ThermalSolverLinearDirect::getDirectTCacheStats(netB.directTContext());
    expectTrue(statsA.calls == 2, "netA calls == 2");
    expectTrue(statsB.calls == 1, "netB calls == 1");
    expectTrue(statsA.calls != statsB.calls, "contexts not shared");

    // A をさらに回しても B は増えない
    solverA.solveTemperatures(constants);
    const auto statsA2 = ThermalSolverLinearDirect::getDirectTCacheStats(netA.directTContext());
    const auto statsB2 = ThermalSolverLinearDirect::getDirectTCacheStats(netB.directTContext());
    expectTrue(statsA2.calls == 3, "netA calls == 3 after extra solve");
    expectTrue(statsB2.calls == 1, "netB unchanged");
}

// 不採用圧力→フォールバック時、Ceres CONVERGENCE を UNKNOWN と出さない
void testFallbackLogsConvergenceNotUnknown() {
    std::ostringstream logs;
    TimingList timings;
    auto constants = baseConstants();
    constants.logVerbosity = 1;
    constants.temperatureCalc = false;

    auto room = makeVentNode("ROOM", true, 0.0);
    auto ext = makeVentNode("EXT", false, 10.0);
    std::vector<VertexProperties> nodes = {room, ext};
    std::vector<EdgeProperties> vent = {makeFixedFlow("EXT->ROOM", "EXT", "ROOM", 0.1)};

    VentilationNetwork ventNet;
    ThermalNetwork thermal;
    ventNet.buildFromData(nodes, vent, constants, logs);
    thermal.buildFromData(nodes, {}, vent, constants, logs);

    try {
        (void)performCoupledStepCalculation(ventNet, thermal, constants, logs, timings, "log");
    } catch (const simulation::Error&) {
        // expected
    }
    const std::string text = logs.str();
    expectTrue(text.find("UNKNOWN (0)") == std::string::npos,
               "fallback log must not say UNKNOWN (0) for CONVERGENCE");
    // CONVERGENCE または NO_CONVERGENCE のどちらかが明示されていればよい
    expectTrue(text.find("CONVERGENCE") != std::string::npos ||
                   text.find("NO_CONVERGENCE") != std::string::npos,
               "fallback log names a known Ceres termination");
}

} // namespace

int main() {
    try {
        testRejectedPressureNotAppliedAndNoThermal();
        testFixedPressureOnlyAccepted();
        testThermalNoActiveNodeState();
        testAcceptedPressureAppliedInCoupledStep();
        testDeadEndRoundoffAndSmallPrescribedFlow();
        testDeadEndTreeHydrostaticsAndFanRestart();
        testFailedReducedSolveRestoresTopologyAndState();
        testDirectTContextsIndependentAcrossNetworks();
        testFallbackLogsConvergenceNotUnknown();
        std::cout << "[OK] all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
