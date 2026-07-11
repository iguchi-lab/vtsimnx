#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/range/iterator_range.hpp>

#include "core/thermal/thermal_solver.h"
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

} // namespace

int main() {
    try {
        testRejectedPressureNotAppliedAndNoThermal();
        testFixedPressureOnlyAccepted();
        testThermalNoActiveNodeState();
        testAcceptedPressureAppliedInCoupledStep();
        std::cout << "[OK] all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
