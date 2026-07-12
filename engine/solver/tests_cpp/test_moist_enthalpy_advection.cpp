#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/thermal/heat_calculation.h"
#include "core/thermal/thermal_direct_internal.h"
#include "core/thermal/thermal_moist_air.h"
#include "core/thermal/thermal_solver.h"
#include "core/thermal/thermal_solver_linear_direct.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_operation_mode.h"
#include "network/thermal_network.h"
#include "parser/sim_constants_parser.h"

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

VertexProperties makeAir(const std::string& key, bool calcT, double t, double v, double x) {
    VertexProperties n{};
    n.key = key;
    n.type = "normal";
    n.calc_t = calcT;
    n.calc_x = true;
    n.current_t = t;
    n.current_x = x;
    n.v = v;
    n.heat_source = 0.0;
    return n;
}

VertexProperties makeCapacity(const std::string& key, const std::string& ref, double t) {
    VertexProperties n{};
    n.key = key;
    n.type = "capacity";
    n.calc_t = false;
    n.ref_node = ref;
    n.current_t = t;
    return n;
}

nlohmann::json minimalSimJson(bool humidityCalc) {
    return nlohmann::json{
        {"simulation",
         {{"index", {{"start", "0"}, {"end", "1"}, {"timestep", 1}, {"length", 1}}},
          {"tolerance",
           {{"ventilation", 1e-6}, {"thermal", 1e-6}, {"convergence", 1e-6}}},
          {"calc_flag",
           {{"p", false}, {"t", true}, {"x", humidityCalc}, {"c", false}}}}},
    };
}

void testHelperDiagnostics() {
    // x=0: h ≈ cp_dry * T
    expectNear(thermal_moist_air::moistAirEnthalpy(20.0, 0.0),
               archenv::SPECIFIC_HEAT_AIR * 20.0, 1e-9, "h(x=0)");
    expectNear(thermal_moist_air::moistAirCp(0.0), archenv::SPECIFIC_HEAT_AIR, 1e-12, "cp(x=0)");

    const double Q = 0.05; // m3/s
    const double Hdot =
        thermal_moist_air::advectionEnthalpyFluxW(20.0, 0.010, 20.0, 0.005, Q);
    const double Qsens = thermal_moist_air::advectionSensibleFluxW(20.0, 20.0, Q);
    expectNear(Qsens, 0.0, 1e-12, "same-T sensible=0");
    expectTrue(std::abs(Hdot) > 1.0, "same-T different-x => H_dot != 0");

    const double HdotX0 =
        thermal_moist_air::advectionEnthalpyFluxW(25.0, 0.0, 20.0, 0.0, Q);
    const double QsensDT = thermal_moist_air::advectionSensibleFluxW(25.0, 20.0, Q);
    expectNear(HdotX0, QsensDT, 1e-9, "x=0 enthalpy flux == sensible");
}

void testParserMoistEnthalpyFlag() {
    std::ostringstream logs;
    {
        auto j = minimalSimJson(true);
        j["simulation"]["coupling"] = {
            {"moist_enthalpy_enabled", true},
            {"moisture_enabled", true},
        };
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.moistEnthalpyEnabled, "parser: moist_enthalpy_enabled true");
    }
    {
        auto j = minimalSimJson(false);
        j["simulation"]["coupling"] = {{"moist_enthalpy_enabled", true}};
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("calc_flag.x") != std::string::npos,
                       "parser: moist without x message");
        }
        expectTrue(threw, "parser: moist without x throws");
    }
    {
        auto j = minimalSimJson(true);
        j["simulation"]["calc_flag"]["t"] = false;
        j["simulation"]["coupling"] = {
            {"moist_enthalpy_enabled", true},
            {"moisture_enabled", true},
        };
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("calc_flag.t") != std::string::npos,
                       "parser: moist without t message");
        }
        expectTrue(threw, "parser: moist without t throws");
    }
    {
        auto j = minimalSimJson(true);
        j["simulation"]["coupling"] = {
            {"moist_enthalpy_enabled", true},
            {"moisture_enabled", false},
        };
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("moisture_enabled") != std::string::npos,
                       "parser: moist without coupling message");
        }
        expectTrue(threw, "parser: moist without moisture coupling throws");
    }
    {
        auto j = minimalSimJson(true);
        j["simulation"]["coupling"] = {
            {"moist_enthalpy_enabled", true},
            {"latent_coupling_mode", "from_humidity_change"},
            {"moisture_enabled", true},
        };
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("from_humidity_change") != std::string::npos,
                       "parser: moist+from_humidity_change message");
        }
        expectTrue(threw, "parser: moist+from_humidity_change throws");
    }
    {
        auto j = minimalSimJson(true);
        j["simulation"]["coupling"] = {
            {"moist_enthalpy_enabled", true},
            {"latent_coupling_mode", "from_phase_change"},
            {"moisture_enabled", true},
        };
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("from_phase_change") != std::string::npos,
                       "parser: moist+from_phase_change message");
        }
        expectTrue(threw, "parser: moist+from_phase_change throws");
    }
}

SimulationConstants makeThermalConstants(bool moist) {
    SimulationConstants c{};
    c.timestep = 60;
    c.thermalTolerance = 1e-9;
    c.thermalBalanceToleranceW = 1e-6;
    c.thermalLinearResidualRelativeTolerance = 1e-10;
    c.logVerbosity = 0;
    c.temperatureCalc = true;
    c.humidityCalc = true;
    c.pressureCalc = false;
    c.moistEnthalpyEnabled = moist;
    return c;
}

// 2室換気 + 空気 capacity。moist ON/OFF 比較用。
struct TwoRoomNet {
    ThermalNetwork net;
    Vertex vA = 0;
    Vertex vB = 0;
    Vertex vAc = 0;
    Vertex vBc = 0;
};

TwoRoomNet buildTwoRoom(double tA,
                        double tB,
                        double xA,
                        double xB,
                        double flowAB,
                        double thermalMass) {
    TwoRoomNet out;
    const double dt = 60.0;
    auto A = makeAir("A", true, tA, 50.0, xA);
    auto B = makeAir("B", true, tB, 50.0, xB);
    auto Ac = makeCapacity("A_c", "A", tA);
    auto Bc = makeCapacity("B_c", "B", tB);
    out.net.addNode(A);
    out.net.addNode(B);
    out.net.addNode(Ac);
    out.net.addNode(Bc);
    const auto& map = out.net.getKeyToVertex();
    out.vA = map.at("A");
    out.vB = map.at("B");
    out.vAc = map.at("A_c");
    out.vBc = map.at("B_c");

    EdgeProperties capA{};
    capA.key = "A_c->A";
    capA.unique_id = "A_c->A";
    capA.type = "conductance";
    capA.subtype = "capacity";
    capA.source = "A_c";
    capA.target = "A";
    capA.conductance = thermalMass / dt;
    out.net.addEdge(capA);

    EdgeProperties capB{};
    capB.key = "B_c->B";
    capB.unique_id = "B_c->B";
    capB.type = "conductance";
    capB.subtype = "capacity";
    capB.source = "B_c";
    capB.target = "B";
    capB.conductance = thermalMass / dt;
    out.net.addEdge(capB);

    EdgeProperties adv{};
    adv.key = "adv";
    adv.unique_id = "adv";
    adv.type = "advection";
    adv.source = "A";
    adv.target = "B";
    adv.flow_rate = flowAB;
    out.net.addEdge(adv);

    // 戻り換気（質量保存の簡易対称）
    EdgeProperties advRet{};
    advRet.key = "adv_ret";
    advRet.unique_id = "adv_ret";
    advRet.type = "advection";
    advRet.source = "B";
    advRet.target = "A";
    advRet.flow_rate = flowAB;
    out.net.addEdge(advRet);

    return out;
}

void syncXnFromCurrent(ThermalNetwork& net) {
    auto& g = net.getGraph();
    std::vector<double> xn(boost::num_vertices(g), 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        xn[static_cast<size_t>(v)] = g[v].current_x;
    }
    net.setMoistEnthalpyHumidityXn(std::move(xn));
}

void testX0MatchesSensible() {
    ThermalSolverLinearDirect::resetDirectTSolverContext();
    const double tA = 25.0, tB = 20.0, x = 0.0, Q = 0.02;
    const double C = archenv::DENSITY_DRY_AIR * 50.0 * archenv::SPECIFIC_HEAT_AIR;

    auto off = buildTwoRoom(tA, tB, x, x, Q, C);
    auto on = buildTwoRoom(tA, tB, x, x, Q, C);
    syncXnFromCurrent(off.net);
    syncXnFromCurrent(on.net);

    std::ostringstream logs;
    auto cOff = makeThermalConstants(false);
    auto cOn = makeThermalConstants(true);
    ThermalSolver sOff(off.net, logs);
    ThermalSolver sOn(on.net, logs);
    sOff.solveTemperatures(cOff);
    sOn.solveTemperatures(cOn);

    expectNear(on.net.getGraph()[on.vA].current_t, off.net.getGraph()[off.vA].current_t, 1e-6,
               "x=0 moist A matches dry");
    expectNear(on.net.getGraph()[on.vB].current_t, off.net.getGraph()[off.vB].current_t, 1e-6,
               "x=0 moist B matches dry");
}

void testSameTDifferentX() {
    ThermalSolverLinearDirect::resetDirectTSolverContext();
    // 外気(固定) → 室: 同温度・高湿度の外気が入っても室温が下がらないこと
    const double T = 20.0;
    const double C = archenv::DENSITY_DRY_AIR * 50.0 * archenv::SPECIFIC_HEAT_AIR;
    const double dt = 60.0;

    ThermalNetwork net;
    auto OUT = makeAir("OUT", false, T, 0.0, 0.015);
    auto ROOM = makeAir("ROOM", true, T, 50.0, 0.005);
    auto Rc = makeCapacity("ROOM_c", "ROOM", T);
    net.addNode(OUT);
    net.addNode(ROOM);
    net.addNode(Rc);
    const auto& map = net.getKeyToVertex();

    EdgeProperties cap{};
    cap.key = "ROOM_c->ROOM";
    cap.unique_id = "ROOM_c->ROOM";
    cap.type = "conductance";
    cap.subtype = "capacity";
    cap.source = "ROOM_c";
    cap.target = "ROOM";
    cap.conductance = C / dt;
    net.addEdge(cap);

    EdgeProperties adv{};
    adv.key = "adv";
    adv.unique_id = "adv";
    adv.type = "advection";
    adv.source = "OUT";
    adv.target = "ROOM";
    adv.flow_rate = 0.05;
    net.addEdge(adv);

    syncXnFromCurrent(net);
    std::ostringstream logs;
    ThermalSolver solver(net, logs);
    solver.solveTemperatures(makeThermalConstants(true));

    const double tRoom = net.getGraph()[map.at("ROOM")].current_t;
    expectTrue(tRoom >= T - 1e-6, "humid outdoor same-T: room must not cool");
    expectTrue(tRoom > T + 0.01, "humid outdoor same-T: room enthalpy gain raises T");

    // OFF だと同Tで移流熱=0のまま
    ThermalSolverLinearDirect::resetDirectTSolverContext();
    ThermalNetwork netOff;
    netOff.addNode(OUT);
    netOff.addNode(ROOM);
    netOff.addNode(Rc);
    netOff.addEdge(cap);
    netOff.addEdge(adv);
    syncXnFromCurrent(netOff);
    ThermalSolver solverOff(netOff, logs);
    solverOff.solveTemperatures(makeThermalConstants(false));
    expectNear(netOff.getGraph()[netOff.getKeyToVertex().at("ROOM")].current_t, T, 1e-6,
               "OFF same-T: no temperature change");
}

void testSameXDifferentTNearSensible() {
    ThermalSolverLinearDirect::resetDirectTSolverContext();
    const double x = 0.008;
    const double C = archenv::DENSITY_DRY_AIR * 50.0 * archenv::SPECIFIC_HEAT_AIR;
    auto off = buildTwoRoom(25.0, 20.0, x, x, 0.03, C);
    auto on = buildTwoRoom(25.0, 20.0, x, x, 0.03, C);
    syncXnFromCurrent(off.net);
    syncXnFromCurrent(on.net);

    std::ostringstream logs;
    ThermalSolver sOff(off.net, logs);
    ThermalSolver sOn(on.net, logs);
    sOff.solveTemperatures(makeThermalConstants(false));
    sOn.solveTemperatures(makeThermalConstants(true));

    // 同湿度なら cp(x) 補正のみで近い（潜熱項は打ち消し）
    expectNear(on.net.getGraph()[on.vA].current_t, off.net.getGraph()[off.vA].current_t, 0.05,
               "same-x different-T A near dry");
    expectNear(on.net.getGraph()[on.vB].current_t, off.net.getGraph()[off.vB].current_t, 0.05,
               "same-x different-T B near dry");
}

void testClosedSystemMixingEnthalpy() {
    ThermalSolverLinearDirect::resetDirectTSolverContext();
    // 閉鎖系対称換気・同T異x: 湿り側はエンタルピーを輸出し温度が下がるが、
    // 乾き側は上がり、2室平均温度はほぼ初期のまま（系エネルギー保存の近似）
    const double T = 22.0;
    const double C = 5.0e5;
    auto net = buildTwoRoom(T, T, 0.015, 0.005, 0.1, C);
    syncXnFromCurrent(net.net);
    std::ostringstream logs;
    ThermalSolver solver(net.net, logs);
    solver.solveTemperatures(makeThermalConstants(true));
    const double tA = net.net.getGraph()[net.vA].current_t;
    const double tB = net.net.getGraph()[net.vB].current_t;
    expectNear(0.5 * (tA + tB), T, 0.15, "closed mix: mean T conserved");
    expectTrue(tA < T && tB > T, "closed mix: wet cools, dry warms");
}

void testFlagOffRegression() {
    ThermalSolverLinearDirect::resetDirectTSolverContext();
    const double C = archenv::DENSITY_DRY_AIR * 50.0 * archenv::SPECIFIC_HEAT_AIR;
    auto a = buildTwoRoom(24.0, 18.0, 0.01, 0.006, 0.04, C);
    auto b = buildTwoRoom(24.0, 18.0, 0.01, 0.006, 0.04, C);
    // moist flag off: humidity must not affect
    std::ostringstream logs;
    ThermalSolver sA(a.net, logs);
    ThermalSolver sB(b.net, logs);
    sA.solveTemperatures(makeThermalConstants(false));
    // change x but keep moist off
    b.net.getGraph()[b.vA].current_x = 0.020;
    b.net.getGraph()[b.vB].current_x = 0.001;
    sB.solveTemperatures(makeThermalConstants(false));
    expectNear(a.net.getGraph()[a.vA].current_t, b.net.getGraph()[b.vA].current_t, 1e-9,
               "OFF: x change does not affect T");
    expectNear(a.net.getGraph()[a.vB].current_t, b.net.getGraph()[b.vB].current_t, 1e-9,
               "OFF: x change does not affect T B");
}

void testAirconProcessedEnthalpyHelper() {
    const double Q = 0.1;
    const double tIn = 27.0, tOut = 14.0;
    const double xIn = 0.012, xOut = 0.008;
    const double qCool = thermal_moist_air::processedEnthalpyHeatW(tIn, xIn, tOut, xOut, Q, false);
    const double qHeatWrong =
        thermal_moist_air::processedEnthalpyHeatW(tIn, xIn, tOut, xOut, Q, true);
    expectTrue(qCool > 0.0, "cooling enthalpy heat > 0");
    expectNear(qHeatWrong, 0.0, 1e-12, "heating with cooling delta => 0");

    // x=0: equals dry sensible
    const double qSens =
        archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * Q * (tIn - tOut);
    const double qX0 =
        thermal_moist_air::processedEnthalpyHeatW(tIn, 0.0, tOut, 0.0, Q, false);
    expectNear(qX0, qSens, 1e-6, "aircon x=0 enthalpy == dry sensible");

    // same T, dehumidify: still positive cooling load
    const double qLatOnly =
        thermal_moist_air::processedEnthalpyHeatW(20.0, 0.012, 20.0, 0.008, Q, false);
    expectTrue(qLatOnly > 100.0, "same-T dehumidify => positive enthalpy heat");
}

void testAirconLatentProcessMoistTotal() {
    AirconValidationData vd{};
    vd.indoorTemp = 27.0;
    vd.airconTemp = 14.0;
    vd.indoorX = 0.012;
    vd.outdoorTemp = 35.0;
    vd.outdoorX = 0.020;
    vd.setTemp = 26.0;

    VertexProperties node{};
    node.key = "AC";
    node.type = "aircon";
    node.ac_spec = nlohmann::json::object();
    node.ac_spec["latent_method"] = "rh95";

    const double flow = 0.05;
    const double qsDry =
        archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flow * (27.0 - 14.0);
    const auto off = aircon::latent::estimateLatentProcess(
        vd, OperationMode::Cooling, qsDry, flow, node, /*moistEnthalpyEnabled=*/false);
    const auto on = aircon::latent::estimateLatentProcess(
        vd, OperationMode::Cooling, qsDry, flow, node, /*moistEnthalpyEnabled=*/true);

    const double qEnth = thermal_moist_air::processedEnthalpyHeatW(
        vd.indoorTemp, vd.indoorX, vd.airconTemp, on.supplyX, flow, /*heating=*/false);
    expectNear(aircon::latent::totalHeatCapacity(on), qEnth, 1e-3, "moist ON: total == mDot*|Δh|");
    expectTrue(aircon::latent::totalHeatCapacity(on) + 1e-6 >=
                   aircon::latent::totalHeatCapacity(off),
               "moist ON total >= OFF");
    expectNear(on.sensibleHeatCapacity + on.latentHeatCapacity,
               aircon::latent::totalHeatCapacity(on), 1e-9, "Qs+Ql == total");
    expectNear(on.condensationRateKgPerS,
               archenv::DENSITY_DRY_AIR * flow * std::max(0.0, vd.indoorX - on.supplyX), 1e-12,
               "condensation = mDot*(xIn-xSupply)");
}

} // namespace

int main() {
    try {
        testHelperDiagnostics();
        testParserMoistEnthalpyFlag();
        testX0MatchesSensible();
        testSameTDifferentX();
        testSameXDifferentTNearSensible();
        testClosedSystemMixingEnthalpy();
        testFlagOffRegression();
        testAirconProcessedEnthalpyHelper();
        testAirconLatentProcessMoistTotal();
        std::cout << "OK moist enthalpy advection/storage\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
