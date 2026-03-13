#include <iostream>
#include <sstream>
#include <string>

#include "core/thermal/thermal_solver_linear_direct.h"
#include "network/thermal_network.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectEqU64(std::uint64_t actual, std::uint64_t expected, const std::string& msg) {
    if (actual != expected) {
        fail(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
    }
}

EdgeProperties* findEdgeByUniqueId(Graph& g, const std::string& uid) {
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        if (g[e].unique_id == uid) return &g[e];
    }
    return nullptr;
}

} // namespace

int main() {
    // ------------------------------------------------------------
    // DirectT: 係数シグネチャが同一なら rhsOnlyBuild/solveCached に入ること
    // ------------------------------------------------------------
    ThermalSolverLinearDirect::resetDirectTCacheStats();

    ThermalNetwork net;

    // unknown (calc_t): 1ノードだけにして、常に可解な（固定温度）行にする
    VertexProperties n{};
    n.key = "N";
    n.type = "normal";
    n.calc_t = true;
    n.current_t = 25.0;
    net.addNode(n);

    // known node (calc_t=false): advection の相手
    VertexProperties k{};
    k.key = "K";
    k.type = "normal";
    k.calc_t = false;
    k.current_t = 20.0;
    net.addNode(k);

    VertexProperties ac{};
    ac.key = "AC";
    ac.type = "aircon";
    ac.calc_t = false;       // aircon 自体は未知数に含めない（N 行を固定温度化するための装置）
    ac.set_node = "N";       // N 行を fixed row にする
    ac.on = true;
    ac.current_pre_temp = 22.0;
    ac.current_t = 20.0;
    net.addNode(ac);

    // aircon を OFF にしても可解になるよう、N-K 間に conductance を入れておく
    EdgeProperties cond{};
    cond.key = "cond";
    cond.unique_id = "cond";
    cond.type = "conductance";
    cond.subtype = "conduction";
    cond.source = "N";
    cond.target = "K";
    cond.conductance = 1.0;
    net.addEdge(cond);

    EdgeProperties adv{};
    adv.key = "adv";
    adv.unique_id = "adv";
    adv.type = "advection";
    adv.source = "N";
    adv.target = "K";
    adv.flow_rate = 0.0; // coeffSig=0 扱い
    net.addEdge(adv);

    SimulationConstants constants{};
    constants.timestep = 1;
    constants.length = 1;
    constants.ventilationTolerance = 1e-9;
    constants.thermalTolerance = 1e-6;
    constants.convergenceTolerance = 1e-9;
    constants.maxInnerIteration = 50;
    constants.pressureCalc = false;
    constants.temperatureCalc = true;
    constants.logVerbosity = 0;

    std::ostringstream log;

    const auto s0 = ThermalSolverLinearDirect::getDirectTCacheStats();
    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, constants, log);
    } catch (const std::exception& e) {
        fail(std::string("DirectT: first solve threw exception: ") + e.what());
        std::cerr << "[LOG]\n" << log.str() << "\n";
        return 1;
    }
    const auto s1 = ThermalSolverLinearDirect::getDirectTCacheStats();

    expectEqU64(s1.calls, s0.calls + 1, "DirectT: calls increments");
    expectEqU64(s1.fullBuild, s0.fullBuild + 1, "DirectT: first call uses fullBuild");
    expectEqU64(s1.solveFull, s0.solveFull + 1, "DirectT: first call uses solveFull");

    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, constants, log);
    } catch (const std::exception& e) {
        fail(std::string("DirectT: second solve threw exception: ") + e.what());
        std::cerr << "[LOG]\n" << log.str() << "\n";
        return 1;
    }
    const auto s2 = ThermalSolverLinearDirect::getDirectTCacheStats();
    expectEqU64(s2.calls, s1.calls + 1, "DirectT: calls increments (2)");
    expectEqU64(s2.rhsOnlyBuild, s1.rhsOnlyBuild + 1, "DirectT: second call uses rhsOnlyBuild (cached)");
    expectEqU64(s2.solveCached, s1.solveCached + 1, "DirectT: second call uses solveCached (cached)");

    // ------------------------------------------------------------
    // coeffSig を変える（aircon on/off）→ fullBuild に戻る
    // ------------------------------------------------------------
    // aircon OFF
    net.getGraph()[net.getKeyToVertex().at("AC")].on = false;
    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, constants, log);
    } catch (const std::exception& e) {
        fail(std::string("DirectT: third solve (aircon off) threw exception: ") + e.what());
        std::cerr << "[LOG]\n" << log.str() << "\n";
        return 1;
    }
    const auto s3 = ThermalSolverLinearDirect::getDirectTCacheStats();
    expectEqU64(s3.calls, s2.calls + 1, "DirectT: calls increments (3)");
    expectEqU64(s3.fullBuild, s2.fullBuild + 1, "DirectT: aircon toggle -> fullBuild");
    expectEqU64(s3.solveFull, s2.solveFull + 1, "DirectT: aircon toggle -> solveFull");

    // もう一回（状態不変: aircon off）→ rhsOnlyBuild に戻る
    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, constants, log);
    } catch (const std::exception& e) {
        fail(std::string("DirectT: fourth solve (aircon off cached) threw exception: ") + e.what());
        std::cerr << "[LOG]\n" << log.str() << "\n";
        return 1;
    }
    const auto s4 = ThermalSolverLinearDirect::getDirectTCacheStats();
    expectEqU64(s4.calls, s3.calls + 1, "DirectT: calls increments (4)");
    expectEqU64(s4.rhsOnlyBuild, s3.rhsOnlyBuild + 1, "DirectT: fourth call uses rhsOnlyBuild (cached)");
    expectEqU64(s4.solveCached, s3.solveCached + 1, "DirectT: fourth call uses solveCached (cached)");

    // ------------------------------------------------------------
    // coeffSig を変える（advection flow_rate を閾値以上に変化）→ fullBuild に戻る
    // ------------------------------------------------------------
    EdgeProperties* advPtr = findEdgeByUniqueId(net.getGraph(), "adv");
    expectTrue(advPtr != nullptr, "DirectT: advection edge exists");
    if (advPtr) {
        advPtr->flow_rate = 0.05; // FLOW_RATE_MIN を超える想定
    }
    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, constants, log);
    } catch (const std::exception& e) {
        fail(std::string("DirectT: fifth solve (flow changed) threw exception: ") + e.what());
        std::cerr << "[LOG]\n" << log.str() << "\n";
        return 1;
    }
    const auto s5 = ThermalSolverLinearDirect::getDirectTCacheStats();
    expectEqU64(s5.calls, s4.calls + 1, "DirectT: calls increments (5)");
    expectEqU64(s5.fullBuild, s4.fullBuild + 1, "DirectT: flow change -> fullBuild");
    expectEqU64(s5.solveFull, s4.solveFull + 1, "DirectT: flow change -> solveFull");

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


