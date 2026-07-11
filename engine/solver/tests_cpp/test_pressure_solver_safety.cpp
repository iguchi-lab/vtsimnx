#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/ventilation/edge_mutation_guard.h"
#include "core/ventilation/pressure_balance.h"
#include "vtsim_solver.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectNear(double a, double b, double tol, const std::string& msg) {
    if (!(std::abs(a - b) <= tol)) {
        fail(msg + " (a=" + std::to_string(a) + ", b=" + std::to_string(b) + ")");
    }
}

void testBalanceMetricsIndependentOfNodeCount() {
    FlowBalanceMap small{{"A", 1e-4}, {"B", -1e-4}};
    FlowBalanceMap large = small;
    for (int i = 0; i < 50; ++i) {
        large["N" + std::to_string(i)] = 1e-4 * ((i % 2 == 0) ? 1.0 : -1.0);
    }
    // 追加ノードも |r|=1e-4 なので maxAbs は同じ
    const auto ms = ventilation::computeBalanceMetrics(small);
    const auto ml = ventilation::computeBalanceMetrics(large);
    expectNear(ms.maxAbs, 1e-4, 1e-15, "small maxAbs");
    expectNear(ml.maxAbs, 1e-4, 1e-15, "large maxAbs same local residual");
    expectTrue(ventilation::acceptMassBalance(ms, 1e-4), "accept at boundary");
    expectTrue(!ventilation::acceptMassBalance(ms, 1e-5), "reject below maxAbs");
}

void testAcceptMassBalanceBoundary() {
    FlowBalanceMap bal{{"A", 2e-3}, {"B", -1e-3}};
    const auto m = ventilation::computeBalanceMetrics(bal);
    expectNear(m.maxAbs, 2e-3, 1e-15, "maxAbs");
    expectTrue(ventilation::acceptMassBalance(m, 2e-3), "equal tol accepts");
    expectTrue(!ventilation::acceptMassBalance(m, 1.999e-3), "strictly below rejects");
    expectTrue(!ventilation::acceptMassBalance(m, 0.0), "non-positive tol rejects");
}

void testEdgeMutationGuardRestoreNormalAndException() {
    Graph g;
    Vertex a = boost::add_vertex(g);
    Vertex b = boost::add_vertex(g);
    g[a].key = "A";
    g[b].key = "B";
    EdgeProperties ep{};
    ep.type = "gap";
    ep.unique_id = "e1";
    ep.current_vol = 1.25;
    bool ok = false;
    Edge e;
    boost::tie(e, ok) = boost::add_edge(a, b, ep, g);
    expectTrue(ok, "add_edge");

    {
        ventilation::EdgeMutationGuard guard(g);
        guard.convertToFixedFlow(e, 9.0);
        expectTrue(g[e].type == "fixed_flow", "mutated type");
        expectNear(g[e].current_vol, 9.0, 1e-15, "mutated vol");
    }
    expectTrue(g[e].type == "gap", "restored type after scope");
    expectNear(g[e].current_vol, 1.25, 1e-15, "restored vol after scope");

    try {
        ventilation::EdgeMutationGuard guard(g);
        guard.convertToFixedFlow(e, -3.0);
        expectTrue(g[e].type == "fixed_flow", "mutated before throw");
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
        // expected
    }
    expectTrue(g[e].type == "gap", "restored type after exception");
    expectNear(g[e].current_vol, 1.25, 1e-15, "restored vol after exception");
}

void testFixedPressureBoundaryExcludedFromAcceptance() {
    // outside(fixed) --q--> room(calc_p) --q--> outside2(fixed)
    // room balance = 0, outside balances = ±q. 全ノード maxAbs は |q| だが
    // 計算ノードのみなら合格する。
    Graph g;
    Vertex out1 = boost::add_vertex(g);
    Vertex room = boost::add_vertex(g);
    Vertex out2 = boost::add_vertex(g);
    g[out1].key = "outside";
    g[out1].calc_p = false;
    g[room].key = "room";
    g[room].calc_p = true;
    g[out2].key = "outside2";
    g[out2].calc_p = false;

    const double q = 0.05;
    FlowBalanceMap bal{
        {"outside", -q},
        {"room", 0.0},
        {"outside2", +q},
    };

    const auto allMetrics = ventilation::computeBalanceMetrics(bal);
    const auto solvedMetrics = ventilation::computePressureUnknownBalanceMetrics(bal, g);
    expectNear(allMetrics.maxAbs, q, 1e-15, "all-node maxAbs includes boundary");
    expectNear(solvedMetrics.maxAbs, 0.0, 1e-15, "solved-node maxAbs ignores boundary");
    expectTrue(solvedMetrics.nodeCount == 1, "only room counted");
    expectTrue(!ventilation::acceptMassBalance(allMetrics, 1e-6), "all-node would reject");
    expectTrue(ventilation::acceptMassBalance(solvedMetrics, 1e-6), "solved-node accepts");
}

void testMissingCalcPBalanceRejected() {
    Graph g;
    Vertex room = boost::add_vertex(g);
    Vertex out = boost::add_vertex(g);
    g[room].key = "room";
    g[room].calc_p = true;
    g[out].key = "outside";
    g[out].calc_p = false;

    // calc_p の room が balance に無い → complete=false / 不合格
    FlowBalanceMap bal{{"outside", -0.1}};
    const auto m = ventilation::computePressureUnknownBalanceMetrics(bal, g);
    expectTrue(!m.complete, "missing calc_p entry marks incomplete");
    expectTrue(m.nodeCount == 0, "missing node not counted");
    expectTrue(!ventilation::acceptMassBalance(m, 1e-6), "missing calc_p balance rejects");
}

void testNonFiniteBalanceRejected() {
    Graph g;
    Vertex room = boost::add_vertex(g);
    g[room].key = "room";
    g[room].calc_p = true;

    FlowBalanceMap balNan{{"room", std::numeric_limits<double>::quiet_NaN()}};
    const auto mNan = ventilation::computePressureUnknownBalanceMetrics(balNan, g);
    expectTrue(!mNan.finite, "NaN marks non-finite");
    expectTrue(!ventilation::acceptMassBalance(mNan, 1e-6), "NaN balance rejects");

    FlowBalanceMap balInf{{"room", std::numeric_limits<double>::infinity()}};
    const auto mInf = ventilation::computePressureUnknownBalanceMetrics(balInf, g);
    expectTrue(!mInf.finite, "Inf marks non-finite");
    expectTrue(!ventilation::acceptMassBalance(mInf, 1e-6), "Inf balance rejects");

    // 全ノード集計側も同様
    const auto allNan = ventilation::computeBalanceMetrics(balNan);
    expectTrue(!allNan.finite, "all-node NaN non-finite");
    expectTrue(!ventilation::acceptMassBalance(allNan, 1.0), "all-node NaN rejects");
}

void testFallbackStageBGate() {
    expectTrue(ventilation::canProceedToFallbackStageB(true, false), "ok proceeds");
    expectTrue(!ventilation::canProceedToFallbackStageB(false, false), "Stage A fail blocks");
    expectTrue(!ventilation::canProceedToFallbackStageB(true, true), "freeze skip blocks");
    expectTrue(!ventilation::canProceedToFallbackStageB(false, true), "both fail blocks");
}

void testInterfaceFlowConsistencyAcceptance() {
    ventilation::InterfaceFlowConsistency good;
    good.maxAbs = 1e-7;
    good.edgeCount = 2;
    good.finite = true;
    good.ok = true;
    expectTrue(ventilation::acceptInterfaceFlowConsistency(good, 1e-6), "iface accept");

    ventilation::InterfaceFlowConsistency badAbs = good;
    badAbs.maxAbs = 1e-3;
    expectTrue(!ventilation::acceptInterfaceFlowConsistency(badAbs, 1e-6), "iface reject large");

    ventilation::InterfaceFlowConsistency badFinite = good;
    badFinite.finite = false;
    expectTrue(!ventilation::acceptInterfaceFlowConsistency(badFinite, 1e-6), "iface reject nonfinite");

    ventilation::InterfaceFlowConsistency badOk = good;
    badOk.ok = false;
    expectTrue(!ventilation::acceptInterfaceFlowConsistency(badOk, 1e-6), "iface reject !ok");
}

void testFallbackAcceptanceRequiresRestoredMassAndInterface() {
    // 採用条件は mass と iface の両方が必要（仮ネットワーク単独合格では不十分）
    ventilation::BalanceMetrics massOk{};
    massOk.maxAbs = 0.0;
    massOk.nodeCount = 1;
    massOk.complete = true;
    massOk.finite = true;
    ventilation::InterfaceFlowConsistency ifaceBad{};
    ifaceBad.maxAbs = 1.0;
    ifaceBad.edgeCount = 1;
    ifaceBad.finite = true;
    ifaceBad.ok = true;
    expectTrue(ventilation::acceptMassBalance(massOk, 1e-6), "mass alone ok");
    expectTrue(!ventilation::acceptInterfaceFlowConsistency(ifaceBad, 1e-6), "iface alone bad");
    expectTrue(!(ventilation::acceptMassBalance(massOk, 1e-6) &&
                 ventilation::acceptInterfaceFlowConsistency(ifaceBad, 1e-6)),
               "both required: mass ok iface bad rejects");

    ventilation::InterfaceFlowConsistency ifaceOk = ifaceBad;
    ifaceOk.maxAbs = 0.0;
    expectTrue(ventilation::acceptMassBalance(massOk, 1e-6) &&
                   ventilation::acceptInterfaceFlowConsistency(ifaceOk, 1e-6),
               "both ok accepts");
}

void testMakeTolerancesUsesVentilationTolerance() {
    SimulationConstants c{};
    c.ventilationTolerance = 3.5e-4;
    const auto t = ventilation::makePressureSolverTolerances(c);
    expectNear(t.massBalanceMaxAbs, 3.5e-4, 0.0, "massBalanceMaxAbs from ventilationTolerance");
    expectNear(t.interfaceFlowMaxAbs, 3.5e-4, 0.0, "interfaceFlowMaxAbs from ventilationTolerance");
    expectTrue(t.ceresFunctionRelative > 0.0, "ceres relative default");
}

} // namespace

int main() {
    testBalanceMetricsIndependentOfNodeCount();
    testAcceptMassBalanceBoundary();
    testEdgeMutationGuardRestoreNormalAndException();
    testFixedPressureBoundaryExcludedFromAcceptance();
    testMissingCalcPBalanceRejected();
    testNonFiniteBalanceRejected();
    testFallbackStageBGate();
    testInterfaceFlowConsistencyAcceptance();
    testFallbackAcceptanceRequiresRestoredMassAndInterface();
    testMakeTolerancesUsesVentilationTolerance();

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[DONE] failures=" << g_failures << "\n";
    return 1;
}
