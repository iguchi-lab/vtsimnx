#include <cmath>
#include <iostream>
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

void testMakeTolerancesUsesVentilationTolerance() {
    SimulationConstants c{};
    c.ventilationTolerance = 3.5e-4;
    const auto t = ventilation::makePressureSolverTolerances(c);
    expectNear(t.massBalanceMaxAbs, 3.5e-4, 0.0, "massBalanceMaxAbs from ventilationTolerance");
    expectTrue(t.ceresFunctionRelative > 0.0, "ceres relative default");
}

} // namespace

int main() {
    testBalanceMetricsIndependentOfNodeCount();
    testAcceptMassBalanceBoundary();
    testEdgeMutationGuardRestoreNormalAndException();
    testMakeTolerancesUsesVentilationTolerance();

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[DONE] failures=" << g_failures << "\n";
    return 1;
}
