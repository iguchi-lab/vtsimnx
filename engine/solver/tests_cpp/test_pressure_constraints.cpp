#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>

#include "core/ventilation/flow_calculation.h"
#include "core/ventilation/flow_jacobian.h"
#include "core/ventilation/pressure_constraints.h"

#include "../archenv/include/archenv.h"

// pressure_constraints.cpp は calculateDensity を外部参照する（pressure_solver.cpp に本体）。
// constraints 単体テストでは最小リンクにするため、ここで提供する。
double calculateDensity(double temperature) {
    return archenv::STANDARD_ATMOSPHERIC_PRESSURE /
           (archenv::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        fail(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) +
             ", diff=" + std::to_string(diff) + ", tol=" + std::to_string(tol) + ")");
    }
}

struct TwoNodeOneEdgeFixture {
    Graph g;
    Vertex vA{};
    Vertex vB{};
    Edge eAB{};
    bool okEdge = false;

    std::unordered_map<std::string, Vertex> nodeKeyToVertex;
    std::vector<int> vertexToParameterIndexVec;
    std::vector<std::vector<Edge>> incidentEdgesByVertex;
    std::ostringstream log;

    TwoNodeOneEdgeFixture() {
        vA = boost::add_vertex(g);
        vB = boost::add_vertex(g);

        g[vA].key = "A";
        g[vB].key = "B";
        g[vA].current_p = 0.0;
        g[vB].current_p = 0.0;
        g[vA].current_t = 20.0;
        g[vB].current_t = 20.0;

        EdgeProperties ep{};
        ep.type = "simple_opening";
        ep.alpha = 0.6;
        ep.area = 1.2;
        ep.h_from = 0.0;
        ep.h_to = 0.0;

        boost::tie(eAB, okEdge) = boost::add_edge(vA, vB, ep, g);
        expectTrue(okEdge, "fixture: add_edge succeeded");

        nodeKeyToVertex.emplace("A", vA);
        nodeKeyToVertex.emplace("B", vB);

        const size_t nV = boost::num_vertices(g);
        vertexToParameterIndexVec.assign(nV, -1);
        vertexToParameterIndexVec[static_cast<size_t>(vA)] = 0;
        vertexToParameterIndexVec[static_cast<size_t>(vB)] = 1;

        incidentEdgesByVertex.assign(nV, {});
        incidentEdgesByVertex[static_cast<size_t>(vA)].push_back(eAB);
        incidentEdgesByVertex[static_cast<size_t>(vB)].push_back(eAB);
    }
};

double evalResidual(ceres::CostFunction& f, const std::vector<double>& p) {
    const double* params[] = {p.data()};
    double r[1] = {0.0};
    const bool ok = f.Evaluate(params, r, nullptr);
    expectTrue(ok, "Evaluate returns true");
    return r[0];
}

std::vector<double> evalJacobian(ceres::CostFunction& f, const std::vector<double>& p) {
    const double* params[] = {p.data()};
    double r[1] = {0.0};
    std::vector<double> J(p.size(), 0.0);
    double* jacobians[] = {J.data()};
    const bool ok = f.Evaluate(params, r, jacobians);
    expectTrue(ok, "Evaluate returns true (with jacobians)");
    return J;
}

} // namespace

int main() {
    // -----------------------------
    // FlowBalanceConstraint: residual sign and analytic Jacobian
    // -----------------------------
    {
        TwoNodeOneEdgeFixture fx;
        std::unique_ptr<ceres::CostFunction> cA(
            PressureConstraints::createFlowBalanceConstraint(
                "A", fx.g, fx.nodeKeyToVertex, fx.vertexToParameterIndexVec,
                fx.incidentEdgesByVertex, /*numParameters=*/2, fx.log));
        std::unique_ptr<ceres::CostFunction> cB(
            PressureConstraints::createFlowBalanceConstraint(
                "B", fx.g, fx.nodeKeyToVertex, fx.vertexToParameterIndexVec,
                fx.incidentEdgesByVertex, /*numParameters=*/2, fx.log));

        std::vector<double> p = {10.0, 0.0}; // dp = pA - pB > 0 => flow A->B
        const double dp = p[0] - p[1];
        const double q = FlowCalculation::calculateUnifiedFlow(dp, fx.g[fx.eAB]);
        const double dQdp = FlowJacobianCommon::calculateJacobian(dp, fx.g[fx.eAB]);

        expectNear(evalResidual(*cA, p), -q, 1e-12, "flow_balance(A): residual == -Q");
        expectNear(evalResidual(*cB, p), +q, 1e-12, "flow_balance(B): residual == +Q");

        const auto JA = evalJacobian(*cA, p);
        const auto JB = evalJacobian(*cB, p);
        expectNear(JA[0], -dQdp, 1e-9, "flow_balance(A): dR/dpA");
        expectNear(JA[1], +dQdp, 1e-9, "flow_balance(A): dR/dpB");
        expectNear(JB[0], +dQdp, 1e-9, "flow_balance(B): dR/dpA");
        expectNear(JB[1], -dQdp, 1e-9, "flow_balance(B): dR/dpB");

        // numerical derivative check
        const double h = 1e-6;
        for (int i = 0; i < 2; ++i) {
            std::vector<double> p_plus = p;
            std::vector<double> p_minus = p;
            p_plus[static_cast<size_t>(i)] += h;
            p_minus[static_cast<size_t>(i)] -= h;
            const double dnumA = (evalResidual(*cA, p_plus) - evalResidual(*cA, p_minus)) / (2.0 * h);
            const double dnumB = (evalResidual(*cB, p_plus) - evalResidual(*cB, p_minus)) / (2.0 * h);
            expectNear(JA[static_cast<size_t>(i)], dnumA, 1e-6, "flow_balance(A): numeric jacobian match");
            expectNear(JB[static_cast<size_t>(i)], dnumB, 1e-6, "flow_balance(B): numeric jacobian match");
        }
    }

    // -----------------------------
    // FlowBalanceConstraint: missing node should be zero residual/jacobian
    // -----------------------------
    {
        TwoNodeOneEdgeFixture fx;
        std::unique_ptr<ceres::CostFunction> cMissing(
            PressureConstraints::createFlowBalanceConstraint(
                "NO_SUCH_NODE", fx.g, fx.nodeKeyToVertex, fx.vertexToParameterIndexVec,
                fx.incidentEdgesByVertex, /*numParameters=*/2, fx.log));

        std::vector<double> p = {10.0, 0.0};
        const double r = evalResidual(*cMissing, p);
        expectNear(r, 0.0, 0.0, "flow_balance(missing): residual == 0");
        const auto J = evalJacobian(*cMissing, p);
        expectNear(J[0], 0.0, 0.0, "flow_balance(missing): dR/dp0 == 0");
        expectNear(J[1], 0.0, 0.0, "flow_balance(missing): dR/dp1 == 0");
    }

    // -----------------------------
    // GroupFlowBalanceConstraint: group={A} should match flow_balance(A)
    // -----------------------------
    {
        TwoNodeOneEdgeFixture fx;
        std::vector<Vertex> group = {fx.vA};
        std::unique_ptr<ceres::CostFunction> cGroup(
            PressureConstraints::createGroupFlowBalanceConstraint(
                group, fx.g, fx.vertexToParameterIndexVec, fx.incidentEdgesByVertex,
                /*numParameters=*/2, fx.log));

        std::vector<double> p = {10.0, 0.0};
        const double dp = p[0] - p[1];
        const double q = FlowCalculation::calculateUnifiedFlow(dp, fx.g[fx.eAB]);
        const double dQdp = FlowJacobianCommon::calculateJacobian(dp, fx.g[fx.eAB]);

        expectNear(evalResidual(*cGroup, p), -q, 1e-12, "group_flow_balance({A}): residual == -Q");
        const auto J = evalJacobian(*cGroup, p);
        expectNear(J[0], -dQdp, 1e-9, "group_flow_balance({A}): dR/dpA");
        expectNear(J[1], +dQdp, 1e-9, "group_flow_balance({A}): dR/dpB");
    }

    // -----------------------------
    // SoftAnchorConstraint: residual and jacobian
    // -----------------------------
    {
        std::unique_ptr<ceres::CostFunction> c(
            PressureConstraints::createSoftAnchorConstraint(/*parameterIndex=*/0,
                                                           /*targetPressure=*/123.0,
                                                           /*weight=*/0.5,
                                                           /*numParameters=*/2));
        std::vector<double> p = {10.0, 0.0};
        expectNear(evalResidual(*c, p), 0.5 * (10.0 - 123.0), 1e-12, "soft_anchor: residual");
        const auto J = evalJacobian(*c, p);
        expectNear(J[0], 0.5, 0.0, "soft_anchor: dR/dp0 == weight");
        expectNear(J[1], 0.0, 0.0, "soft_anchor: dR/dp1 == 0");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


