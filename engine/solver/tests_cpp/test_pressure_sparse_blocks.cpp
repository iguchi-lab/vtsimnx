#include <ceres/ceres.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "core/ventilation/pressure_constraints.h"
#include "core/ventilation/pressure_solver_trial_spec.h"
#include "core/ventilation/edge_mutation_guard.h"
#include "archenv/include/archenv.h"

double calculateDensity(double temperature) {
    return archenv::STANDARD_ATMOSPHERIC_PRESSURE /
        (archenv::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << message << '\n'; }
}
void near(double a, double b, double tol, const char* message) {
    check(std::isfinite(a) && std::abs(a-b) <= tol, message);
}

struct Network {
    Graph graph;
    std::vector<int> indices;
    std::vector<std::vector<Edge>> incident;
    std::vector<double> pressures;
    std::vector<double> density;
    explicit Network(int n) : indices(n), incident(n), pressures(n), density(n) {
        for (int i = 0; i < n; ++i) {
            auto v = boost::add_vertex(graph);
            indices[i] = i;
            pressures[i] = 20.0 - i * 0.1;
            graph[v].current_p = pressures[i];
            graph[v].current_t = 20.0 + i * 0.2;
            density[i] = calculateDensity(graph[v].current_t);
        }
    }
    Edge add(int s, int t, bool fixed = false) {
        EdgeProperties ep{};
        ep.type = fixed ? "fixed_flow" : "gap";
        ep.a = .002; ep.n = 2.0; ep.current_vol = .03;
        ep.h_from = ep.h_to = .5;
        auto e = boost::add_edge(s, t, ep, graph).first;
        incident[s].push_back(e); incident[t].push_back(e);
        return e;
    }
};

void checkResidual(Network& n, const std::vector<Vertex>& group, bool node) {
    std::ostringstream log;
    std::unique_ptr<ceres::CostFunction> dense(node
        ? PressureConstraints::createFlowBalanceConstraint(group[0], n.graph, n.indices,
            n.incident, &n.density, n.pressures.size(), log)
        : PressureConstraints::createGroupFlowBalanceConstraint(group, n.graph, n.indices,
            n.incident, &n.density, n.pressures.size(), log));
    ceres::Problem sparse;
    if (node) PressureConstraints::addFlowBalanceConstraint(sparse, group[0], n.graph,
        n.indices, n.incident, &n.density, n.pressures);
    else PressureConstraints::addGroupFlowBalanceConstraint(sparse, group, n.graph,
        n.indices, n.incident, &n.density, n.pressures);
    std::vector<ceres::ResidualBlockId> ids;
    sparse.GetResidualBlocks(&ids);
    check(ids.size() == 1, "one balance residual");
    const auto* cost = sparse.GetCostFunctionForResidualBlock(ids[0]);
    std::vector<double*> blocks;
    sparse.GetParameterBlocksForResidualBlock(ids[0], &blocks);
    std::vector<const double*> params(blocks.begin(), blocks.end());
    std::vector<double> jac(blocks.size()), denseJac(n.pressures.size());
    std::vector<double*> jp;
    for (auto& x : jac) jp.push_back(&x);
    double r = 0, reference = 0;
    const double* dp[] = {n.pressures.data()};
    double* dj[] = {denseJac.data()};
    dense->Evaluate(dp, &reference, dj);
    cost->Evaluate(params.data(), &r, jp.data());
    near(r, reference, 1e-14, "scalar and dense residuals agree");
    for (size_t i = 0; i < blocks.size(); ++i) {
        check(cost->parameter_block_sizes()[i] == 1, "scalar block size");
        const auto index = blocks[i] - n.pressures.data();
        near(jac[i], denseJac[index], 1e-14, "scalar and dense derivatives agree");
        const double old = *blocks[i], h = 1e-5;
        double plus, minus;
        *blocks[i] = old+h; cost->Evaluate(params.data(), &plus, nullptr);
        *blocks[i] = old-h; cost->Evaluate(params.data(), &minus, nullptr);
        *blocks[i] = old;
        near(jac[i], (plus-minus)/(2*h), 1e-8, "independent finite difference");
        jp[i] = nullptr;
        cost->Evaluate(params.data(), &r, jp.data());
        near(r, reference, 1e-14, "partial null Jacobian block supported");
        jp[i] = &jac[i];
    }
}
}

int main() {
    {
        // Rebuilds must classify the effective branch after timestep changes and
        // temporary fallback overrides. Caching the type on the edge is unsafe.
        Network n(2);
        n.indices[1] = -1;
        n.density = {1.2, 1.2};
        auto edge = n.add(0, 1);
        auto& ep = n.graph[edge];
        auto evaluate = [&](double expected, double derivative, int columns) {
            ceres::Problem problem;
            PressureConstraints::addFlowBalanceConstraint(problem, 0, n.graph,
                n.indices, n.incident, &n.density, n.pressures);
            std::vector<double> residuals;
            ceres::CRSMatrix jac;
            check(problem.Evaluate({}, nullptr, &residuals, nullptr, &jac), "rebuilt balance evaluates");
            near(residuals.at(0), expected, 1e-12, "rebuilt balance selects the current flow law");
            check(jac.num_cols == columns, "pressure dependencies follow the current flow law");
            if (columns) near(jac.values.at(0), derivative, 1e-12, "rebuilt flow derivative");
        };
        const double gap = -.002 * std::sqrt(.1);
        const double slope = -.001 / std::sqrt(.1);
        evaluate(gap, slope, 1);
        {
            ventilation::EdgeMutationGuard guard(n.graph);
            guard.convertToFixedFlow(edge, .04);
            evaluate(-.04, 0.0, 0);
            guard.convertToFixedFlow(edge, .05);
            evaluate(-.05, 0.0, 0);
        }
        evaluate(gap, slope, 1);
        ep.enabled = {false, true};
        ep.updateForTimestep(0);
        evaluate(0.0, 0.0, 0);
        ep.updateForTimestep(1);
        evaluate(gap, slope, 1);
        ep.type = "fan";
        ep.q_max = .5; ep.q1 = .3; ep.p1 = 50; ep.p_max = 100;
        evaluate(-.5, 0.0, 1);
        ep.has_prescribed_vol = true; ep.current_vol = .06;
        evaluate(-.06, 0.0, 0);
        ep.current_enabled = false;
        evaluate(0.0, 0.0, 0); // disabled takes precedence over prescribed flow
        ep.current_enabled = true; ep.has_prescribed_vol = false;
        ep.vol = {.07}; ep.updateForTimestep(0);
        ep.current_enabled = true;
        evaluate(-.07, 0.0, 0); // legacy C++ vol-only fixed branch
        ep.vol.clear(); ep.type = "unrecognized";
        evaluate(0.0, 0.0, 1); // existing direct-C++ unknown-type behavior
    }
    {
        ceres::Solver::Options o;
        o.linear_solver_type = ceres::DENSE_QR;
        o.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        ventilation::selectPressureLinearSolver(o, 17);
        check(o.linear_solver_type == ceres::DENSE_QR, "small systems retain dense QR");
        ventilation::selectPressureLinearSolver(o, 256);
        const bool available = ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)
            || ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE);
        check(o.linear_solver_type == (available ? ceres::SPARSE_NORMAL_CHOLESKY : ceres::DENSE_QR),
            "large systems select an available sparse backend");
        o.linear_solver_type = ceres::DENSE_QR;
        o.trust_region_strategy_type = ceres::DOGLEG;
        ventilation::selectPressureLinearSolver(o, 256);
        check(o.linear_solver_type == ceres::DENSE_QR, "robust dense fallback remains dense");
    }
    {
        Network n(7);
        n.add(0,1); n.add(0,1); n.add(1,2); n.add(2,3);
        n.add(1,4, true);
        auto disabled = n.add(1,5); n.graph[disabled].current_enabled = false;
        n.indices[3] = -1;  // fixed pressure boundary
        checkResidual(n, {1}, true);
        checkResidual(n, {1,2}, false);
        n.indices[2] = 1;   // Stage A: two vertices share one scalar unknown
        checkResidual(n, {1}, true);
        checkResidual(n, {1,2}, false);
        // A boundary whose endpoints alias must contribute flow but zero d/dp.
        n.indices[3] = 1;
        checkResidual(n, {1,2}, false);
    }
    {
        Network n(2); n.add(0,1,true);
        ceres::Problem problem;
        PressureConstraints::addFlowBalanceConstraint(problem, 0, n.graph, n.indices,
            n.incident, &n.density, n.pressures);
        check(problem.NumParameters() == 0, "fixed-flow balance has no pressure dependency");
        double cost = 0;
        std::vector<double> residuals;
        check(problem.Evaluate({}, &cost, &residuals, nullptr, nullptr), "constant balance evaluates");
        near(residuals.at(0), -.03, 1e-15, "unbalanced fixed flow is not silently dropped");
        ceres::Solver::Options options; ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        near(summary.final_cost, .5*.03*.03, 1e-15, "constant cost retained by Ceres");
    }
    for (int count : {17, 64, 256}) {
        Network n(count);
        for (int i=1; i<count; ++i) n.add(i-1,i);
        ceres::Problem problem;
        for (int i=0; i<count; ++i) {
            PressureConstraints::addFlowBalanceConstraint(problem, i, n.graph, n.indices,
                n.incident, &n.density, n.pressures);
        }
        ceres::CRSMatrix jac;
        check(problem.Evaluate({}, nullptr, nullptr, nullptr, &jac), "chain Jacobian evaluates");
        check(jac.num_rows == count && jac.num_cols == count, "chain dimensions");
        check(static_cast<int>(jac.values.size()) == 3*count-2,
            "chain Jacobian storage grows linearly, not quadratically");
    }
    return failures == 0 ? 0 : 1;
}
