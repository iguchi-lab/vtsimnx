#include "core/ventilation/pressure_solver_impl.h"
#include "core/ventilation/pressure_constraints.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "network/ventilation_network.h"

PressureSolver::Impl::StageBSolveResult PressureSolver::Impl::solveStageBFull(
        const SimulationConstants& constants,
        Graph& g,
        const SupernodePartition& partition,
        const PressureMap& pressureMapFB_A,
        const FallbackLogger& fallbackLog) {
    StageBSolveResult result;
    result.setup = buildStageBSetup(g, pressureMapFB_A);
    ceres::Problem problemFB2;
    auto& vToParamIdxB = result.setup.vertexToParamIndex;
    const auto& nodeNamesFBB = result.setup.nodeNames;
    std::vector<double>& pressuresFBB = result.setup.pressures;

    for (const auto& nodeName : nodeNamesFBB) {
        ceres::CostFunction* costFunction = PressureConstraints::createFlowBalanceConstraint(
            nodeName,
            g,
            network_.getKeyToVertex(),
            result.setup.vertexToParamIndexVec,
            partition.incidentEdgesByVertex,
            pressuresFBB.size(),
            logFile_
        );
        problemFB2.AddResidualBlock(costFunction, nullptr, pressuresFBB.data());
    }
    if (!nodeNamesFBB.empty()) {
        problemFB2.AddResidualBlock(
            PressureConstraints::createSoftAnchorConstraint(0, 0.0, 1e-9, pressuresFBB.size()),
            nullptr,
            pressuresFBB.data());
    }

    result.ok = runStageBTrials(constants, problemFB2, result.summary, fallbackLog);

    auto vrB = boost::vertices(g);
    for (auto v : boost::make_iterator_range(vrB)) {
        const auto& node = g[v];
        if (node.calc_p) {
            size_t idx = vToParamIdxB[v];
            result.pressureMap[node.key] = pressuresFBB[idx];
        } else {
            result.pressureMap[node.key] = node.current_p;
        }
    }

    return result;
}
