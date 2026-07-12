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
        auto it = network_.getKeyToVertex().find(nodeName);
        if (it == network_.getKeyToVertex().end()) {
            continue;
        }
        ceres::CostFunction* costFunction = PressureConstraints::createFlowBalanceConstraint(
            it->second,
            g,
            result.setup.vertexToParamIndexVec,
            partition.incidentEdgesByVertex,
            network_.densityCache(),
            pressuresFBB.size(),
            logFile_
        );
        problemFB2.AddResidualBlock(costFunction, nullptr, pressuresFBB.data());
    }
    // Primary と同じ連結成分解析で、固定圧境界のない成分へゲージアンカーを付ける。
    addPressureGaugeAnchors(g,
                            result.setup.vertexToParamIndexVec,
                            pressuresFBB,
                            &partition.incidentEdgesByVertex,
                            problemFB2,
                            /*anchorWeight=*/1.0);

    result.ok = runStageBTrials(
        constants,
        problemFB2,
        result.summary,
        result.setup,
        ventilation::makePressureSolverTolerances(constants).massBalanceMaxAbs,
        fallbackLog);

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
