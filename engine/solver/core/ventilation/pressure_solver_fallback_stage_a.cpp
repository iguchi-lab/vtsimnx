#include "core/ventilation/pressure_solver_impl.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "core/ventilation/pressure_solver_trial_spec.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

PressureSolver::Impl::StageASolveResult PressureSolver::Impl::solveStageAReduced(
        const SimulationConstants& constants,
        Graph& g,
        const SupernodePartition& partition,
        const PressureMap& prevPressureMapFB,
        const FallbackLogger& fallbackLog) {
    auto log2 = [&](const std::string& msg) { fallbackLog(2, msg); };

    StageASolveResult result;
    result.mapping = buildStageAMapping(g, partition.vertices, partition.groupOfVertex);
    auto& vToParamIdx = result.mapping.vertexToParamIndex;
    result.pressures = initializeStageAPressures(g, result.mapping, prevPressureMapFB);

    ceres::Problem problemFB;
    result.superCount = *std::max_element(partition.groupOfVertex.begin(), partition.groupOfVertex.end()) + 1;
    setupStageAProblem(
        problemFB,
        result.mapping,
        g,
        partition.vertices,
        partition.groupOfVertex,
        prevPressureMapFB,
        result.pressures,
        result.superCount,
        partition.incidentEdgesByVertex);

    bool fbOKA = false;
    auto tryTrialA = [&](const std::string& label,
                         const std::function<void(ceres::Solver::Options&)>& configure) {
        if (fbOKA) return;
        TrialResult r = runSolverTrial(label,
                                       /*successLog=*/"",
                                       problemFB,
                                       result.summary,
                                       constants.ventilationTolerance,
                                       configure,
                                       log2);
        fbOKA = r.converged;
        if (fbOKA) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << result.summary.final_cost;
            fallbackLog(2, label + " 収束 | residual=" + os.str() + " | tol=" +
                               std::to_string(r.usedTolerance));
        }
    };

    for (const auto& t : ventilation::stageATrustRegionTrials()) {
        if (fbOKA) break;
        tryTrialA(t.startLog, [&](ceres::Solver::Options& o) { t.configure(o, constants); });
    }

    if (!fbOKA) {
        fallbackLog(2, "[A-⑤] 段階的緩和法でソルバーを再実行します");
        TrialResult r = runTwoStageRelaxation(
            constants,
            problemFB,
            result.summary,
            "[A-⑤] 段階1",
            "[A-⑤] 段階2",
            [&](const ceres::Solver::Summary& s1) {
                std::ostringstream os;
                os << std::scientific << std::setprecision(6) << s1.final_cost;
                fallbackLog(3, "[A-⑤] 段階1完了 | residual=" + os.str());
            },
            log2);
        fbOKA = r.converged;
        if (fbOKA) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << result.summary.final_cost;
            fallbackLog(2, "[A-⑤] 収束 | residual=" + os.str() + " | tol=" +
                               std::to_string(r.usedTolerance));
        }
    }

    tryTrialA("[A-⑥] Line Search方式でソルバーを再実行します",
              [&](ceres::Solver::Options& o) { ventilation::configureLineSearchLbfgs(o, constants); });

    if (!fbOKA) {
        fallbackLog(2, "[A-⑦] 超精密設定で最終試行します");
        const double refCost = result.summary.final_cost;
        TrialResult r = runUltraPreciseTrial(
            constants,
            problemFB,
            result.summary,
            "[A-⑦] 超精密設定",
            refCost,
            [&](double usedTol) {
                fallbackLog(3, "[A-⑦] 調整済み許容誤差=" + std::to_string(usedTol));
            },
            log2);
        fbOKA = r.converged;
        if (fbOKA) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << result.summary.final_cost;
            fallbackLog(2, "[A-⑦] 収束 | residual=" + os.str() + " | tol=" +
                               std::to_string(r.usedTolerance));
        }
    }

    if (!fbOKA) {
        std::ostringstream os;
        os.setf(std::ios::scientific);
        os << std::setprecision(6) << result.summary.final_cost;
        fallbackLog(2, "[A] 未収束 | residual=" + os.str() +
                           " | tol=" + std::to_string(constants.ventilationTolerance));
    }
    result.ok = fbOKA;

    auto vrA = boost::vertices(g);
    for (auto v : boost::make_iterator_range(vrA)) {
        const auto& node = g[v];
        if (node.calc_p) {
            size_t idx = vToParamIdx[v];
            result.pressureMap[node.key] = result.pressures[idx];
        } else {
            result.pressureMap[node.key] = node.current_p;
        }
    }

    if (result.superCount > 0) {
        double sumG0 = 0.0;
        int cntG0 = 0;
        for (size_t i = 0; i < partition.vertices.size(); ++i) {
            if (partition.groupOfVertex[i] != 0) continue;
            const auto& node = g[partition.vertices[i]];
            if (!node.calc_p) continue;
            auto it = result.pressureMap.find(node.key);
            if (it != result.pressureMap.end()) {
                sumG0 += it->second;
                cntG0++;
            }
        }
        if (cntG0 > 0) {
            result.anchorTargetPressure = sumG0 / static_cast<double>(cntG0);
            result.hasAnchorTarget = true;
        }
    }

    return result;
}
