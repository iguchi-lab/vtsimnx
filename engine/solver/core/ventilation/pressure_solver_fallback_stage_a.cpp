#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/pressure_solver_internal.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

PressureSolver::StageASolveResult PressureSolver::solveStageAReduced(
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

    struct TrialSpecA {
        const char* label;
        std::function<void(ceres::Solver::Options&)> configure;
    };
    const std::vector<TrialSpecA> trialsA = {
        {"[A-①] 標準設定でソルバーを実行します", [&](ceres::Solver::Options& o) {
             o.linear_solver_type = ceres::DENSE_QR;
             o.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
             o.max_num_iterations = constants.maxInnerIterations;
             o.function_tolerance = constants.ventilationTolerance;
             o.parameter_tolerance = constants.ventilationTolerance;
             o.minimizer_progress_to_stdout = false;
         }},
        {"[A-②] 堅牢設定でソルバーを再実行します", [&](ceres::Solver::Options& o) {
             o.trust_region_strategy_type = ceres::DOGLEG;
             o.linear_solver_type = ceres::DENSE_QR;
             o.max_num_iterations = std::max(500, static_cast<int>(constants.maxInnerIterations * 2));
             o.function_tolerance = constants.ventilationTolerance * 0.01;
             o.parameter_tolerance = constants.ventilationTolerance * 0.01;
             o.gradient_tolerance = constants.ventilationTolerance * 0.1;
             o.jacobi_scaling = true;
             o.use_inner_iterations = true;
             o.max_trust_region_radius = 1e4;
             o.initial_trust_region_radius = 1e2;
             o.minimizer_progress_to_stdout = false;
         }},
        {"[A-③] DENSE_SCHUR設定でソルバーを再実行します", [&](ceres::Solver::Options& o) {
             o.trust_region_strategy_type = ceres::DOGLEG;
             o.linear_solver_type = ceres::DENSE_SCHUR;
             o.max_num_iterations = 500;
             o.function_tolerance = constants.ventilationTolerance * 0.01;
             o.parameter_tolerance = constants.ventilationTolerance * 0.01;
             o.gradient_tolerance = constants.ventilationTolerance * 0.1;
             o.jacobi_scaling = true;
             o.use_inner_iterations = true;
             o.max_trust_region_radius = 1e4;
             o.initial_trust_region_radius = 1e2;
             o.minimizer_progress_to_stdout = false;
         }},
        {"[A-④] SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します", [&](ceres::Solver::Options& o) {
             o.trust_region_strategy_type = ceres::DOGLEG;
             o.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
             o.max_num_iterations = 1000;
             o.function_tolerance = constants.ventilationTolerance * 0.001;
             o.parameter_tolerance = constants.ventilationTolerance * 0.001;
             o.gradient_tolerance = constants.ventilationTolerance * 0.01;
             o.jacobi_scaling = true;
             o.use_inner_iterations = true;
             o.inner_iteration_tolerance = 1e-8;
             o.max_trust_region_radius = 1e3;
             o.initial_trust_region_radius = 1e1;
             o.minimizer_progress_to_stdout = false;
         }},
    };
    for (const auto& t : trialsA) {
        if (fbOKA) break;
        tryTrialA(t.label, t.configure);
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

    tryTrialA("[A-⑥] Line Search方式でソルバーを再実行します", [&](ceres::Solver::Options& o) {
        o.minimizer_type = ceres::LINE_SEARCH;
        o.line_search_direction_type = ceres::LBFGS;
        o.line_search_type = ceres::WOLFE;
        o.max_num_iterations = 1000;
        o.function_tolerance = constants.ventilationTolerance;
        o.parameter_tolerance = constants.ventilationTolerance;
        o.gradient_tolerance = constants.ventilationTolerance * 10;
        o.jacobi_scaling = true;
        o.minimizer_progress_to_stdout = false;
    });

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
