#include "core/pressure_solver.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace {

std::string sanitizeLogLabel(const std::string& logMessage) {
    if (logMessage.empty()) return "ソルバー試行";
    size_t start = logMessage.find_first_not_of("- \t");
    std::string label = (start == std::string::npos) ? logMessage : logMessage.substr(start);
    size_t dots = label.find("...");
    if (dots != std::string::npos) {
        label = label.substr(0, dots);
    }
    while (!label.empty() && std::isspace(static_cast<unsigned char>(label.back()))) {
        label.pop_back();
    }
    if (label.empty()) {
        return "ソルバー試行";
    }
    return label;
}

std::string formatSeconds(double seconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << seconds;
    return oss.str();
}

} // namespace

void PressureSolver::logCeresTiming(const std::string& label,
                                    const ceres::Solver::Summary& summary) {
    std::string sanitized = sanitizeLogLabel(label);
    std::ostringstream oss;
    oss << "--------" << sanitized << " 所要時間: " << formatSeconds(summary.total_time_in_seconds) << "秒"
        << " (前処理 " << formatSeconds(summary.preprocessor_time_in_seconds) << "秒"
        << ", 残差評価 " << formatSeconds(summary.residual_evaluation_time_in_seconds) << "秒"
        << ", ヤコビアン評価 " << formatSeconds(summary.jacobian_evaluation_time_in_seconds) << "秒"
        << ", 線形ソルバー " << formatSeconds(summary.linear_solver_time_in_seconds) << "秒"
        << ", 最適化 " << formatSeconds(summary.minimizer_time_in_seconds) << "秒)";
    writeLog(logFile_, oss.str());
}

// =============================================================================
// Ceresソルバー実行ユーティリティ
// =============================================================================

bool PressureSolver::runSolverTrial(const std::string& startLog,
                                    const std::string& successLog,
                                    ceres::Problem& problem,
                                    ceres::Solver::Summary& summary,
                                    double successTolerance,
                                    const std::function<void(ceres::Solver::Options&)>& configureOptions) {
    if (!startLog.empty()) {
        writeLog(logFile_, startLog);
    }
    ceres::Solver::Options options;
    configureOptions(options);
    // 設定された許容誤差を使用（デフォルト値の場合はsuccessToleranceを使用）
    double usedTolerance = (options.function_tolerance > 0.0) ? options.function_tolerance : successTolerance;
    ceres::Solve(options, &problem, &summary);
    logCeresTiming(startLog.empty() ? successLog : startLog, summary);
    // 設定した許容誤差で判定（調整済み許容誤差での収束を許容）
    bool converged = (summary.termination_type == ceres::CONVERGENCE) &&
                     (summary.final_cost <= usedTolerance);
    if (converged && !successLog.empty()) {
        writeLog(logFile_, successLog);
    } else if (!converged) {
        // 収束しなかった場合の詳細情報を出力
        std::string terminationType;
        switch(summary.termination_type) {
            case ceres::CONVERGENCE:
                terminationType = "CONVERGENCE";
                break;
            case ceres::NO_CONVERGENCE:
                terminationType = "NO_CONVERGENCE (最大反復回数到達)";
                break;
            case ceres::FAILURE:
                terminationType = "FAILURE (計算失敗)";
                break;
            case ceres::USER_FAILURE:
                terminationType = "USER_FAILURE (ユーザー関数エラー)";
                break;
            default:
                terminationType = "UNKNOWN (" + std::to_string(static_cast<int>(summary.termination_type)) + ")";
        }
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "-----未収束: 終了理由=" << terminationType
            << ", 最終残差=" << summary.final_cost
            << ", 許容誤差=" << usedTolerance
            << ", 反復回数=" << summary.num_successful_steps;
        writeLog(logFile_, oss.str());
    }
    return converged;
}

// =============================================================================
// プライマリソルバー（初回圧力計算）
// =============================================================================

void PressureSolver::runPrimarySolvers(const SimulationConstants& constants,
                                       ceres::Problem& problem,
                                       ceres::Solver::Summary& summary) {
    bool converged = false;

    converged = runSolverTrial(
        "----①標準設定でソルバーを実行します...",
        "----標準設定で収束しました",
        problem,
        summary,
        constants.ventilationTolerance,
        [&](ceres::Solver::Options& options) {
            options.linear_solver_type = ceres::DENSE_QR;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.max_num_iterations = constants.maxInnerIteration;
            options.function_tolerance = constants.ventilationTolerance;
            options.parameter_tolerance = constants.ventilationTolerance;
            options.minimizer_progress_to_stdout = false;
        });

    if (!converged) {
        converged = runSolverTrial(
            "----②堅牢設定でソルバーを再実行します...",
            "----堅牢設定で収束しました",
            problem,
            summary,
            constants.ventilationTolerance,
            [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = std::max(500, static_cast<int>(constants.maxInnerIteration * 2));
                options.function_tolerance = constants.ventilationTolerance * 0.01;
                options.parameter_tolerance = constants.ventilationTolerance * 0.01;
                options.gradient_tolerance = constants.ventilationTolerance * 0.1;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.max_trust_region_radius = 1e4;
                options.initial_trust_region_radius = 1e2;
                options.minimizer_progress_to_stdout = false;
            });
    }

    if (!converged) {
        converged = runSolverTrial(
            "----③DENSE_SCHUR設定でソルバーを再実行します...",
            "----DENSE_SCHUR設定で収束しました",
            problem,
            summary,
            constants.ventilationTolerance,
            [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.max_num_iterations = 500;
                options.function_tolerance = constants.ventilationTolerance * 0.01;
                options.parameter_tolerance = constants.ventilationTolerance * 0.01;
                options.gradient_tolerance = constants.ventilationTolerance * 0.1;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.max_trust_region_radius = 1e4;
                options.initial_trust_region_radius = 1e2;
                options.minimizer_progress_to_stdout = false;
            });
    }

    if (!converged) {
        converged = runSolverTrial(
            "----④SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します...",
            "----SPARSE_NORMAL_CHOLESKY設定で収束しました",
            problem,
            summary,
            constants.ventilationTolerance,
            [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.max_num_iterations = 1000;
                options.function_tolerance = constants.ventilationTolerance * 0.001;
                options.parameter_tolerance = constants.ventilationTolerance * 0.001;
                options.gradient_tolerance = constants.ventilationTolerance * 0.01;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.inner_iteration_tolerance = 1e-8;
                options.max_trust_region_radius = 1e3;
                options.initial_trust_region_radius = 1e1;
                options.minimizer_progress_to_stdout = false;
            });
    }

    if (!converged) {
        writeLog(logFile_, "----⑤段階的緩和法でソルバーを再実行します...");

        ceres::Solver::Options options1;
        options1.trust_region_strategy_type = ceres::DOGLEG;
        options1.linear_solver_type = ceres::DENSE_QR;
        options1.max_num_iterations = 200;
        options1.function_tolerance = constants.ventilationTolerance * 10;
        options1.parameter_tolerance = constants.ventilationTolerance * 10;
        options1.gradient_tolerance = constants.ventilationTolerance;
        options1.jacobi_scaling = true;
        options1.minimizer_progress_to_stdout = false;

        ceres::Solve(options1, &problem, &summary);
        logCeresTiming("----⑤段階的緩和法(段階1)", summary);
        {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6);
            oss << "-----段階1完了: 残差=" << summary.final_cost
                << ", 終了理由=";
            switch(summary.termination_type) {
                case ceres::CONVERGENCE: oss << "CONVERGENCE"; break;
                case ceres::NO_CONVERGENCE: oss << "NO_CONVERGENCE"; break;
                case ceres::FAILURE: oss << "FAILURE"; break;
                case ceres::USER_FAILURE: oss << "USER_FAILURE"; break;
                default: oss << "UNKNOWN(" << static_cast<int>(summary.termination_type) << ")"; break;
            }
            oss << ", 反復回数=" << summary.num_successful_steps;
            writeLog(logFile_, oss.str());
        }

        ceres::Solver::Options options2;
        options2.trust_region_strategy_type = ceres::DOGLEG;
        options2.linear_solver_type = ceres::DENSE_QR;
        options2.max_num_iterations = 1000;
        options2.function_tolerance = constants.ventilationTolerance * 0.01;
        options2.parameter_tolerance = constants.ventilationTolerance * 0.01;
        options2.gradient_tolerance = constants.ventilationTolerance * 0.1;
        options2.jacobi_scaling = true;
        options2.use_inner_iterations = true;
        options2.minimizer_progress_to_stdout = false;

        ceres::Solve(options2, &problem, &summary);
        logCeresTiming("----⑤段階的緩和法(段階2)", summary);
        // 段階的緩和法の段階2では、設定した許容誤差で判定（調整済み許容誤差での収束を許容）
        converged = (summary.termination_type == ceres::CONVERGENCE) &&
                    (summary.final_cost <= options2.function_tolerance);
        if (!converged) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6);
            oss << "-----段階2未収束: 終了理由=";
            switch(summary.termination_type) {
                case ceres::CONVERGENCE: oss << "CONVERGENCE"; break;
                case ceres::NO_CONVERGENCE: oss << "NO_CONVERGENCE (最大反復回数到達)"; break;
                case ceres::FAILURE: oss << "FAILURE (計算失敗)"; break;
                case ceres::USER_FAILURE: oss << "USER_FAILURE (ユーザー関数エラー)"; break;
                default: oss << "UNKNOWN(" << static_cast<int>(summary.termination_type) << ")"; break;
            }
            oss << ", 最終残差=" << summary.final_cost
                << ", 許容誤差=" << options2.function_tolerance
                << ", 反復回数=" << summary.num_successful_steps;
            writeLog(logFile_, oss.str());
        }

        if (converged) {
            writeLog(logFile_, "----段階的緩和法で収束しました");
        }
    }

    if (!converged) {
        converged = runSolverTrial(
            "----⑥Line Search方式でソルバーを再実行します...",
            "----Line Search方式で収束しました",
            problem,
            summary,
            constants.ventilationTolerance,
            [&](ceres::Solver::Options& options) {
                options.minimizer_type = ceres::LINE_SEARCH;
                options.line_search_direction_type = ceres::LBFGS;
                options.line_search_type = ceres::WOLFE;
                options.max_num_iterations = 1000;
                options.function_tolerance = constants.ventilationTolerance;
                options.parameter_tolerance = constants.ventilationTolerance;
                options.gradient_tolerance = constants.ventilationTolerance * 10;
                options.jacobi_scaling = true;
                options.minimizer_progress_to_stdout = false;
            });
    }

    if (!converged) {
        writeLog(logFile_, "----⑦超精密設定で最終試行します...");
        ceres::Solver::Options options;
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 5000;
        double tolerance_factor = std::max(1.0, summary.final_cost / constants.ventilationTolerance * 0.1);
        options.function_tolerance = constants.ventilationTolerance * tolerance_factor;
        options.parameter_tolerance = constants.ventilationTolerance * tolerance_factor;
                options.gradient_tolerance = constants.ventilationTolerance * tolerance_factor * 10;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.inner_iteration_tolerance = 1e-12;
                options.max_trust_region_radius = 1e2;
                options.initial_trust_region_radius = 1e0;
                options.min_trust_region_radius = 1e-8;
                options.minimizer_progress_to_stdout = false;
        writeLog(logFile_, "-----調整済み許容誤差: " + std::to_string(options.function_tolerance));

        ceres::Solve(options, &problem, &summary);
        logCeresTiming("----⑦超精密設定", summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) &&
                   (summary.final_cost <= options.function_tolerance);

        if (converged) {
            writeLog(logFile_, "----超精密設定で収束しました");
        } else {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6);
            oss << "-----超精密設定未収束: 終了理由=";
            switch(summary.termination_type) {
                case ceres::CONVERGENCE: oss << "CONVERGENCE"; break;
                case ceres::NO_CONVERGENCE: oss << "NO_CONVERGENCE (最大反復回数到達)"; break;
                case ceres::FAILURE: oss << "FAILURE (計算失敗)"; break;
                case ceres::USER_FAILURE: oss << "USER_FAILURE (ユーザー関数エラー)"; break;
                default: oss << "UNKNOWN(" << static_cast<int>(summary.termination_type) << ")"; break;
            }
            oss << ", 最終残差=" << summary.final_cost
                << ", 許容誤差=" << options.function_tolerance
                << ", 反復回数=" << summary.num_successful_steps;
            writeLog(logFile_, oss.str());
        }
    }

        if (!converged) {
            writeLog(logFile_, "----全てのソルバー手法で収束に失敗しました");
            writeLog(logFile_, "----最終残差: " + std::to_string(summary.final_cost) +
                               " (目標: " + std::to_string(constants.ventilationTolerance) + ")");
    }
}

// =============================================================================
// Stage A: スーパーノード代表圧フェーズ
// =============================================================================

PressureSolver::StageAMapping PressureSolver::buildStageAMapping(
    const Graph& graph,
    const std::vector<Vertex>& vertices,
    const std::vector<int>& groupOfVertex) {
    StageAMapping mapping;
    for (size_t i = 0; i < vertices.size(); ++i) {
        Vertex v = vertices[i];
        const auto& node = graph[v];
        if (!node.calc_p) {
            continue;
        }
        int gid = groupOfVertex[i];
        if (gid >= 0) {
            if (!mapping.groupToParamIndex.count(gid)) {
                size_t idx = mapping.groupToParamIndex.size();
                mapping.groupToParamIndex[gid] = idx;
            }
            mapping.vertexToParamIndex[v] = mapping.groupToParamIndex[gid];
        } else {
            size_t idx = mapping.groupToParamIndex.size() + 1000000 + i;
            mapping.vertexToParamIndex[v] = idx;
        }
        mapping.nodeNames.push_back(node.key);
    }

    std::map<size_t, size_t> remap;
    size_t next = mapping.groupToParamIndex.size();
    for (auto& kv : mapping.vertexToParamIndex) {
        size_t oldIdx = kv.second;
        if (oldIdx >= 1000000) {
            auto it = remap.find(oldIdx);
            if (it == remap.end()) {
                size_t newIdx = next++;
                remap[oldIdx] = newIdx;
                kv.second = newIdx;
            } else {
                kv.second = it->second;
            }
        }
    }
    mapping.parameterCount = next;
    return mapping;
}

std::vector<double> PressureSolver::initializeStageAPressures(
    const Graph& graph,
    const StageAMapping& mapping,
    const PressureMap& prevPressureMapFB) {
    std::vector<double> pressures(mapping.parameterCount, 0.0);
    for (const auto& kv : mapping.vertexToParamIndex) {
        Vertex v = kv.first;
        size_t idx = kv.second;
        const auto& node = graph[v];
        double p0 = prevPressureMapFB.count(node.key)
                        ? prevPressureMapFB.at(node.key)
                        : node.current_p;
        if (idx < pressures.size()) {
            pressures[idx] = p0;
        }
    }
    return pressures;
}

void PressureSolver::setupStageAProblem(
    ceres::Problem& problemFB,
    const StageAMapping& mapping,
    Graph& graph,
    const std::vector<Vertex>& vertices,
    const std::vector<int>& groupOfVertex,
    const PressureMap& prevPressureMapFB,
    std::vector<double>& pressuresFB,
    int superCountA) {
    auto& vToParamIdx = mapping.vertexToParamIndex;
    size_t parameterCount = pressuresFB.size();
    double* parameterData = pressuresFB.data();
    const auto& nodeKeyToVertex = network_.getKeyToVertex();

    auto addNodeResidual = [&](const std::string& nodeName) {
        ceres::CostFunction* costFunction = new FlowBalanceConstraint(
            nodeName,
            graph,
            nodeKeyToVertex,
            vToParamIdx,
            parameterCount,
            logFile_
        );
        problemFB.AddResidualBlock(costFunction, nullptr, parameterData);
    };

    if (superCountA > 0) {
        std::vector<std::vector<Vertex>> groupVertices(superCountA);
        std::vector<Vertex> nonGroupVertices;
        for (size_t i = 0; i < vertices.size(); ++i) {
            const auto& node = graph[vertices[i]];
            if (!node.calc_p) continue;
            int gid = groupOfVertex[i];
            if (gid >= 0) {
                groupVertices[gid].push_back(vertices[i]);
            } else {
                nonGroupVertices.push_back(vertices[i]);
            }
        }

        for (const auto& gv : groupVertices) {
            if (gv.empty()) continue;
            ceres::CostFunction* costG = new GroupFlowBalanceConstraint(
                gv,
                graph,
                nodeKeyToVertex,
                vToParamIdx,
                parameterCount,
                logFile_
            );
            problemFB.AddResidualBlock(costG, nullptr, parameterData);
        }

        std::vector<double> groupMean(superCountA, 0.0);
        std::vector<int> groupCount(superCountA, 0);
        for (int gid = 0; gid < superCountA; ++gid) {
            for (auto v : groupVertices[gid]) {
                const auto& node = graph[v];
                auto it = prevPressureMapFB.find(node.key);
                if (it != prevPressureMapFB.end()) {
                    groupMean[gid] += it->second;
                    groupCount[gid]++;
                }
            }
        }

        for (const auto& kv : mapping.groupToParamIndex) {
            int gid = kv.first;
            size_t idx = kv.second;
            double target = (gid >= 0 && gid < superCountA && groupCount[gid] > 0)
                                ? (groupMean[gid] / static_cast<double>(groupCount[gid]))
                                : 0.0;
            SoftAnchorConstraint* constraint = new SoftAnchorConstraint(idx, target, 1e-9, parameterCount);
            problemFB.AddResidualBlock(constraint, nullptr, parameterData);
        }

        for (auto v : nonGroupVertices) {
            const auto& node = graph[v];
            if (!node.calc_p) continue;
            addNodeResidual(node.key);
        }
    } else {
        for (const auto& nodeName : mapping.nodeNames) {
            addNodeResidual(nodeName);
        }
    }
}

// =============================================================================
// Stage B: フルノード再解フェーズ
// =============================================================================

PressureSolver::StageBSetup PressureSolver::buildStageBSetup(
    const Graph& graph,
    const PressureMap& stageAPressureMap) {
    StageBSetup setup;
    size_t nextIndex = 0;
    auto vr = boost::vertices(graph);
    for (auto v : boost::make_iterator_range(vr)) {
        const auto& node = graph[v];
        if (!node.calc_p) {
            continue;
        }
        setup.vertexToParamIndex[v] = nextIndex++;
        setup.nodeNames.push_back(node.key);
    }
    setup.pressures.resize(nextIndex, 0.0);
    for (const auto& kv : setup.vertexToParamIndex) {
        Vertex v = kv.first;
        size_t idx = kv.second;
        const auto& node = graph[v];
        double p0 = stageAPressureMap.count(node.key)
                        ? stageAPressureMap.at(node.key)
                        : node.current_p;
        setup.pressures[idx] = p0;
    }
    return setup;
}

bool PressureSolver::runStageBTrials(const SimulationConstants& constants,
                                     ceres::Problem& problemFB2,
                                     ceres::Solver::Summary& fbSummary2,
                                     const std::function<void(int, const std::string&)>& fallbackLog) {
    bool fbOK2 = false;

    auto logSolve = [&](const std::string& startMsg, const std::string& successMsg,
                        const std::function<void(ceres::Solver::Options&)>& configure) {
        if (fbOK2) return;
        fallbackLog(2, startMsg);
        ceres::Solver::Options opts;
        configure(opts);
        double usedTolerance = opts.function_tolerance;  // 設定された許容誤差を使用
        ceres::Solve(opts, &problemFB2, &fbSummary2);
        logCeresTiming(startMsg, fbSummary2);
        // フォールバックでは設定した許容誤差で判定（調整済み許容誤差での収束を許容）
        fbOK2 = (fbSummary2.termination_type == ceres::CONVERGENCE) &&
                (fbSummary2.final_cost <= usedTolerance);
        if (fbOK2 && !successMsg.empty()) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(2, successMsg + os.str() + " | tol=" + std::to_string(usedTolerance));
        }
    };

    logSolve("[B-①] 標準設定でソルバーを実行します",
             "[B-①] 収束 | residual=",
             [&](ceres::Solver::Options& o) {
                 o.linear_solver_type = ceres::DENSE_QR;
                 o.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                 o.max_num_iterations = constants.maxInnerIteration;
                 o.function_tolerance = constants.ventilationTolerance;
                 o.parameter_tolerance = constants.ventilationTolerance;
                 o.minimizer_progress_to_stdout = false;
             });

    logSolve("[B-②] 堅牢設定でソルバーを再実行します",
             "[B-②] 収束 | residual=",
             [&](ceres::Solver::Options& o) {
                 o.trust_region_strategy_type = ceres::DOGLEG;
                 o.linear_solver_type = ceres::DENSE_QR;
                 o.max_num_iterations = std::max(500, static_cast<int>(constants.maxInnerIteration * 2));
                 o.function_tolerance = constants.ventilationTolerance * 0.01;
                 o.parameter_tolerance = constants.ventilationTolerance * 0.01;
                 o.gradient_tolerance = constants.ventilationTolerance * 0.1;
                 o.jacobi_scaling = true;
                 o.use_inner_iterations = true;
                 o.max_trust_region_radius = 1e4;
                 o.initial_trust_region_radius = 1e2;
                 o.minimizer_progress_to_stdout = false;
             });

    logSolve("[B-③] DENSE_SCHUR設定でソルバーを再実行します",
             "[B-③] 収束 | residual=",
             [&](ceres::Solver::Options& o) {
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
             });

    logSolve("[B-④] SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します",
             "[B-④] 収束 | residual=",
             [&](ceres::Solver::Options& o) {
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
             });

    if (!fbOK2) {
        fallbackLog(2, "[B-⑤] 段階的緩和法でソルバーを再実行します");
        ceres::Solver::Options o1;
        o1.trust_region_strategy_type = ceres::DOGLEG;
        o1.linear_solver_type = ceres::DENSE_QR;
        o1.max_num_iterations = 200;
        o1.function_tolerance = constants.ventilationTolerance * 10;
        o1.parameter_tolerance = constants.ventilationTolerance * 10;
        o1.gradient_tolerance = constants.ventilationTolerance;
        o1.jacobi_scaling = true;
        o1.minimizer_progress_to_stdout = false;
        ceres::Solve(o1, &problemFB2, &fbSummary2);
        logCeresTiming("[B-⑤] 段階1", fbSummary2);
        {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(3, "[B-⑤] 段階1完了 | residual=" + os.str());
        }
        ceres::Solver::Options o2;
        o2.trust_region_strategy_type = ceres::DOGLEG;
        o2.linear_solver_type = ceres::DENSE_QR;
        o2.max_num_iterations = 1000;
        o2.function_tolerance = constants.ventilationTolerance * 0.01;
        o2.parameter_tolerance = constants.ventilationTolerance * 0.01;
        o2.gradient_tolerance = constants.ventilationTolerance * 0.1;
        o2.jacobi_scaling = true;
        o2.use_inner_iterations = true;
        o2.minimizer_progress_to_stdout = false;
        ceres::Solve(o2, &problemFB2, &fbSummary2);
        logCeresTiming("[B-⑤] 段階2", fbSummary2);
        // 段階的緩和法の段階2では、設定した許容誤差で判定（調整済み許容誤差での収束を許容）
        double usedTolerance = o2.function_tolerance;
        fbOK2 = (fbSummary2.termination_type == ceres::CONVERGENCE) &&
                (fbSummary2.final_cost <= usedTolerance);
        if (fbOK2) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(2, "[B-⑤] 収束 | residual=" + os.str() + " | tol=" +
                               std::to_string(usedTolerance));
        }
    }

    logSolve("[B-⑥] Line Search方式でソルバーを再実行します",
             "[B-⑥] 収束 | residual=",
             [&](ceres::Solver::Options& o) {
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

    if (!fbOK2) {
        fallbackLog(2, "[B-⑦] 超精密設定で最終試行します");
        ceres::Solver::Options o;
        o.trust_region_strategy_type = ceres::DOGLEG;
        o.linear_solver_type = ceres::DENSE_QR;
        o.max_num_iterations = 5000;
        double tolerance_factor = std::max(1.0, fbSummary2.final_cost / constants.ventilationTolerance * 0.1);
        o.function_tolerance = constants.ventilationTolerance * tolerance_factor;
        o.parameter_tolerance = constants.ventilationTolerance * tolerance_factor;
        o.gradient_tolerance = constants.ventilationTolerance * tolerance_factor * 10;
        o.jacobi_scaling = true;
        o.use_inner_iterations = true;
        o.inner_iteration_tolerance = 1e-12;
        o.max_trust_region_radius = 1e2;
        o.initial_trust_region_radius = 1e0;
        o.min_trust_region_radius = 1e-8;
        o.minimizer_progress_to_stdout = false;
        fallbackLog(3, "[B-⑦] 調整済み許容誤差=" + std::to_string(o.function_tolerance));
        ceres::Solve(o, &problemFB2, &fbSummary2);
        logCeresTiming("[B-⑦] 超精密設定", fbSummary2);
        fbOK2 = (fbSummary2.termination_type == ceres::CONVERGENCE) &&
                (fbSummary2.final_cost <= o.function_tolerance);
        if (fbOK2) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(2, "[B-⑦] 収束 | residual=" + os.str() + " | tol=" +
                               std::to_string(o.function_tolerance));
        }
    }

    return fbOK2;
}

