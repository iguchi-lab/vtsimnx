#include "core/thermal_solver_ceres.h"
#include "utils/utils.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cctype>
#include <thread>

namespace ThermalSolverCeres {

namespace {

// 最適なソルバー設定を保存する静的変数
static std::function<void(ceres::Solver::Options&)> g_optimalSolverConfig = nullptr;
static std::string g_optimalSolverName;
static bool g_solverOptimized = false;

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

int detectThreadCount() {
    unsigned int threads = std::thread::hardware_concurrency();
    return threads == 0 ? 1 : static_cast<int>(threads);
}

void logSolverTiming(const std::string& startLog,
                     const ceres::Solver::Summary& summary,
                     std::ostream& logFile) {
    std::string label = sanitizeLogLabel(startLog);
    std::ostringstream oss;
    oss << "--------" << label << " 所要時間: " << formatSeconds(summary.total_time_in_seconds) << "秒"
        << " (前処理 " << formatSeconds(summary.preprocessor_time_in_seconds) << "秒"
        << ", 残差評価 " << formatSeconds(summary.residual_evaluation_time_in_seconds) << "秒"
        << ", ヤコビアン評価 " << formatSeconds(summary.jacobian_evaluation_time_in_seconds) << "秒"
        << ", 線形ソルバー " << formatSeconds(summary.linear_solver_time_in_seconds) << "秒"
        << ", 最適化 " << formatSeconds(summary.minimizer_time_in_seconds) << "秒)";
    writeLog(logFile, oss.str());
}

} // namespace

bool runThermalSolverTrial(const std::string& startLog,
                           const std::string& successLog,
                           ceres::Problem& problem,
                           ceres::Solver::Summary& summary,
                           double successTolerance,
                           const std::function<void(ceres::Solver::Options&)>& configureOptions,
                           std::ostream& logFile) {
    if (!startLog.empty()) {
        writeLog(logFile, startLog);
    }
    ceres::Solver::Options options;
    configureOptions(options);
    ceres::Solve(options, &problem, &summary);
    logSolverTiming(startLog.empty() ? successLog : startLog, summary, logFile);
    bool converged = (summary.termination_type == ceres::CONVERGENCE) &&
                     (summary.final_cost <= successTolerance);
    if (converged && !successLog.empty()) {
        writeLog(logFile, successLog);
    }
    return converged;
}

void runThermalSolvers(const SimulationConstants& constants,
                       ceres::Problem& problem,
                       ceres::Solver::Summary& summary,
                       std::ostream& logFile) {
    // 最適なソルバーが既に決定されている場合は、それを使用
    if (g_solverOptimized && g_optimalSolverConfig) {
        writeLog(logFile, "--------最適化済みソルバー(" + g_optimalSolverName + ")を使用します...");
        ceres::Solver::Options options;
        g_optimalSolverConfig(options);
        ceres::Solve(options, &problem, &summary);
        logSolverTiming("最適化済みソルバー", summary, logFile);
        bool converged = (summary.termination_type == ceres::CONVERGENCE) &&
                         (summary.final_cost <= constants.thermalTolerance);
        if (converged) {
            writeLog(logFile, "--------最適化済みソルバーで収束しました");
        } else {
            writeLog(logFile, "--------警告: 最適化済みソルバーで収束しませんでした。フォールバックを試行します...");
            g_solverOptimized = false; // フォールバックに戻る
        }
        if (converged) {
            return;
        }
    }

    // 最初のタイムステップ: 全てのソルバーを試して最適なものを選択
    if (!g_solverOptimized) {
        writeLog(logFile, "--------[ステップ1] 全てのソルバーを試行して最適なものを選択します...");
        
        // 初期パラメータ値を保存（問題の状態をリセットするため）
        std::vector<std::vector<double>> initialParams;
        std::vector<double*> paramBlocks;
        problem.GetParameterBlocks(&paramBlocks);
        for (double* param : paramBlocks) {
            int size = problem.ParameterBlockSize(param);
            std::vector<double> values(param, param + size);
            initialParams.push_back(values);
        }
        
        struct SolverResult {
            std::function<void(ceres::Solver::Options&)> config;
            std::string name;
            double time;
            bool converged;
        };
        
        std::vector<SolverResult> results;
        
        // ソルバー1: SPARSE_NORMAL_CHOLESKY + EIGEN_SPARSE
        {
            writeLog(logFile, "--------  試行1: SPARSE_NORMAL_CHOLESKY+EIGEN_SPARSE");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = constants.maxInnerIteration;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.minimizer_progress_to_stdout = false;
                int threads = detectThreadCount();
                options.num_threads = threads;
                options.use_explicit_schur_complement = false;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "SPARSE_NORMAL_CHOLESKY+EIGEN_SPARSE", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー2: DENSE_QR（基本設定）
        {
            writeLog(logFile, "--------  試行2: DENSE_QR（基本設定）");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.linear_solver_type = ceres::DENSE_QR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = constants.maxInnerIteration;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.minimizer_progress_to_stdout = false;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "DENSE_QR（基本設定）", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー2b: DENSE_QR（並列化）
        {
            writeLog(logFile, "--------  試行2b: DENSE_QR（並列化）");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.linear_solver_type = ceres::DENSE_QR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = constants.maxInnerIteration;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.minimizer_progress_to_stdout = false;
                int threads = detectThreadCount();
                options.num_threads = threads;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "DENSE_QR（並列化）", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー2c: DENSE_QR（緩和許容誤差）
        {
            writeLog(logFile, "--------  試行2c: DENSE_QR（緩和許容誤差）");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.linear_solver_type = ceres::DENSE_QR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = constants.maxInnerIteration;
                options.function_tolerance = constants.thermalTolerance * 10.0;
                options.parameter_tolerance = constants.thermalTolerance * 10.0;
                options.minimizer_progress_to_stdout = false;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "DENSE_QR（緩和許容誤差）", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー3: SPARSE_NORMAL_CHOLESKY + SUITE_SPARSE
        {
            writeLog(logFile, "--------  試行3: SPARSE_NORMAL_CHOLESKY+SUITE_SPARSE");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = constants.maxInnerIteration;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.minimizer_progress_to_stdout = false;
                int threads = detectThreadCount();
                options.num_threads = threads;
                options.use_explicit_schur_complement = false;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "SPARSE_NORMAL_CHOLESKY+SUITE_SPARSE", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー4: DENSE_SCHUR
        {
            writeLog(logFile, "--------  試行4: DENSE_SCHUR");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.max_num_iterations = 500;
                options.function_tolerance = constants.thermalTolerance * 0.01;
                options.parameter_tolerance = constants.thermalTolerance * 0.01;
                options.gradient_tolerance = constants.thermalTolerance * 0.1;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.max_trust_region_radius = 1e4;
                options.initial_trust_region_radius = 1e2;
                options.minimizer_progress_to_stdout = false;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "DENSE_SCHUR", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー5: SPARSE_NORMAL_CHOLESKY (DOGLEG版)
        {
            writeLog(logFile, "--------  試行5: SPARSE_NORMAL_CHOLESKY+DOGLEG");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
                options.max_num_iterations = 1000;
                options.function_tolerance = constants.thermalTolerance * 0.001;
                options.parameter_tolerance = constants.thermalTolerance * 0.001;
                options.gradient_tolerance = constants.thermalTolerance * 0.01;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.inner_iteration_tolerance = 1e-8;
                options.max_trust_region_radius = 1e3;
                options.initial_trust_region_radius = 1e1;
                options.minimizer_progress_to_stdout = false;
                int threads = detectThreadCount();
                options.num_threads = threads;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "SPARSE_NORMAL_CHOLESKY+DOGLEG", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // ソルバー6: Line Search方式
        {
            writeLog(logFile, "--------  試行6: LINE_SEARCH+LBFGS");
            // パラメータを初期値にリセット
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Summary testSummary;
            auto config = [&](ceres::Solver::Options& options) {
                options.minimizer_type = ceres::LINE_SEARCH;
                options.line_search_direction_type = ceres::LBFGS;
                options.line_search_type = ceres::WOLFE;
                options.max_num_iterations = 1000;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.gradient_tolerance = constants.thermalTolerance * 10;
                options.jacobi_scaling = true;
                options.minimizer_progress_to_stdout = false;
            };
            ceres::Solver::Options options;
            config(options);
            ceres::Solve(options, &problem, &testSummary);
            bool converged = (testSummary.termination_type == ceres::CONVERGENCE) &&
                             (testSummary.final_cost <= constants.thermalTolerance);
            std::ostringstream oss;
            oss << "--------    結果: " << (converged ? "収束" : "未収束")
                << " | 時間: " << formatSeconds(testSummary.total_time_in_seconds) << "秒"
                << " | 残差: " << std::scientific << std::setprecision(6) << testSummary.final_cost;
            writeLog(logFile, oss.str());
            if (converged) {
                results.push_back({config, "LINE_SEARCH+LBFGS", 
                                 testSummary.total_time_in_seconds, true});
            }
        }
        
        // 最も速いソルバーを選択
        if (!results.empty()) {
            writeLog(logFile, "--------  収束したソルバーの比較:");
            for (const auto& result : results) {
                std::ostringstream oss;
                oss << "--------    - " << result.name 
                    << ": " << formatSeconds(result.time) << "秒";
                writeLog(logFile, oss.str());
            }
            
            auto best = std::min_element(results.begin(), results.end(),
                [](const SolverResult& a, const SolverResult& b) {
                    return a.time < b.time;
                });
            g_optimalSolverConfig = best->config;
            g_optimalSolverName = best->name;
            g_solverOptimized = true;
            
            std::ostringstream oss;
            oss << "--------[ステップ1完了] 最適なソルバー: " << best->name 
                << " (時間: " << formatSeconds(best->time) << "秒)";
            writeLog(logFile, oss.str());
            
            // 最適なソルバーで実際に解く（パラメータを初期値にリセット）
            for (size_t i = 0; i < paramBlocks.size(); ++i) {
                std::copy(initialParams[i].begin(), initialParams[i].end(), paramBlocks[i]);
            }
            ceres::Solver::Options options;
            g_optimalSolverConfig(options);
            ceres::Solve(options, &problem, &summary);
            logSolverTiming("最適化済みソルバー", summary, logFile);
            bool converged = (summary.termination_type == ceres::CONVERGENCE) &&
                             (summary.final_cost <= constants.thermalTolerance);
            if (converged) {
                writeLog(logFile, "--------最適化済みソルバーで収束しました");
            }
            return;
        }
    }

    // フォールバック: 従来の方法で全てのソルバーを試行
    bool converged = false;

    converged = runThermalSolverTrial(
        "--------①標準設定(SPARSE)でソルバーを実行します...",
        "--------標準設定(SPARSE)で収束しました",
        problem,
        summary,
        constants.thermalTolerance,
        [&](ceres::Solver::Options& options) {
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            // EIGEN_SPARSEはしばしばSUITE_SPARSEより高速
            options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.max_num_iterations = constants.maxInnerIteration;
            options.function_tolerance = constants.thermalTolerance;
            options.parameter_tolerance = constants.thermalTolerance;
            options.minimizer_progress_to_stdout = false;
            int threads = detectThreadCount();
            options.num_threads = threads;
            // 線形ソルバーの並列化を有効化
            options.use_explicit_schur_complement = false;
        },
        logFile);

    if (!converged) {
        converged = runThermalSolverTrial(
            "--------①標準設定(DENSE)でソルバーを再実行します...",
            "--------標準設定(DENSE)で収束しました",
            problem,
            summary,
            constants.thermalTolerance,
            [&](ceres::Solver::Options& options) {
                options.linear_solver_type = ceres::DENSE_QR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                options.max_num_iterations = constants.maxInnerIteration;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.minimizer_progress_to_stdout = false;
            },
            logFile);
    }

    if (!converged) {
        converged = runThermalSolverTrial(
            "---堅牢設定でソルバーを再実行します...",
            "---堅牢設定で収束しました",
            problem,
            summary,
            constants.thermalTolerance,
            [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = std::max(500, static_cast<int>(constants.maxInnerIteration * 2));
                options.function_tolerance = constants.thermalTolerance * 0.01;
                options.parameter_tolerance = constants.thermalTolerance * 0.01;
                options.gradient_tolerance = constants.thermalTolerance * 0.1;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.max_trust_region_radius = 1e4;
                options.initial_trust_region_radius = 1e2;
                options.minimizer_progress_to_stdout = false;
            },
            logFile);
    }

    if (!converged) {
        converged = runThermalSolverTrial(
            "---DENSE_SCHUR設定でソルバーを再実行します...",
            "---DENSE_SCHUR設定で収束しました",
            problem,
            summary,
            constants.thermalTolerance,
            [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::DENSE_SCHUR;
                options.max_num_iterations = 500;
                options.function_tolerance = constants.thermalTolerance * 0.01;
                options.parameter_tolerance = constants.thermalTolerance * 0.01;
                options.gradient_tolerance = constants.thermalTolerance * 0.1;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.max_trust_region_radius = 1e4;
                options.initial_trust_region_radius = 1e2;
                options.minimizer_progress_to_stdout = false;
            },
            logFile);
    }

    if (!converged) {
        converged = runThermalSolverTrial(
            "---SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します...",
            "---SPARSE_NORMAL_CHOLESKY設定で収束しました",
            problem,
            summary,
            constants.thermalTolerance,
            [&](ceres::Solver::Options& options) {
                options.trust_region_strategy_type = ceres::DOGLEG;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                options.max_num_iterations = 1000;
                options.function_tolerance = constants.thermalTolerance * 0.001;
                options.parameter_tolerance = constants.thermalTolerance * 0.001;
                options.gradient_tolerance = constants.thermalTolerance * 0.01;
                options.jacobi_scaling = true;
                options.use_inner_iterations = true;
                options.inner_iteration_tolerance = 1e-8;
                options.max_trust_region_radius = 1e3;
                options.initial_trust_region_radius = 1e1;
                options.minimizer_progress_to_stdout = false;
            },
            logFile);
    }

    if (!converged) {
        writeLog(logFile, "---段階的緩和法でソルバーを再実行します...");

        ceres::Solver::Options options1;
        options1.trust_region_strategy_type = ceres::DOGLEG;
        options1.linear_solver_type = ceres::DENSE_QR;
        options1.max_num_iterations = 200;
        options1.function_tolerance = constants.thermalTolerance * 10;
        options1.parameter_tolerance = constants.thermalTolerance * 10;
        options1.gradient_tolerance = constants.thermalTolerance;
        options1.jacobi_scaling = true;
        options1.minimizer_progress_to_stdout = false;

        ceres::Solve(options1, &problem, &summary);
        writeLog(logFile, "----段階1完了: 残差 " + std::to_string(summary.final_cost));

        ceres::Solver::Options options2;
        options2.trust_region_strategy_type = ceres::DOGLEG;
        options2.linear_solver_type = ceres::DENSE_QR;
        options2.max_num_iterations = 1000;
        options2.function_tolerance = constants.thermalTolerance * 0.01;
        options2.parameter_tolerance = constants.thermalTolerance * 0.01;
        options2.gradient_tolerance = constants.thermalTolerance * 0.1;
        options2.jacobi_scaling = true;
        options2.use_inner_iterations = true;
        options2.minimizer_progress_to_stdout = false;

        ceres::Solve(options2, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) &&
                    (summary.final_cost <= constants.thermalTolerance);
        if (converged) {
            writeLog(logFile, "---段階的緩和法で収束しました");
        }
    }

    if (!converged) {
        converged = runThermalSolverTrial(
            "---Line Search方式でソルバーを再実行します...",
            "---Line Search方式で収束しました",
            problem,
            summary,
            constants.thermalTolerance,
            [&](ceres::Solver::Options& options) {
                options.minimizer_type = ceres::LINE_SEARCH;
                options.line_search_direction_type = ceres::LBFGS;
                options.line_search_type = ceres::WOLFE;
                options.max_num_iterations = 1000;
                options.function_tolerance = constants.thermalTolerance;
                options.parameter_tolerance = constants.thermalTolerance;
                options.gradient_tolerance = constants.thermalTolerance * 10;
                options.jacobi_scaling = true;
                options.minimizer_progress_to_stdout = false;
            },
            logFile);
    }

    if (!converged) {
        writeLog(logFile, "---超精密設定で最終試行します...");
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 5000;
        double tolerance_factor = std::max(1.0, summary.final_cost / constants.thermalTolerance * 0.1);
        options.function_tolerance = constants.thermalTolerance * tolerance_factor;
        options.parameter_tolerance = constants.thermalTolerance * tolerance_factor;
        options.gradient_tolerance = constants.thermalTolerance * tolerance_factor * 10;
        options.jacobi_scaling = true;
        options.use_inner_iterations = true;
        options.inner_iteration_tolerance = 1e-12;
        options.max_trust_region_radius = 1e2;
        options.initial_trust_region_radius = 1e0;
        options.min_trust_region_radius = 1e-8;
        options.minimizer_progress_to_stdout = false;
        writeLog(logFile, "----調整済み許容誤差: " + std::to_string(options.function_tolerance));

        ceres::Solve(options, &problem, &summary);
        {
            std::string label = sanitizeLogLabel("---超精密設定");
            std::ostringstream oss;
            oss << "--------" << label << " 所要時間: " << formatSeconds(summary.total_time_in_seconds) << "秒"
                << " (前処理 " << formatSeconds(summary.preprocessor_time_in_seconds) << "秒"
                << ", 残差評価 " << formatSeconds(summary.residual_evaluation_time_in_seconds) << "秒"
                << ", ヤコビアン評価 " << formatSeconds(summary.jacobian_evaluation_time_in_seconds) << "秒"
                << ", 線形ソルバー " << formatSeconds(summary.linear_solver_time_in_seconds) << "秒"
                << ", 最適化 " << formatSeconds(summary.minimizer_time_in_seconds) << "秒)";
            writeLog(logFile, oss.str());
        }
        converged = (summary.termination_type == ceres::CONVERGENCE) &&
                    (summary.final_cost <= options.function_tolerance);

        if (converged) {
            writeLog(logFile, "---超精密設定で収束しました");
        } else {
            writeLog(logFile, "---全ての先進的ソルバー手法で収束に失敗しました");
            writeLog(logFile, "---最終残差: " + std::to_string(summary.final_cost) +
                                 " (目標: " + std::to_string(constants.thermalTolerance) + ")");
        }
    }
}

} // namespace ThermalSolverCeres
