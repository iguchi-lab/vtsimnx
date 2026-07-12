#pragma once

#include <ceres/ceres.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "../../types/common_types.h"
#include "core/ventilation/pressure_balance.h"

namespace ventilation {

// Primary / Stage A / Stage B で共有する Ceres trial 設定。
struct SolverTrialSpec {
    const char* startLog = "";
    const char* successLog = "";
    std::function<void(ceres::Solver::Options&, const SimulationConstants&)> configure;
};

// Ceres 停止条件は相対値。ventilationTolerance（体積流量収支 [m³/s]）とは分離する。
inline void applyCeresStopTolerances(ceres::Solver::Options& o,
                                     const SimulationConstants& c,
                                     double functionScale = 1.0,
                                     double parameterScale = 1.0,
                                     double gradientScale = 1.0) {
    const auto t = makePressureSolverTolerances(c);
    o.function_tolerance = t.ceresFunctionRelative * functionScale;
    o.parameter_tolerance = t.ceresParameter * parameterScale;
    o.gradient_tolerance = t.ceresGradient * gradientScale;
}

inline void configureStandardLmQr(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.linear_solver_type = ceres::DENSE_QR;
    o.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    o.max_num_iterations = static_cast<int>(c.maxInnerIterations);
    applyCeresStopTolerances(o, c);
    o.minimizer_progress_to_stdout = false;
}

inline void configureRobustDoglegQr(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.trust_region_strategy_type = ceres::DOGLEG;
    o.linear_solver_type = ceres::DENSE_QR;
    o.max_num_iterations = std::max(500, static_cast<int>(c.maxInnerIterations * 2));
    // 堅牢試行は反復・トラスト領域を広げるが、停止条件の次元は Ceres 相対のまま。
    applyCeresStopTolerances(o, c, /*functionScale=*/10.0, /*parameterScale=*/10.0,
                             /*gradientScale=*/10.0);
    o.jacobi_scaling = true;
    o.use_inner_iterations = true;
    o.max_trust_region_radius = 1e4;
    o.initial_trust_region_radius = 1e2;
    o.minimizer_progress_to_stdout = false;
}

inline void configureDoglegDenseSchur(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.trust_region_strategy_type = ceres::DOGLEG;
    o.linear_solver_type = ceres::DENSE_SCHUR;
    o.max_num_iterations = 500;
    applyCeresStopTolerances(o, c, 10.0, 10.0, 10.0);
    o.jacobi_scaling = true;
    o.use_inner_iterations = true;
    o.max_trust_region_radius = 1e4;
    o.initial_trust_region_radius = 1e2;
    o.minimizer_progress_to_stdout = false;
}

inline void configureDoglegSparseCholesky(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.trust_region_strategy_type = ceres::DOGLEG;
    o.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    o.max_num_iterations = 1000;
    applyCeresStopTolerances(o, c, 100.0, 100.0, 100.0);
    o.jacobi_scaling = true;
    o.use_inner_iterations = true;
    o.inner_iteration_tolerance = 1e-8;
    o.max_trust_region_radius = 1e3;
    o.initial_trust_region_radius = 1e1;
    o.minimizer_progress_to_stdout = false;
}

inline void configureLineSearchLbfgs(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.minimizer_type = ceres::LINE_SEARCH;
    o.line_search_direction_type = ceres::LBFGS;
    o.line_search_type = ceres::WOLFE;
    o.max_num_iterations = 1000;
    applyCeresStopTolerances(o, c, 1.0, 1.0, 10.0);
    o.jacobi_scaling = true;
    o.minimizer_progress_to_stdout = false;
}

inline const std::vector<SolverTrialSpec>& primaryTrustRegionTrials() {
    static const std::vector<SolverTrialSpec> trials = {
        {"[圧力] ①標準設定でソルバーを実行します...",
         "[圧力] 標準設定: Ceres相対停止 (CONVERGENCE)。物理収支は別判定",
         configureStandardLmQr},
        {"[圧力] ②堅牢設定でソルバーを再実行します...",
         "[圧力] 堅牢設定: Ceres相対停止 (CONVERGENCE)。物理収支は別判定",
         configureRobustDoglegQr},
        {"[圧力] ③DENSE_SCHUR設定でソルバーを再実行します...",
         "[圧力] DENSE_SCHUR設定: Ceres相対停止 (CONVERGENCE)。物理収支は別判定",
         configureDoglegDenseSchur},
        {"[圧力] ④SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します...",
         "[圧力] SPARSE_NORMAL_CHOLESKY設定: Ceres相対停止 (CONVERGENCE)。物理収支は別判定",
         configureDoglegSparseCholesky},
    };
    return trials;
}

inline const std::vector<SolverTrialSpec>& stageATrustRegionTrials() {
    static const std::vector<SolverTrialSpec> trials = {
        {"[A-①] 標準設定でソルバーを実行します", "", configureStandardLmQr},
        {"[A-②] 堅牢設定でソルバーを再実行します", "", configureRobustDoglegQr},
        {"[A-③] DENSE_SCHUR設定でソルバーを再実行します", "", configureDoglegDenseSchur},
        {"[A-④] SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します", "",
         configureDoglegSparseCholesky},
    };
    return trials;
}

inline const std::vector<SolverTrialSpec>& stageBTrustRegionTrials() {
    static const std::vector<SolverTrialSpec> trials = {
        {"[B-①] 標準設定でソルバーを実行します", "[B-①] 収束 | residual=", configureStandardLmQr},
        {"[B-②] 堅牢設定でソルバーを再実行します", "[B-②] 収束 | residual=", configureRobustDoglegQr},
        {"[B-③] DENSE_SCHUR設定でソルバーを再実行します", "[B-③] 収束 | residual=",
         configureDoglegDenseSchur},
        {"[B-④] SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します", "[B-④] 収束 | residual=",
         configureDoglegSparseCholesky},
    };
    return trials;
}

} // namespace ventilation
