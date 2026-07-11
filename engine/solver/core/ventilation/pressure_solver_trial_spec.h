#pragma once

#include <ceres/ceres.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "../../types/common_types.h"

namespace ventilation {

// Primary / Stage A / Stage B で共有する Ceres trial 設定。
struct SolverTrialSpec {
    const char* startLog = "";
    const char* successLog = "";
    std::function<void(ceres::Solver::Options&, const SimulationConstants&)> configure;
};

inline void configureStandardLmQr(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.linear_solver_type = ceres::DENSE_QR;
    o.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    o.max_num_iterations = static_cast<int>(c.maxInnerIterations);
    o.function_tolerance = c.ventilationTolerance;
    o.parameter_tolerance = c.ventilationTolerance;
    o.minimizer_progress_to_stdout = false;
}

inline void configureRobustDoglegQr(ceres::Solver::Options& o, const SimulationConstants& c) {
    o.trust_region_strategy_type = ceres::DOGLEG;
    o.linear_solver_type = ceres::DENSE_QR;
    o.max_num_iterations = std::max(500, static_cast<int>(c.maxInnerIterations * 2));
    o.function_tolerance = c.ventilationTolerance * 0.01;
    o.parameter_tolerance = c.ventilationTolerance * 0.01;
    o.gradient_tolerance = c.ventilationTolerance * 0.1;
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
    o.function_tolerance = c.ventilationTolerance * 0.01;
    o.parameter_tolerance = c.ventilationTolerance * 0.01;
    o.gradient_tolerance = c.ventilationTolerance * 0.1;
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
    o.function_tolerance = c.ventilationTolerance * 0.001;
    o.parameter_tolerance = c.ventilationTolerance * 0.001;
    o.gradient_tolerance = c.ventilationTolerance * 0.01;
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
    o.function_tolerance = c.ventilationTolerance;
    o.parameter_tolerance = c.ventilationTolerance;
    o.gradient_tolerance = c.ventilationTolerance * 10;
    o.jacobi_scaling = true;
    o.minimizer_progress_to_stdout = false;
}

inline const std::vector<SolverTrialSpec>& primaryTrustRegionTrials() {
    static const std::vector<SolverTrialSpec> trials = {
        {"----①標準設定でソルバーを実行します...", "----標準設定で収束しました",
         configureStandardLmQr},
        {"----②堅牢設定でソルバーを再実行します...", "----堅牢設定で収束しました",
         configureRobustDoglegQr},
        {"----③DENSE_SCHUR設定でソルバーを再実行します...", "----DENSE_SCHUR設定で収束しました",
         configureDoglegDenseSchur},
        {"----④SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します...",
         "----SPARSE_NORMAL_CHOLESKY設定で収束しました",
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
