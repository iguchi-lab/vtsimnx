#pragma once

#include "core/ventilation/pressure_solver_impl.h"

#include <optional>

// Fallback 経路（partition / StageA / mutation / StageB / decision）の入口。
// Ceres 依存は実装側に閉じ、公開 PressureSolver ヘッダには出さない。
class PressureFallbackSolver {
public:
    explicit PressureFallbackSolver(PressureSolver::Impl& owner);

    std::optional<PressureSolveResult> run(
        const SimulationConstants& constants,
        PressureSolver::Impl::SolverSetup& setup,
        ceres::Solver::Summary& summary);

private:
    PressureSolver::Impl& owner_;
};
