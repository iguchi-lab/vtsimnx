#pragma once

#include "../../vtsim_solver.h"
#include "core/ventilation/pressure_solve_result.h"

#include <iosfwd>
#include <memory>

// 前方宣言
class VentilationNetwork;

// 密度計算のヘルパー関数
double calculateDensity(double temperature);

// 圧力ソルバ（公開 API）。Ceres / fallback 実装詳細は Impl 側。
class PressureSolver {
public:
    using SolverResult = PressureSolveResult;

    PressureSolver(VentilationNetwork& network, std::ostream& logFile);
    ~PressureSolver();

    PressureSolver(const PressureSolver&) = delete;
    PressureSolver& operator=(const PressureSolver&) = delete;
    PressureSolver(PressureSolver&&) noexcept;
    PressureSolver& operator=(PressureSolver&&) noexcept;

    // Ceres Solver + LM法を使った圧力計算
    SolverResult solvePressures(const SimulationConstants& constants);

private:
    struct Impl;
    friend class PressureFallbackSolver;
    std::unique_ptr<Impl> impl_;
};
