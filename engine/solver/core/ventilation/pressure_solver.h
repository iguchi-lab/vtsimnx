#pragma once

#include "../../vtsim_solver.h"
#include "core/ventilation/pressure_solve_result.h"

#include <iosfwd>
#include <memory>
#include <tuple>

// 前方宣言
class VentilationNetwork;

// 密度計算のヘルパー関数
double calculateDensity(double temperature);

// 圧力ソルバ（公開 API）。Ceres / fallback 実装詳細は Impl 側。
class PressureSolver {
public:
    // 既存互換: auto [pressures, flows, balances] = solver.solvePressures(...);
    using SolverResult = std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>;

    PressureSolver(VentilationNetwork& network, std::ostream& logFile);
    ~PressureSolver();

    PressureSolver(const PressureSolver&) = delete;
    PressureSolver& operator=(const PressureSolver&) = delete;
    PressureSolver(PressureSolver&&) noexcept;
    PressureSolver& operator=(PressureSolver&&) noexcept;

    // 詳細結果（accepted / metrics / method 付き）
    PressureSolveResult solveDetailed(const SimulationConstants& constants);

    // 3-tuple 互換ラッパ（一時オブジェクトからは maps を move）
    SolverResult solvePressures(const SimulationConstants& constants);

private:
    struct Impl;
    friend class PressureFallbackSolver;
    std::unique_ptr<Impl> impl_;
};
