#include "core/thermal_solver.h"
#include "core/thermal_solver_linear_gs.h"
#include "core/thermal_solver_linear_direct.h"
#include "utils/utils.h"

ThermalSolver::ThermalSolver(ThermalNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

// =============================================================================
// メイン温度計算
// =============================================================================

void ThermalSolver::solveTemperatures(const SimulationConstants& constants) {
    // 熱計算は温度に関して線形を前提に疎直接法で解く
    // デフォルト: 絶対温度 AT=b を直接解く（高速）
    // 念のため、失敗時は旧実装（ΔT の直接法）へフォールバックする
    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(network_, constants, logFile_);
    } catch (const std::exception& e) {
        writeLog(logFile_, std::string("----警告: DirectT solver failed, fallback to GS impl: ") + e.what());
        ThermalSolverLinearGS::solveTemperaturesLinearGS(network_, constants, logFile_);
    }
}