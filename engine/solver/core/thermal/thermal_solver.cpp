#include "core/thermal/thermal_solver.h"
#include "core/thermal/thermal_solver_linear_direct.h"
#include "network/thermal_network.h"
#include "utils/utils.h"

ThermalSolver::ThermalSolver(ThermalNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

// =============================================================================
// メイン温度計算
// =============================================================================

void ThermalSolver::solveTemperatures(const SimulationConstants& constants) {
    // 熱計算は温度に関して線形を前提に疎直接法で解く
    // デフォルト: 絶対温度 AT=b を直接解く（高速）
    // 互換性より単純さを優先し、フォールバック実装は持たない（失敗時は状態だけ保持して戻る）
    try {
        ThermalSolverLinearDirect::solveTemperaturesLinearDirect(network_, constants, logFile_);
    } catch (const std::exception& e) {
        writeLog(logFile_, std::string("----警告: DirectT solver failed: ") + e.what());
        // 直近の状態を上位が判断できるように保持する（温度は更新されない）
        network_.setLastThermalConvergence(false, /*rmse*/0.0, /*max*/0.0, "DirectT(failed)");
    }
}