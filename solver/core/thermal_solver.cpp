#include "core/thermal_solver.h"
#include "core/thermal_solver_linear_gs.h"

ThermalSolver::ThermalSolver(ThermalNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

// =============================================================================
// メイン温度計算
// =============================================================================

void ThermalSolver::solveTemperatures(const SimulationConstants& constants) {
    // 熱計算は温度に関して線形を前提に疎直接法で解く（フォールバックなし）
    ThermalSolverLinearGS::solveTemperaturesLinearGS(network_, constants, logFile_);
}