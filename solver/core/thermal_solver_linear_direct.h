#pragma once

#include "../vtsim_solver.h"
#include <ostream>

// 前方宣言
class ThermalNetwork;

namespace ThermalSolverLinearDirect {

// 熱計算（温度に関して線形を前提）を「絶対温度」AT=b の形で疎直接法で解く（LLT/LDLT/SparseLU）
void solveTemperaturesLinearDirect(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    std::ostream& logFile);

} // namespace ThermalSolverLinearDirect


