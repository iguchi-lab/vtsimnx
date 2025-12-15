#pragma once

#include "../vtsim_solver.h"
#include <ostream>

// 前方宣言
class ThermalNetwork;

namespace ThermalSolverLinearGS {

// 熱計算（温度に関して線形を前提）を疎直接法で解く（LLT/LDLT/SparseLU）
void solveTemperaturesLinearGS(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    std::ostream& logFile);

} // namespace ThermalSolverLinearGS


