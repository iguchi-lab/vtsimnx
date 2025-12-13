#pragma once

#include "../vtsim_solver.h"
#include <tuple>
#include <ostream>

// 前方宣言
class VentilationNetwork;

namespace PressureSolverNewtonGS {

// Newton反復の線形サブ問題を Gauss-Seidel+SOR で解く
std::tuple<PressureMap, FlowRateMap, FlowBalanceMap> solvePressuresNewtonGS(
    VentilationNetwork& network,
    const SimulationConstants& constants,
    double omega,
    std::ostream& logFile);

} // namespace PressureSolverNewtonGS


