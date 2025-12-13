#pragma once

#include "../vtsim_solver.h"
#include <vector>
#include <unordered_map>
#include <map>
#include <ostream>
#include <tuple>

// 前方宣言
class ThermalNetwork;

namespace ThermalSolverNewtonGS {

// Newton反復の線形サブ問題を Gauss-Seidel+SOR で解く
// omega=1.0 で標準Gauss-Seidel、1.0<omega<2.0 で過緩和
std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> solveTemperaturesNewtonGS(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    double omega,
    std::ostream& logFile);

} // namespace ThermalSolverNewtonGS


