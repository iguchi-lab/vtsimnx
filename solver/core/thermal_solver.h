#pragma once

#include "../vtsim_solver.h"
#include <ostream>

// 前方宣言
class ThermalNetwork;

// =============================================================================
// ThermalSolver - 温度・熱流量計算のメインソルバー
// =============================================================================
class ThermalSolver {
public:
    ThermalSolver(ThermalNetwork& network, std::ostream& logFile);

    void solveTemperatures(const SimulationConstants& constants);

private:
    ThermalNetwork& network_;
    std::ostream& logFile_;
}; 