#pragma once

#include "../vtsim_solver.h"
#include <ceres/ceres.h>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

// 前方宣言
class ThermalNetwork;

#include "core/thermal_constraints.h"

// =============================================================================
// ThermalSolver - 温度・熱流量計算のメインソルバー
// =============================================================================
class ThermalSolver {
public:
    ThermalSolver(ThermalNetwork& network, std::ostream& logFile);

    std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> solveTemperatures(
        const SimulationConstants& constants);

    // システム診断情報の出力
    void outputThermalSystemDiagnostics();

private:
    ThermalNetwork& network_;
    std::ostream& logFile_;

    void setInitialTemperatures(std::vector<double>& temperatures, const std::vector<std::string>& nodeNames);
    
    TemperatureMap extractTemperatures(const std::vector<double>& temperatures, const std::vector<std::string>& nodeNames);
    
    HeatRateMap calculateHeatRates(const TemperatureMap& tempMap);
    
    HeatBalanceMap verifyBalance(const HeatRateMap& heatRates);
}; 