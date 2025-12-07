#pragma once

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <tuple>
#include <ostream>

class VentilationNetwork;
class ThermalNetwork;
class AirconController;

// 圧力変化量を計算
double calculatePressureChange(const PressureMap& oldPressures, const PressureMap& newPressures);

// 温度変化量を計算
double calculateTemperatureChange(const TemperatureMap& oldTemperatures, const TemperatureMap& newTemperatures);

// 換気・熱計算の連成を行う関数
std::tuple<PressureMap, FlowRateMap, FlowBalanceMap,
           TemperatureMap, HeatRateMap, HeatBalanceMap>
performCoupledCalculation(VentilationNetwork& ventNetwork,
                          ThermalNetwork& thermalNetwork,
                          const SimulationConstants& constants,
                          std::ostream& logs,
                          int& totalIterations,
                          TimingList& timings,
                          const std::string& meta);

// シミュレーションのタイムステップループを実行
void runSimulation(VentilationNetwork& ventNetwork,
                   ThermalNetwork& thermalNetwork,
                   AirconController& airconController,
                   const SimulationConstants& constants,
                   SimulationResults& results,
                   std::ostream& logs,
                   TimingList& timings,
                   const std::string& meta);


