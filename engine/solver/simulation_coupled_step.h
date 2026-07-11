#pragma once

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

#include <ostream>
#include <string>

class VentilationNetwork;
class ThermalNetwork;

// 連成計算（pressure/thermal）1回分の「確定データ」
struct CoupledStepData {
    PressureMap pressureMap;
    FlowRateMap flowRates;
    FlowBalanceMap flowBalance;
};

// 換気・熱計算の「1回分」を実行する（内側連成反復の制御は呼び出し側）
CoupledStepData performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                                              ThermalNetwork& thermalNetwork,
                                              const SimulationConstants& constants,
                                              std::ostream& logs,
                                              int& totalIterations,
                                              TimingList& timings,
                                              const std::string& meta);
