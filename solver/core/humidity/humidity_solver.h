#pragma once

#include "types/common_types.h"
#include "vtsimnx_solver_timing.h"

class VentilationNetwork;
class ThermalNetwork;

namespace core::humidity {

// 湿度（絶対湿度 x）を 1 タイムステップ分だけ更新する。
// - constants.humidityCalc=false の場合は何もしない。
// - flowRates は現状未使用（換気グラフの枝流量を直接参照）だが、API互換のため受け取る。
void updateHumidityIfEnabled(const SimulationConstants& constants,
                             VentilationNetwork& ventNetwork,
                             ThermalNetwork& thermalNetwork,
                             const FlowRateMap& flowRates,
                             std::ostream& logs,
                             TimingList& timings,
                             const std::string& meta);

} // namespace core::humidity

