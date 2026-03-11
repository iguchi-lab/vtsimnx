#pragma once

#include "types/common_types.h"
#include "vtsimnx_solver_timing.h"

class VentilationNetwork;
class ThermalNetwork;

namespace transport {

// NOTE:
// core::humidity が正規の実装レイヤ。
// この transport API は既存呼び出し互換のために維持し、内部で core 実装へ委譲する。

// 湿度（絶対湿度 x）を 1 タイムステップ分だけ更新する。
// - constants.humidityCalc=false の場合は何もしない。
// - flowRates は換気計算（または fixed_flow）で確定したものを渡す。
void updateHumidityIfEnabled(const SimulationConstants& constants,
                             VentilationNetwork& ventNetwork,
                             ThermalNetwork& thermalNetwork,
                             const FlowRateMap& flowRates,
                             std::ostream& logs,
                             TimingList& timings,
                             const std::string& meta);

} // namespace transport


