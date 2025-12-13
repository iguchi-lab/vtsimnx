#pragma once

#include "../vtsim_solver.h"
#include "../../archenv/include/archenv.h"
#include <cmath>

// =============================================================================
// HeatCalculation - 熱流量計算の共通テンプレート関数群
// =============================================================================

namespace HeatCalculation {

// 移流熱流量計算（advection）
template <typename T>
T calcAdvectionHeat(const T& sourceTemp, const T& targetTemp, const EdgeProperties& edgeData) {
    double flowRate = edgeData.flow_rate; // m3/s
    if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) return T(0.0);
    
    double mDotCpAbs = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
    // 温度差ベース（有向）：source - target
    return T(mDotCpAbs) * (sourceTemp - targetTemp);
}

// 伝導熱流量計算（conductance）
template <typename T>
T calcConductionHeat(const T& sourceTemp, const T& targetTemp, const EdgeProperties& edgeData) {
    return T(edgeData.conductance) * (sourceTemp - targetTemp);
}

// 発熱量計算（heat_generation）
template <typename T>
T calcGenerationHeat(const EdgeProperties& edgeData) {
    return T(edgeData.current_heat_generation);
}

// 統一熱流量計算インターフェース（ブランチタイプに応じて適切な計算関数を呼び出す）
template <typename T>
T calculateUnifiedHeat(const T& sourceTemp, const T& targetTemp, const EdgeProperties& edgeData) {
    if (edgeData.type == "advection") {
        return calcAdvectionHeat(sourceTemp, targetTemp, edgeData);
    } else if (edgeData.type == "conductance") {
        return calcConductionHeat(sourceTemp, targetTemp, edgeData);
    } else if (edgeData.type == "heat_generation") {
        return calcGenerationHeat<T>(edgeData);
    } else {
        return T(0.0);
    }
}

} // namespace HeatCalculation


