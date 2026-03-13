#pragma once

#include "../../vtsim_solver.h"
#include "../../archenv/include/archenv.h"
#include <cmath>

// =============================================================================
// HeatCalculation - 熱流量計算の共通関数群
// =============================================================================

namespace HeatCalculation {

// 移流熱流量計算（advection）
inline double calcAdvectionHeat(double sourceTemp, double targetTemp, const EdgeProperties& edgeData) {
    double flowRate = edgeData.flow_rate; // m3/s
    if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) return 0.0;
    
    double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
    // 符号付き風量に基づいた熱移動量（source -> target 向きを正とする）
    // flowRate > 0 のとき: mDotCpAbs * (sourceTemp - targetTemp)
    // flowRate < 0 のとき: mDotCpAbs * (targetTemp - sourceTemp)
    return mDotCp * (sourceTemp - targetTemp);
}

// 伝導熱流量計算（conductance）
inline double calcConductionHeat(double sourceTemp, double targetTemp, const EdgeProperties& edgeData) {
    return edgeData.conductance * (sourceTemp - targetTemp);
}

// 発熱量計算（heat_generation）
inline double calcGenerationHeat(const EdgeProperties& edgeData) {
    return edgeData.current_heat_generation;
}

// 統一熱流量計算インターフェース（ブランチタイプに応じて適切な計算関数を呼び出す）
inline double calculateUnifiedHeat(double sourceTemp, double targetTemp, const EdgeProperties& edgeData) {
    switch (edgeData.getTypeCode()) {
        case EdgeProperties::TypeCode::Advection:
            return calcAdvectionHeat(sourceTemp, targetTemp, edgeData);
        case EdgeProperties::TypeCode::Conductance:
            return calcConductionHeat(sourceTemp, targetTemp, edgeData);
        case EdgeProperties::TypeCode::HeatGeneration:
            return calcGenerationHeat(edgeData);
        default:
            return 0.0;
    }
}

} // namespace HeatCalculation


