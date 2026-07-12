#pragma once

#include "../../archenv/include/archenv.h"

#include <cmath>
#include <cstddef>
#include <vector>

// 湿り空気エンタルピー（換気移流・空気蓄熱）の薄いラッパ。
// 正本は archenv::total_enthalpy_from_x / air_specific_heat。

namespace thermal_moist_air {

inline double moistAirCp(double x) {
    return archenv::air_specific_heat(x);
}

inline double moistAirEnthalpy(double tempC, double x) {
    return archenv::total_enthalpy_from_x(tempC, x);
}

// 符号付き質量流量 [kg/s]（flowRate [m3/s], 乾き空気密度）
inline double massFlowKgPerS(double flowRateM3s) {
    return archenv::DENSITY_DRY_AIR * flowRateM3s;
}

// 診断: エンタルピー流量 H_dot = mDot * (h_src - h_dst) [W]
// flowRate の符号規約は calcAdvectionHeat と同じ（source→target 正）。
inline double advectionEnthalpyFluxW(double tSrc,
                                    double xSrc,
                                    double tDst,
                                    double xDst,
                                    double flowRateM3s) {
    if (std::abs(flowRateM3s) < archenv::FLOW_RATE_MIN) return 0.0;
    const double mDot = massFlowKgPerS(flowRateM3s);
    return mDot * (moistAirEnthalpy(tSrc, xSrc) - moistAirEnthalpy(tDst, xDst));
}

// 診断: 現行顕熱 Q_sens = ρ cp_dry Q (T_src - T_dst)
inline double advectionSensibleFluxW(double tSrc, double tDst, double flowRateM3s) {
    if (std::abs(flowRateM3s) < archenv::FLOW_RATE_MIN) return 0.0;
    const double mDotCp =
        archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRateM3s;
    return mDotCp * (tSrc - tDst);
}

// capacity 枝の conductance = thermal_mass/dt から、x=0 で顕熱と一致する ρV/Δt を復元
inline double rhoVOverDtFromCapacityConductance(double conductance) {
    return conductance / archenv::SPECIFIC_HEAT_AIR;
}

// DirectT 組立へ渡すコンテキスト（TopologyCache が保持）
struct MoistAssembleContext {
    bool enabled = false;
    // タイムステップ初期湿度 x_n（頂点インデックス）。未設定時は current_x を x_n とする。
    const std::vector<double>* humidityXnByVertex = nullptr;
};

} // namespace thermal_moist_air
