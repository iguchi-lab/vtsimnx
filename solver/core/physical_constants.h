#pragma once

#include "../vtsim_solver.h"

// physical_constants.hはPhysicalConstants名前空間を再エクスポートする
// PhysicalConstantsはvtsim_solver.hで定義されている

// 追加の物理定数が必要な場合はここに定義
namespace PhysicalConstants {
    // 小さい値の許容誤差（flow_calculation.hで使用）
    constexpr double TOLERANCE_SMALL = 1e-10;
    constexpr double TOLERANCE_MEDIUM = 1e-6;
    
    // 空気密度（flow_calculation.hで使用）[kg/m³]
    constexpr double RHO_AIR = 1.2;
    
    // 重力加速度 [m/s²]
    constexpr double GRAVITY = 9.80665;
    
    // 空気の比熱 [J/(kg·K)]
    constexpr double C_AIR = 1005.0;
    
    // 最小流量 [kg/s]
    constexpr double FLOW_RATE_MIN = 1e-10;
    
    // 標準大気圧 [Pa]
    constexpr double STANDARD_ATMOSPHERIC_PRESSURE = 101325.0;
    
    // 乾燥空気の気体定数 [J/(kg·K)]
    constexpr double GAS_CONSTANT_DRY_AIR = 287.055;
}

