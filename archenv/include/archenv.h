#pragma once

#include <string>

namespace archenv {

// ====================================================================
// 物理定数
// ====================================================================

// 標準大気圧 [Pa]
constexpr double STANDARD_ATMOSPHERIC_PRESSURE = 101325.0;

// 乾燥空気の気体定数 [J/(kg·K)]
constexpr double GAS_CONSTANT_DRY_AIR = 287.055;

// 空気の比熱 [J/(kg·K)]
constexpr double SPECIFIC_HEAT_AIR = 1006.0;

// 水蒸気の定圧比熱 [J/(kg·K)]
constexpr double SPECIFIC_HEAT_WATER_VAPOR = 1846.0;

// 蒸発潜熱 [J/kg] (0℃基準)
constexpr double LATENT_HEAT_VAPORIZATION = 2500800.0;

// 乾燥空気密度 [kg/m³] (標準状態)
constexpr double DENSITY_DRY_AIR = 1.2;

// ケルビンオフセット
constexpr double KELVIN_OFFSET = 273.15;

// 熱交換器面積定数
constexpr double A_F_HEX_SMALL = 0.2;
constexpr double A_E_HEX_SMALL = 6.2;
constexpr double A_F_HEX_LARGE = 0.3;
constexpr double A_E_HEX_LARGE = 10.6;

// ====================================================================
// ヘルパー関数
// ====================================================================

inline double absolute_temperature(double temp_c) {
    return temp_c + KELVIN_OFFSET;
}

// ====================================================================
// 空気物性計算関数
// ====================================================================

double air_density(double temp_c);
double air_heat_capacity(double volume_m3, double temp_c);
double air_specific_heat(double x);
double vapor_latent_heat(double temp_c);

// ====================================================================
// 水蒸気圧計算関数
// ====================================================================

double saturation_vapor_pressure(double temp_c);
double vapor_pressure(double temp_c, double humidity);
double vapor_pressure_wet_bulb(double t_w, double t_d, double pressure);

// ====================================================================
// 絶対湿度計算関数
// ====================================================================

double absolute_humidity_from_vapor_pressure(double vapor_pressure);
double absolute_humidity(double temp_c, double humidity);

// ====================================================================
// エンタルピ計算関数
// ====================================================================

double sensible_enthalpy(double temp_c);
double latent_enthalpy_from_humidity(double temp_c, double humidity);
double latent_enthalpy_from_x(double temp_c, double x);
double total_enthalpy_from_humidity(double temp_c, double humidity);
double total_enthalpy_from_x(double temp_c, double x);

// ====================================================================
// PMV/PPD計算関数
// ====================================================================

double calc_PMV(double met, double w, double clo, double t_a, double h_a, double t_r, double v_a);
double pmv_to_ppd(double pmv);
double calc_PPD(double met, double w, double clo, double t_a, double h_a, double t_r, double v_a);

// ====================================================================
// 真菌指数計算関数
// ====================================================================

double calc_fungal_index(double h, double t);

// ====================================================================
// JIS規格条件
// ====================================================================

namespace jis {
    // 冷房時条件
    constexpr double T_C_IN = 27.0;      // 室内温度 [℃]
    constexpr double T_WB_C_IN = 19.5;   // 室内湿球温度 [℃]
    constexpr double T_C_EX = 35.0;      // 外気温度 [℃]
    constexpr double T_WB_C_EX = 27.0;   // 外気湿球温度 [℃]
    constexpr double X_C_IN = 0.0105;    // 室内絶対湿度 [kg/kg']
    constexpr double X_C_EX = 0.0205;    // 外気絶対湿度 [kg/kg']
    
    // 暖房時条件
    constexpr double T_H_IN = 20.0;      // 室内温度 [℃]
    constexpr double T_WB_H_IN = 14.0;   // 室内湿球温度 [℃]
    constexpr double T_H_EX = 7.0;       // 外気温度 [℃]
    constexpr double T_WB_H_EX = 6.0;    // 外気湿球温度 [℃]
    constexpr double X_H_IN = 0.0076;    // 室内絶対湿度 [kg/kg']
    constexpr double X_H_EX = 0.0052;    // 外気絶対湿度 [kg/kg']
    
    // ヘルパー関数
    inline double calc_x_from_wet_bulb(double t_wb, double t_d) {
        // 湿球温度から絶対湿度を計算
        double e_wb = vapor_pressure_wet_bulb(t_wb, t_d, STANDARD_ATMOSPHERIC_PRESSURE);
        return absolute_humidity_from_vapor_pressure(e_wb);
    }
    std::string validate_jis_conditions();
}

// ====================================================================
// ライブラリ情報
// ====================================================================

std::string get_library_info();

} // namespace archenv

