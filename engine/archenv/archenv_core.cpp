#include "include/archenv.h"
#include <cmath>
#include <algorithm>
#include <sstream>

// バージョン情報マクロ定義
#define VERSION "1.0.0"
#define DESCRIPTION "C++ port of Python archenv library for building environmental engineering"
#define ORIGINAL_PYTHON_REPO "https://github.com/iguchi-lab/archenv"

namespace archenv {

// ====================================================================
// ライブラリ情報
// ====================================================================

std::string get_library_info() {
    std::ostringstream oss;
    oss << "====================================================\n";
    oss << " archenv - Building Environmental Engineering Library\n";
    oss << "====================================================\n";
    oss << "Version: " << VERSION << "\n";
    oss << "Description: " << DESCRIPTION << "\n";
    oss << "Original Python Repository: " << ORIGINAL_PYTHON_REPO << "\n";
    oss << "\nFeatures:\n";
    oss << "- Psychrometric calculations (air density, vapor pressure, humidity, enthalpy)\n";
    oss << "- PMV/PPD thermal comfort calculations\n";
    oss << "- Fungal growth index calculation\n";
    oss << "- JIS standard measurement conditions\n";
    oss << "- Unified physical constants with CRIEPI model integration\n";
    oss << "====================================================\n";
    return oss.str();
}

// ====================================================================
// 空気物性計算
// ====================================================================

double air_density(double temp_c) {
    // ρ = P / (R_air * T)
    return STANDARD_ATMOSPHERIC_PRESSURE / (GAS_CONSTANT_DRY_AIR * absolute_temperature(temp_c));
}

double air_heat_capacity(double volume_m3, double temp_c) {
    // C = V * c_p * ρ (単位系: m³ * J/(kg·K) * kg/m³ = J/K)
    return volume_m3 * SPECIFIC_HEAT_AIR * air_density(temp_c);
}

double air_specific_heat(double x) {
    // c_p,humid = c_p,air + c_p,vapor * x
    return SPECIFIC_HEAT_AIR + SPECIFIC_HEAT_WATER_VAPOR * x;
}

double vapor_latent_heat(double temp_c) {
    // L(T) = L₀ - 2.3668 * T (温度依存性)
    return LATENT_HEAT_VAPORIZATION - 2366.8 * temp_c;
}

// ====================================================================
// 水蒸気圧計算
// ====================================================================

namespace internal {
    double t_dash(double temp_c) {
        return absolute_temperature(100.0) / absolute_temperature(temp_c);
    }
}

double saturation_vapor_pressure(double temp_c) {
    // Wexler式による飽和水蒸気圧計算（Python版と同じアルゴリズム）
    double t_dash = internal::t_dash(temp_c);
    
    double log_ps = 
        -7.90298 * (t_dash - 1) +
        5.02808 * std::log10(t_dash) -
        1.3816e-7 * (std::pow(10, 11.344 * (1 - 1/t_dash)) - 1) +
        8.1328e-3 * (std::pow(10, -3.4919 * (t_dash - 1)) - 1) +
        std::log10(STANDARD_ATMOSPHERIC_PRESSURE / 100);
    
    return std::pow(10, log_ps) * 100; // hPa → Pa 変換
}

double vapor_pressure(double temp_c, double humidity) {
    return (humidity / 100.0) * saturation_vapor_pressure(temp_c);
}

double vapor_pressure_wet_bulb(double t_w, double t_d, double pressure) {
    // 湿球温度による水蒸気圧計算
    const double A = 0.00066; // 湿球係数 [1/K]
    double e_s_tw = saturation_vapor_pressure(t_w);
    return e_s_tw - A * pressure * (t_d - t_w);
}

// ====================================================================
// 絶対湿度計算
// ====================================================================

double absolute_humidity_from_vapor_pressure(double vapor_pressure) {
    // x = 0.622 * e / (P - e)
    const double capped_vp = std::clamp(
        vapor_pressure,
        0.0,
        STANDARD_ATMOSPHERIC_PRESSURE - TOLERANCE_MEDIUM);
    const double denominator = std::max(
        STANDARD_ATMOSPHERIC_PRESSURE - capped_vp,
        TOLERANCE_SMALL);
    return 0.622 * capped_vp / denominator;
}

double absolute_humidity(double temp_c, double humidity) {
    double e = vapor_pressure(temp_c, humidity);
    return absolute_humidity_from_vapor_pressure(e);
}

// ====================================================================
// エンタルピ計算
// ====================================================================

double sensible_enthalpy(double temp_c) {
    // h_s = c_p,air * T
    return SPECIFIC_HEAT_AIR * temp_c;
}

double latent_enthalpy_from_humidity(double temp_c, double humidity) {
    // h_l = x * (L + c_p,vapor * T) - Python版では0℃の蒸発潜熱を使用
    double x = absolute_humidity(temp_c, humidity);
    return x * (LATENT_HEAT_VAPORIZATION + SPECIFIC_HEAT_WATER_VAPOR * temp_c);
}

double latent_enthalpy_from_x(double temp_c, double x) {
    // h_l = x * (L + c_p,vapor * T)
    return x * (LATENT_HEAT_VAPORIZATION + SPECIFIC_HEAT_WATER_VAPOR * temp_c);
}

double total_enthalpy_from_humidity(double temp_c, double humidity) {
    return sensible_enthalpy(temp_c) + latent_enthalpy_from_humidity(temp_c, humidity);
}

double total_enthalpy_from_x(double temp_c, double x) {
    return sensible_enthalpy(temp_c) + latent_enthalpy_from_x(temp_c, x);
}

} // namespace archenv 