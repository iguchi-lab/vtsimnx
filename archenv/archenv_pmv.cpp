#include "include/archenv.h"
#include <cmath>
#include <algorithm>

namespace archenv {

// ====================================================================
// PMV計算用内部関数
// ====================================================================

namespace internal {

    double calc_R(double f_cl, double t_cl, double t_r) {
        // 放射熱伝達: R = 3.96e-8 * f_cl * (T_cl⁴ - T_r⁴)
        double t_cl_k = t_cl + KELVIN_OFFSET;
        double t_r_k = t_r + KELVIN_OFFSET;
        return 3.96e-8 * f_cl * (std::pow(t_cl_k, 4) - std::pow(t_r_k, 4));
    }

    double calc_C(double f_cl, double h_c, double t_cl, double t_a) {
        // 対流熱伝達: C = f_cl * h_c * (T_cl - T_a)
        return f_cl * h_c * (t_cl - t_a);
    }

    double calc_RC(double f_cl, double h_c, double t_cl, double t_a, double t_r) {
        // 総熱伝達: RC = R + C
        return calc_R(f_cl, t_cl, t_r) + calc_C(f_cl, h_c, t_cl, t_a);
    }

} // namespace internal

// ====================================================================
// PMV/PPD計算関数
// ====================================================================

double calc_PMV(double met, double w, double clo, double t_a, double h_a, double t_r, double v_a) {
    // 基本パラメータ計算
    double M = met * 58.2;      // 代謝量 [W/m²]
    double I_cl = clo * 0.155;  // 着衣熱抵抗 [m²·K/W]

    // 着衣面積係数
    double f_cl = (I_cl < 0.078) ? (1.00 + 1.290 * I_cl) : (1.05 + 0.645 * I_cl);

    // 着衣表面温度の反復計算
    double t_cl = t_a;          // 初期値
    double omega = PMV_OMEGA_DEFAULT;
    double error = 1e12;

    int iterations = 0;
    while (std::abs(error) > PMV_TOLERANCE && iterations < PMV_MAX_ITERATIONS) {
        // 対流熱伝達率
        double h_c = std::max(2.38 * std::pow(std::abs(t_cl - t_a), 0.25), 
                              12.1 * std::sqrt(v_a));
        
        // 新しい着衣表面温度
        double new_t_cl = 35.7 - 0.028 * (M - w) - I_cl * internal::calc_RC(f_cl, h_c, t_cl, t_a, t_r);
        
        error = new_t_cl - t_cl;
        t_cl = t_cl + error * omega;
        iterations++;
    }

    // 最終的な対流熱伝達率
    double h_c = std::max(2.38 * std::pow(std::abs(t_cl - t_a), 0.25), 
                          12.1 * std::sqrt(v_a));

    // 水蒸気圧計算（Python版のe(t_a, h_a)関数相当）
    double e = archenv::vapor_pressure(t_a, h_a);

    // 各種熱損失計算
    double E_d = 3.05e-3 * (5733 - 6.99 * (M - w) - e);           // 拡散による潜熱損失
    double E_s = 0.42 * ((M - w) - 58.15);                        // 発汗による潜熱損失
    double E_re = 1.7e-5 * M * (5867 - e);                        // 呼吸による潜熱損失
    double C_re = 0.0014 * M * (34 - t_a);                        // 呼吸による顕熱損失

    // 熱負荷計算
    double L = (M - w) - E_d - E_s - E_re - C_re - internal::calc_RC(f_cl, h_c, t_cl, t_a, t_r);

    // PMV計算
    double PMV = (0.303 * std::exp(-0.036 * M) + 0.028) * L;

    return PMV;
}

double pmv_to_ppd(double pmv) {
    // PPD = 100 - 95 * exp(-0.03353 * PMV⁴ - 0.2179 * PMV²)
    return 100.0 - 95.0 * std::exp(-0.03353 * std::pow(pmv, 4) - 0.2179 * std::pow(pmv, 2));
}

double calc_PPD(double met, double w, double clo, double t_a, double h_a, double t_r, double v_a) {
    double pmv = calc_PMV(met, w, clo, t_a, h_a, t_r, v_a);
    return pmv_to_ppd(pmv);
}

} // namespace archenv 