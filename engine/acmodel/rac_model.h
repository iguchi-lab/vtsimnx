#pragma once
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "acmodel.h"
#include "aircon_constants.h"
#include "../../archenv/include/archenv.h"  // archenvライブラリ
namespace ae = archenv;

namespace acmodel {

/**
 * RACモデル（Python版のCRIEPI系実装を忠実移植）
 * - 冷房: 顕熱/潜熱分離、SHF最小で潜熱上限制約、部分負荷電力は f_C_Theta
 * - 暖房: 着霜補正 C_df_H(Theta, RH)、部分負荷電力は f_H_Theta
 * - 単位: q_* は W（瞬時能力）、L_* / Q_* は MJ/h（Python準拠）
 * - dualcompressor 切替対応（spec["dualcompressor"]）
 */
class RACModel : public AirconSpec {
public:
    explicit RACModel(const nlohmann::json& config);
    ~RACModel();

    COPResult estimateCOP(const std::string& mode, const InputData& input);
    COPResult estimateCoolingCOP(const InputData& input);
    COPResult estimateHeatingCOP(const InputData& input);

    std::string    getModelName() const { return "RAC"; }
    nlohmann::json getModelParameters() const;

    const std::vector<std::string>& preparationLogs() const { return preparationLogs_; }

private:
    // ===== Python版と同じテーブル =====
    static constexpr double TABLE_3[5][6] = {
        {-0.00236,  0.01324,  0.08418, -0.47143, -1.16944,   6.54886},
        { 0.00427, -0.02392, -0.19226,  0.94213,  2.58632, -12.85618},
        {-0.00275,  0.01542,  0.14947, -0.68303, -2.03594,  10.60561},
        { 0.00063, -0.00351, -0.02865,  0.10522,  0.37336,  -1.09499},
        {-0.00005,  0.00028,  0.00184, -0.01090, -0.09609,   0.59229}
    };
    static constexpr double TABLE_4_A[5][3] = {
        {-0.000056,  0.000786,  0.071625},
        {-0.000145,  0.003337, -0.143643},
        {-0.000240, -0.029471,  1.954343},
        {-0.000035, -0.050909,  1.389751},
        { 0.0,       0.0,       0.076800}
    };
    static constexpr double TABLE_4_B[5][3] = {
        { 0.000108, -0.035658,  3.063873},
        {-0.000017,  0.062546, -5.471556},
        {-0.000245, -0.025126,  4.057590},
        { 0.000323, -0.021166,  0.575459},
        { 0.0,       0.000330,  0.047500}
    };
    static constexpr double TABLE_4_C[5][3] = {
        {-0.001465, -0.030500,  1.920431},
        { 0.002824,  0.041081, -1.835302},
        {-0.001929, -0.009738,  1.582898},
        { 0.000616, -0.014239,  0.546204},
        { 0.0,      -0.000110,  0.023100}
    };
    static constexpr double TABLE_5[5][6] = {
        {0.00000, 0.00000,  0.00000,  0.00000,  0.00000,  0.00000},
        {0.00000, 0.00000, -0.00036,  0.05080, -0.20346,  0.47765},
        {0.00000, 0.00000,  0.00227, -0.03952,  0.04115,  0.23099},
        {0.00000, 0.00000, -0.00911,  0.07102,  0.14950, -1.07335},
        {0.00000, 0.00000,  0.00044, -0.00214, -0.06250,  0.35150}
    };
    static constexpr double TABLE_6_A[5][3] = {
        {-0.0004078,  0.01035, -0.03248},
        { 0.0,        0.04099, -0.818809},
        { 0.0,       -0.04615,  2.10666},
        { 0.0013382, -0.01179, -0.41778},
        { 0.0000000, -0.00102,  0.09270}
    };
    static constexpr double TABLE_6_B[5][3] = {
        {-0.000056, -0.003539, -0.430566},
        { 0.0,       0.015237,  1.188850},
        { 0.0,       0.000527, -0.304645},
        {-0.000179,  0.020543,  0.130373},
        { 0.0,       0.000240,  0.013500}
    };
    static constexpr double TABLE_6_C[5][3] = {
        {-0.0001598,  0.004848,  0.047097},
        { 0.0,        0.016675,  0.362141},
        { 0.0,       -0.008134, -0.023535},
        {-0.0000772,  0.012558,  0.056185},
        { 0.0,       -0.000110,  0.010300}
    };

    // ===== Pythonの定数 =====
    static constexpr double C_AF_H = 0.8;
    static constexpr double C_AF_C = 0.85;
    static constexpr double C_HM_C = 1.15;
    static constexpr double SHF_L_MIN_C = 0.4;

    // ===== p_i（Python eq8/9/23/24） =====
    static inline int idx_row (int i) { return 4 - (i / 10); }
    static inline int idx_col2(int i) { return (2 - (i % 10)) * 2; }
    static inline int idx_col1(int i) { return (2 - (i % 10)); }
    static inline double lerp(double a, double b, double t) { return a*(1.0 - t) + b*t; }

    double calc_p_i_eq8 (int i, double q_rtd_C) const;
    double calc_p_i_eq9 (int i, double q_rtd_C) const;
    double calc_p_i_eq23(int i, double q_rtd_C) const;
    double calc_p_i_eq24(int i, double q_rtd_C) const;

    // ===== a0..a4（Python eq7 / eq22） =====
    void calc_a_eq7 (double q_rtd_C, bool dual, double Theta_ex,
                     double& a0, double& a1, double& a2, double& a3, double& a4) const;
    void calc_a_eq22(double Theta_ex, double q_rtd_C, bool dual,
                     double& a0, double& a1, double& a2, double& a3, double& a4) const;

    // ===== f(x, Theta) =====
    double f_H_Theta(double x, double q_rtd_C, double Theta_ex, bool dual) const;
    double f_C_Theta(double x, double Theta_ex, double q_rtd_C, bool dual) const;

    // ===== 最大出力比 Q_r_max_*（Python eq3 / eq13） =====
    static void   calc_a_eq3 (double q_r_max_H, double q_rtd_C, double& a2, double& a1, double& a0);
    static double calc_Q_r_max_H(double q_rtd_C, double q_r_max_H, double Theta_ex);
    static void   calc_a_eq13(double q_r_max_C, double q_rtd_C, double& a2, double& a1, double& a0);
    static double calc_Q_r_max_C(double q_r_max_C, double q_rtd_C, double Theta_ex);

    // ===== ユーティリティ =====
    static inline double clip(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }
    static inline double MJh_from_W(double W) { return W * 3600.0 * 1e-6; } // W → MJ/h
    static inline double kW_from_MJh(double MJh) { return MJh / 3.6; }      // MJ/h → kW

    // 仕様アクセス（kW → W）
    double qrtdC_W() const; // Q.cooling.rtd
    double qmaxC_W() const; // Q.cooling.max
    double prtdC_W() const; // P.cooling.rtd
    double qrtdH_W() const; // Q.heating.rtd
    double qmaxH_W() const; // Q.heating.max
    double prtdH_W() const; // P.heating.rtd

    bool dualCompressor() const;

    // AirconSpecから継承した純粋仮想関数の実装
    virtual double calculatePowerConsumption(double cooling_load, double outdoor_temp, double indoor_temp) const override;
    virtual double calculateCoolingCapacity(double power_consumption, double outdoor_temp, double indoor_temp) const override;
    virtual bool isValidOperatingCondition(double outdoor_temp, double indoor_temp) const override;

private:
    bool dualcompressor_ = false;

    std::vector<std::string> preparationLogs_;
    std::vector<std::string> calculationLogs_;
};

} // namespace acmodel
