#include "latent_evaluate_model.h"
#include "refrigerant_calculator.h"
#include "../../archenv/include/archenv.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ae = archenv;
namespace acmodel {

LatentEvaluateModel::LatentEvaluateModel(const nlohmann::json& spec)
    : AirconSpec(spec) {
    preparationLogs_.clear();
    preparationLogs_.push_back("　　　LatentEvaluateModel初期化: 細井先生の潜熱評価式を使用");
    preparationLogs_.push_back(
        "　　　定数: C_P_AIR=" + std::to_string(ae::SPECIFIC_HEAT_AIR) +
        " J/kgK, RHO_AIR=" + std::to_string(ae::DENSITY_DRY_AIR) +
        " kg/m3, C_P_W=" + std::to_string(ae::SPECIFIC_HEAT_WATER_VAPOR));
}

LatentEvaluateModel::~LatentEvaluateModel() = default;

std::string LatentEvaluateModel::getModelName() const {
    return "LATENT_EVALUATE";
}

nlohmann::json LatentEvaluateModel::getModelParameters() const {
    nlohmann::json params;
    params["model_type"]  = "LATENT_EVALUATE";
    params["model_name"]  = getModelName();
    params["area_switch_kW"] = 5.6; // 5.6kW未満/以上で A_f_hex/A_e_hex を切り替え
    return params;
}

bool LatentEvaluateModel::isValidOperatingCondition(double outdoor_temp, double indoor_temp) const {
    // 運転条件の妥当性をチェック
    return (outdoor_temp >= -20.0 && outdoor_temp <= 50.0 &&
            indoor_temp >= 10.0 && indoor_temp <= 35.0);
}

COPResult LatentEvaluateModel::estimateCOP(const std::string& mode, const InputData& inputdata) {
    if (!isValidOperatingCondition(inputdata.T_ex, inputdata.T_in)) {
        return COPResult(); // invalid
    }
    if (mode == "cooling") return estimateCoolingCOP(inputdata);
    if (mode == "heating") return estimateHeatingCOP(inputdata);

    // Python 版と同様に顕熱の符号で自動振り分けしたい場合は以下で代替可能
    // if (inputdata.Q_S < 0) return estimateCoolingCOP(inputdata);
    // if (inputdata.Q_S > 0) return estimateHeatingCOP(inputdata);

    return COPResult();
}

/* ========================= 冷房 ========================= */
COPResult LatentEvaluateModel::estimateCoolingCOP(const InputData& in) {
    COPResult result;
    calculationLogs_.clear();
    result.logMessages = preparationLogs_;

    // 仕様値 (W, m3/h へ換算)
    const double q_hs_rtd_C  = getCapacity("cooling", "rtd") * 1000.0;
    const double q_hs_mid_C  = getCapacity("cooling", "mid") * 1000.0;
    const double q_hs_min_C  = getCapacity("cooling", "min") * 1000.0;
    const double P_hs_rtd_C  = getPower("cooling", "rtd")   * 1000.0;
    const double V_fan_rtd_C = getVolume("V_inner", "cooling", "rtd") * 3600.0;
    const double V_fan_mid_C = getVolume("V_inner", "cooling", "mid") * 3600.0;
    const double P_fan_rtd_C = getFanPower("cooling", "rtd") * 1000.0;
    const double P_fan_mid_C = getFanPower("cooling", "mid") * 1000.0;
    const double V_hs_dsgn_C = getVolume("V_inner", "cooling", "dsgn") * 3600.0;

    (void)q_hs_mid_C; (void)q_hs_min_C; (void)P_hs_rtd_C;
    (void)V_fan_rtd_C; (void)V_fan_mid_C; (void)P_fan_rtd_C; (void)P_fan_mid_C; (void)V_hs_dsgn_C;

    const double Theta        = in.T_ex;
    const double Theta_hs_in  = in.T_in;
    const double X_hs_in      = in.X_in;          // 室内空気の絶対湿度[kg/kg(DA)]
    const double q_hs_CS      = in.Q_S;           // 顕熱（冷房は負の想定だが式はそのまま）
    const double q_hs_CL      = in.Q_L;           // 潜熱
    const double V_hs_supply  = in.V_inner * 3600.0; // m3/h

    // 吹出温度
    const double Theta_hs_out = Theta_hs_in - q_hs_CS / (ae::SPECIFIC_HEAT_AIR * ae::DENSITY_DRY_AIR * V_hs_supply / 3600.0);

    // 総冷房能力
    const double q_hs_C = q_hs_CS + q_hs_CL;

    // 熱交換器面積（5.6kW閾値）
    const double A_f_hex = getHeatExchangerFrontArea(q_hs_rtd_C);
    const double A_e_hex = getHeatExchangerSurfaceArea(q_hs_rtd_C);

    // 送風機消費電力（細井式）
    const double E_E_fan_C = calculateFanPowerByHosoi(q_hs_C);

    // 顕熱伝達率（細井式）
    const double alpha_c_hex_C =
        calculateLatentSensibleHeatTransferCoeff(V_hs_supply, X_hs_in, A_f_hex);

    // 室内機熱交換器表面温度
    const double Theta_sur_f_hex_C =
        calculateCoolingSurfaceTemp(Theta_hs_in, Theta_hs_out, V_hs_supply,
                                    alpha_c_hex_C, A_e_hex);

    // 冷媒温度群
    const double Theta_ref_evp_C = (Theta_sur_f_hex_C < -50.0) ? -50.0 : Theta_sur_f_hex_C;

    double Theta_ref_cnd_C = std::max(Theta + 27.4 - 1.35 * Theta_ref_evp_C, Theta);
    Theta_ref_cnd_C = std::min(Theta_ref_cnd_C, 65.0);
    Theta_ref_cnd_C = std::max(Theta_ref_cnd_C, Theta_ref_evp_C + 5.0);

    const double Theta_ref_SC_C = std::max(0.772 * Theta_ref_cnd_C - 25.6, 0.0);
    const double Theta_ref_SH_C = std::max(0.194 * Theta_ref_cnd_C - 3.86, 0.0);

    // 冷房の理論効率 e_th = e_dash - 1
    const double e_dash_th = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Theta_ref_evp_C, Theta_ref_cnd_C, Theta_ref_SC_C, Theta_ref_SH_C);
    const double e_th_C = e_dash_th - 1.0;

    // 効率比 e_r（細井式）
    const double x_kW = q_hs_C / 1000.0;
    const double e_r_C = (q_hs_C > 0.0) ? (-0.0316 * x_kW * x_kW + 0.2944 * x_kW) : 0.0;

    // 熱源機効率
    const double e_hs_C = e_th_C * e_r_C;

    // 圧縮機電力[kW] と 総消費電力[kW]
    const double E_E_comp_C = (q_hs_C > 0.0 && e_hs_C > 0.0) ? (q_hs_C / e_hs_C) * 1e-3 : 0.0;
    const double E_E_C_d_t  = E_E_comp_C + E_E_fan_C;

    // COP
    result.COP   = (E_E_C_d_t > 0.0) ? (q_hs_C * 1e-3) / E_E_C_d_t : 0.0;
    result.power = E_E_C_d_t;
    result.valid = (E_E_C_d_t > 0.0);
    return result;
}

/* ========================= 暖房 ========================= */
COPResult LatentEvaluateModel::estimateHeatingCOP(const InputData& in) {
    COPResult result;
    calculationLogs_.clear();
    result.logMessages = preparationLogs_;

    // 仕様値 (W, m3/h へ換算)
    const double q_hs_rtd_H  = getCapacity("heating", "rtd") * 1000.0;
    const double q_hs_mid_H  = getCapacity("heating", "mid") * 1000.0;
    const double q_hs_min_H  = getCapacity("heating", "min") * 1000.0;
    const double P_hs_rtd_H  = getPower("heating", "rtd")   * 1000.0;
    const double V_fan_rtd_H = getVolume("V_inner", "heating", "rtd") * 3600.0;
    const double V_fan_mid_H = getVolume("V_inner", "heating", "mid") * 3600.0;
    const double P_fan_rtd_H = getFanPower("heating", "rtd") * 1000.0;
    const double P_fan_mid_H = getFanPower("heating", "mid") * 1000.0;
    const double V_hs_dsgn_H = getVolume("V_inner", "heating", "dsgn") * 3600.0;

    (void)q_hs_mid_H; (void)q_hs_min_H; (void)P_hs_rtd_H;
    (void)V_fan_rtd_H; (void)V_fan_mid_H; (void)P_fan_rtd_H; (void)P_fan_mid_H; (void)V_hs_dsgn_H;

    const double Theta        = in.T_ex;
    const double Theta_hs_in  = in.T_in;
    const double X_hs_in      = in.X_in;          // 未使用（暖房では顕熱のみ）
    const double V_hs_supply  = in.V_inner * 3600.0; // m3/h

    (void)X_hs_in;

    // 暖房負荷（顕熱）
    double q_hs_H = in.Q_S;

    // 吹出温度
    const double Theta_hs_out = Theta_hs_in + q_hs_H / (ae::SPECIFIC_HEAT_AIR * ae::DENSITY_DRY_AIR * V_hs_supply / 3600.0);

    // 相対湿度 [%] を archenv で算出 → デフロスト補正
    const double e_sat  = ae::saturation_vapor_pressure(Theta);
    const double X_sat  = ae::absolute_humidity_from_vapor_pressure(e_sat);
    const double h_pct  = (X_sat > 0.0) ? (in.X_ex / X_sat * 100.0) : 0.0;

    const double C_df_H = ((Theta < 5.0) && (h_pct >= 80.0)) ? 0.77 : 1.0;
    q_hs_H = q_hs_H * (1.0 / C_df_H);

    // 熱交換器面積（5.6kW閾値）
    const double A_f_hex = getHeatExchangerFrontArea(q_hs_rtd_H);
    const double A_e_hex = getHeatExchangerSurfaceArea(q_hs_rtd_H);

    // 送風機消費電力（細井式）
    const double E_E_fan_H = calculateFanPowerByHosoi(q_hs_H);

    // 暖房時の顕熱伝達率（経験式）
    const double alpha_c_hex_H = calculateHeatingHeatTransferCoeff(V_hs_supply, A_f_hex);

    // 室内機熱交換器表面温度
    const double Theta_sur_f_hex_H =
        calculateHeatingSurfaceTemp(Theta_hs_in, Theta_hs_out, V_hs_supply,
                                    alpha_c_hex_H, A_e_hex);

    // 冷媒温度群（暖房）
    const double Theta_ref_cnd_H = (Theta_sur_f_hex_H > 65.0) ? 65.0 : Theta_sur_f_hex_H;

    double Theta_ref_evp_H = Theta - (0.100 * Theta_ref_cnd_H + 2.95);
    if (Theta_ref_evp_H < -50.0)            Theta_ref_evp_H = -50.0;
    if (Theta_ref_evp_H > Theta_ref_cnd_H - 5.0)
        Theta_ref_evp_H = Theta_ref_cnd_H - 5.0;

    const double Theta_ref_SC_H = 0.245 * Theta_ref_cnd_H - 1.72;
    const double Theta_ref_SH_H = 4.49 - 0.036 * Theta_ref_cnd_H;

    // 暖房の理論効率（Python はそのまま e_th = e_dash_th）
    const double e_th_H = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Theta_ref_evp_H, Theta_ref_cnd_H, Theta_ref_SC_H, Theta_ref_SH_H);

    // 効率比 e_r（細井式）
    const double x_kW_H = q_hs_H / 1000.0;
    const double e_r_H  = (q_hs_H > 0.0) ? (-0.0316 * x_kW_H * x_kW_H + 0.2944 * x_kW_H) : 0.0;

    // 熱源機効率
    const double e_hs_H = e_th_H * e_r_H;

    // 電力[kW]
    const double E_E_comp_H = (q_hs_H > 0.0 && e_hs_H > 0.0) ? (q_hs_H / e_hs_H) * 1e-3 : 0.0;
    const double E_E_H_d_t  = E_E_comp_H + E_E_fan_H;

    // COP
    result.COP   = (E_E_H_d_t > 0.0) ? (q_hs_H * 1e-3) / E_E_H_d_t : 0.0;
    result.power = E_E_H_d_t;
    result.valid = (E_E_H_d_t > 0.0);
    return result;
}

/* ========================= 参考ユーティリティ ========================= */

double LatentEvaluateModel::calculatePowerConsumption(double cooling_load, double outdoor_temp, double /*indoor_temp*/) const {
    if (cooling_load <= 0.0) return 0.0;
    double efficiency_factor = 1.0;
    if (outdoor_temp > 35.0) efficiency_factor = 0.8;
    else if (outdoor_temp < 5.0) efficiency_factor = 0.7;

    // ベース COP は仕様/設計次第。簡易に 3.0 相当で計算（必要なら spec_ から取得して更新）
    const double base_cop = 3.0;
    const double effective_cop = base_cop * efficiency_factor;
    return cooling_load / effective_cop;
}

double LatentEvaluateModel::calculateCoolingCapacity(double power_consumption, double outdoor_temp, double /*indoor_temp*/) const {
    if (power_consumption <= 0.0) return 0.0;
    double efficiency_factor = 1.0;
    if (outdoor_temp > 35.0) efficiency_factor = 0.8;
    else if (outdoor_temp < 5.0) efficiency_factor = 0.7;

    const double base_cop = 3.0;
    const double effective_cop = base_cop * efficiency_factor;
    return power_consumption * effective_cop;
}

/* ========================= 細井式・下請け関数群 ========================= */

double LatentEvaluateModel::getHeatExchangerFrontArea(double rated_capacity_W) const {
    // 5.6kW 未満/以上で切替
    return (rated_capacity_W < 5600.0) ? ae::A_F_HEX_SMALL : ae::A_F_HEX_LARGE;
}

double LatentEvaluateModel::getHeatExchangerSurfaceArea(double rated_capacity_W) const {
    return (rated_capacity_W < 5600.0) ? ae::A_E_HEX_SMALL : ae::A_E_HEX_LARGE;
}

double LatentEvaluateModel::calculateFanPowerByHosoi(double capacity_W) const {
    if (capacity_W <= 0.0) return 0.0;
    const double x = capacity_W / 1000.0; // kW
    // E_E_fan [kW]
    return (1.4675 * x * x * x - 8.5886 * x * x + 20.217 * x + 50.0) * 1e-3;
}

double LatentEvaluateModel::calculateLatentSensibleHeatTransferCoeff(double v_flow_m3ph,
                                                                     double x_in_kgpkg,
                                                                     double a_f_hex_m2) const {
    const double v_clip = std::max(v_flow_m3ph, 360.0);
    const double x = v_clip / (3600.0 * a_f_hex_m2);
    const double alpha_dash = 0.0631 * x + 0.0015; // [kg/(m2 s)] → 顕熱換算
    return alpha_dash * (ae::SPECIFIC_HEAT_AIR + ae::SPECIFIC_HEAT_WATER_VAPOR * 1000.0 * x_in_kgpkg); // [W/(m2 K)] (C_P_W: kJ/kg/K -> J/kg/K)
}

double LatentEvaluateModel::calculateLatentLatentHeatTransferCoeff(double v_flow_m3ph,
                                                                   double a_f_hex_m2) const {
    const double v_clip = std::max(v_flow_m3ph, 360.0);
    const double x = v_clip / (3600.0 * a_f_hex_m2);
    return 0.0631 * x + 0.0015; // [kg/(m2 s)]
}

double LatentEvaluateModel::calculateCoolingSurfaceTemp(double theta_in_C, double theta_out_C,
                                                        double v_flow_m3ph,
                                                        double alpha_c_W_m2K,
                                                        double a_e_hex_m2) const {
    return ((theta_in_C + theta_out_C) / 2.0) -
           (ae::SPECIFIC_HEAT_AIR * ae::DENSITY_DRY_AIR * v_flow_m3ph * (theta_in_C - theta_out_C) / 3600.0) /
               (a_e_hex_m2 * alpha_c_W_m2K);
}

double LatentEvaluateModel::calculateHeatingSurfaceTemp(double theta_in_C, double theta_out_C,
                                                        double v_flow_m3ph,
                                                        double alpha_c_W_m2K,
                                                        double a_e_hex_m2) const {
    return ((theta_in_C + theta_out_C) / 2.0) +
           (ae::SPECIFIC_HEAT_AIR * ae::DENSITY_DRY_AIR * v_flow_m3ph * (theta_out_C - theta_in_C) / 3600.0) /
               (a_e_hex_m2 * alpha_c_W_m2K);
}

double LatentEvaluateModel::calculateCoolingCondensingTemp(double theta_ex_C, double theta_evp_C) const {
    double theta_cnd = std::max(theta_ex_C + 27.4 - 1.35 * theta_evp_C, theta_ex_C);
    theta_cnd = std::min(theta_cnd, 65.0);
    theta_cnd = std::max(theta_cnd, theta_evp_C + 5.0);
    return theta_cnd;
}

double LatentEvaluateModel::calculateHeatingEvaporatingTemp(double theta_ex_C, double theta_cnd_C) const {
    double theta_evp = theta_ex_C - (0.100 * theta_cnd_C + 2.95);
    theta_evp = std::max(theta_evp, -50.0);
    theta_evp = std::min(theta_evp, theta_cnd_C - 5.0);
    return theta_evp;
}

double LatentEvaluateModel::calculateCoolingSubcooling(double theta_cnd_C) const {
    return std::max(0.772 * theta_cnd_C - 25.6, 0.0);
}

double LatentEvaluateModel::calculateHeatingSubcooling(double theta_cnd_C) const {
    return 0.245 * theta_cnd_C - 1.72;
}

double LatentEvaluateModel::calculateCoolingSuperheating(double theta_cnd_C) const {
    return std::max(0.194 * theta_cnd_C - 3.86, 0.0);
}

double LatentEvaluateModel::calculateHeatingSuperheating(double theta_cnd_C) const {
    return 4.49 - 0.036 * theta_cnd_C;
}

double LatentEvaluateModel::calculateHeatingHeatTransferCoeff(double v_flow_m3ph,
                                                              double a_f_hex_m2) const {
    const double v_ratio = (v_flow_m3ph / 3600.0) / a_f_hex_m2;
    return (-0.0017 * v_ratio * v_ratio + 0.044 * v_ratio + 0.0271) * 1000.0; // [W/(m2 K)]
}

} // namespace acmodel
