#include "duct_central_model.h"
#include "refrigerant_calculator.h"  // RefrigerantCalculator::calculateTheoreticalHeatingEfficiency
#include "acmodel.h"  // acmodel::log
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace ae = archenv;
namespace acmodel {

// =============== コンストラクタ ===============
DuctCentralModel::DuctCentralModel(const nlohmann::json& spec)
    : AirconSpec(spec) {
    preparationLogs_.clear();
    acmodel::log("　　　DuctCentralModel初期化: " + spec.dump());
    
    std::ostringstream oss;
    oss << "　　　DuctCentralModel初期化: A_F_HEX=" << A_F_HEX << " m2, A_E_HEX=" << A_E_HEX << " m2";
    preparationLogs_.push_back(oss.str());
    
    // 定格仕様のログ出力
    try {
        double q_rtd_C = getCapacity("cooling", "rtd");
        double q_max_C = getCapacity("cooling", "mid");
        double p_rtd_C = getPower("cooling", "rtd");
        double q_rtd_H = getCapacity("heating", "rtd");
        double q_max_H = getCapacity("heating", "mid");
        double p_rtd_H = getPower("heating", "rtd");
        
        preparationLogs_.push_back("　　　冷房定格: 能力=" + std::to_string(q_rtd_C/1000.0) + "kW, 中間=" + 
                                   std::to_string(q_max_C/1000.0) + "kW, 電力=" + std::to_string(p_rtd_C/1000.0) + "kW");
        preparationLogs_.push_back("　　　暖房定格: 能力=" + std::to_string(q_rtd_H/1000.0) + "kW, 中間=" + 
                                   std::to_string(q_max_H/1000.0) + "kW, 電力=" + std::to_string(p_rtd_H/1000.0) + "kW");
        
        // 熱交換器面積などの物性定数
        preparationLogs_.push_back("　　　物性定数: C_P_AIR=" + std::to_string(C_P_AIR) + "J/kg/K, RHO_AIR=" + 
                                   std::to_string(RHO_AIR) + "kg/m3, L_WTR=" + std::to_string(L_WTR) + "kJ/kg");
        
        acmodel::log("　　　DuctCentralModel初期化完了");
    } catch (const std::exception& e) {
        preparationLogs_.push_back("　　　警告: 仕様データの一部取得に失敗: " + std::string(e.what()));
    }
}
DuctCentralModel::~DuctCentralModel() = default;

// =============== JSON 仕様アクセス ===============
double DuctCentralModel::getCapacity(const char* mode, const char* key) const {
    // kW → W
    return spec_.at("Q").at(mode).at(key).get<double>() * 1000.0;
}
double DuctCentralModel::getPower(const char* mode, const char* key) const {
    // kW → W
    return spec_.at("P").at(mode).at(key).get<double>() * 1000.0;
}
double DuctCentralModel::getFanPower(const char* mode, const char* key) const {
    // kW → W
    return spec_.at("P_fan").at(mode).at(key).get<double>() * 1000.0;
}
double DuctCentralModel::getVolume(const char* vwhere, const char* mode, const char* key) const {
    // m3/s → m3/h
    return spec_.at(vwhere).at(mode).at(key).get<double>() * 3600.0;
}

// =============== 公開I/F ===============
COPResult DuctCentralModel::estimateCOP(const std::string& mode, const InputData& inputdata) {
    if (!isValidOperatingCondition(inputdata.T_ex, inputdata.T_in)) {
        COPResult r; r.valid = false; r.COP = 0.0; r.power = 0.0;
        return r;
    }
    if (mode == "cooling") return estimateCoolingCOP(inputdata);
    if (mode == "heating") return estimateHeatingCOP(inputdata);
    // Pythonと同様に Q_S で自動分岐してもよいが、ここでは明示モードのみ
    COPResult r; r.valid = false; r.COP = 0.0; r.power = 0.0;
    return r;
}

// =============== 冷房 ===============
COPResult DuctCentralModel::estimateCoolingCOP(const InputData& in) {
    COPResult result; calculationLogs_.clear();
    result.logMessages = preparationLogs_;

    calculationLogs_.push_back("　　　DuctCentral冷房COP推定開始: 外気=" + std::to_string(in.T_ex) + 
                               "°C, 室内=" + std::to_string(in.T_in) + "°C");

    const double q_hs_rtd_C  = getCapacity("cooling","rtd");
    const double q_hs_mid_C  = getCapacity("cooling","mid");
    const double q_hs_min_C  = getCapacity("cooling","min");
    const double P_hs_rtd_C  = getPower   ("cooling","rtd");
    const double V_fan_rtd_C = getVolume  ("V_inner","cooling","rtd"); // m3/h
    const double V_fan_mid_C = getVolume  ("V_inner","cooling","mid"); // m3/h
    const double P_fan_rtd_C = getFanPower("cooling","rtd");
    // const double P_fan_mid_C = getFanPower("cooling","mid"); // 未使用のためコメントアウト
    const double V_hs_dsgn_C = getVolume  ("V_inner","cooling","dsgn");
    
    calculationLogs_.push_back("　　　定格仕様: q_rtd=" + std::to_string(q_hs_rtd_C) + "W, q_mid=" + 
                               std::to_string(q_hs_mid_C) + "W, q_min=" + std::to_string(q_hs_min_C) + "W");
    calculationLogs_.push_back("　　　電力仕様: P_rtd=" + std::to_string(P_hs_rtd_C) + "W, P_fan_rtd=" + 
                               std::to_string(P_fan_rtd_C) + "W");
    calculationLogs_.push_back("　　　風量仕様: V_rtd=" + std::to_string(V_fan_rtd_C) + "m3/h, V_mid=" + 
                               std::to_string(V_fan_mid_C) + "m3/h, V_dsgn=" + std::to_string(V_hs_dsgn_C) + "m3/h");

    const double Theta       = in.T_ex;
    const double Theta_hs_in = in.T_in;
    const double X_hs_in     = in.X_in;              // 室内側絶対湿度
    const double q_hs_CS     = in.Q_S;               // 顕熱 [W]
    const double q_hs_CL     = in.Q_L;               // 潜熱 [W]
    const double V_hs_supply = in.V_inner * 3600.0;  // m3/h
    
    calculationLogs_.push_back("　　　入力条件: 外気温=" + std::to_string(Theta) + "°C, 室内温=" + 
                               std::to_string(Theta_hs_in) + "°C, 室内湿度=" + std::to_string(X_hs_in) + "kg/kg");
    calculationLogs_.push_back("　　　負荷条件: 顕熱=" + std::to_string(q_hs_CS) + "W, 潜熱=" + 
                               std::to_string(q_hs_CL) + "W, 送風量=" + std::to_string(V_hs_supply) + "m3/h");

    // 吹出温度
    const double Theta_hs_out = Theta_hs_in - q_hs_CS / (C_P_AIR * RHO_AIR * (V_hs_supply/3600.0));
    calculationLogs_.push_back("　　　吹出温度: " + std::to_string(Theta_hs_out) + "°C");

    // 総冷房能力
    const double q_hs_C = q_hs_CS + q_hs_CL;
    calculationLogs_.push_back("　　　総冷房能力: " + std::to_string(q_hs_C) + "W");

    // 送風機消費電力（q_hs_C > 0 のときのみ）
    const double E_E_fan_C_kW = (q_hs_C > 0.0)
        ? calculateFanPower(V_hs_supply, V_hs_dsgn_C, P_fan_rtd_C)
        : 0.0;
    calculationLogs_.push_back("　　　送風機電力: " + std::to_string(E_E_fan_C_kW) + "kW");

    // ---- 中間点：bisect で Θ_surf を解く → 冷媒温度 → e_th_mid_C ----
    const double Theta_sur_mid = solveCoolingSurfaceTempByBisect(V_fan_mid_C, q_hs_mid_C, 27.0 /*式の内部定数*/);
    double Theta_ref_evp_mid = std::max(Theta_sur_mid, -50.0);
    double Theta_ref_cnd_mid = coolingCondensingTemp_mid_rtd(Theta_ref_evp_mid);
    const double Theta_ref_SC_mid = coolingSubcooling(Theta_ref_cnd_mid);
    const double Theta_ref_SH_mid = coolingSuperheating(Theta_ref_cnd_mid);
    const double e_dash_th_mid    = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Theta_ref_evp_mid, Theta_ref_cnd_mid, Theta_ref_SC_mid, Theta_ref_SH_mid);
    const double e_th_mid_C = e_dash_th_mid - 1.0;
    
    calculationLogs_.push_back("　　　中間点計算: 表面温度=" + std::to_string(Theta_sur_mid) + "°C, 蒸発温度=" + 
                               std::to_string(Theta_ref_evp_mid) + "°C, 凝縮温度=" + std::to_string(Theta_ref_cnd_mid) + "°C");
    calculationLogs_.push_back("　　　中間点冷媒: SC=" + std::to_string(Theta_ref_SC_mid) + "°C, SH=" + 
                               std::to_string(Theta_ref_SH_mid) + "°C, 理論効率=" + std::to_string(e_th_mid_C));

    // ---- 定格点：bisect で Θ_surf を解く → 冷媒温度 → e_th_rtd_C ----
    const double Theta_sur_rtd = solveCoolingSurfaceTempByBisect(V_fan_rtd_C, q_hs_rtd_C, 27.0);
    double Theta_ref_evp_rtd = std::max(Theta_sur_rtd, -50.0);
    double Theta_ref_cnd_rtd = coolingCondensingTemp_mid_rtd(Theta_ref_evp_rtd);
    const double Theta_ref_SC_rtd = coolingSubcooling(Theta_ref_cnd_rtd);
    const double Theta_ref_SH_rtd = coolingSuperheating(Theta_ref_cnd_rtd);
    const double e_dash_th_rtd    = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Theta_ref_evp_rtd, Theta_ref_cnd_rtd, Theta_ref_SC_rtd, Theta_ref_SH_rtd);
    const double e_th_rtd_C = e_dash_th_rtd - 1.0;
    
    calculationLogs_.push_back("　　　定格点計算: 表面温度=" + std::to_string(Theta_sur_rtd) + "°C, 蒸発温度=" + 
                               std::to_string(Theta_ref_evp_rtd) + "°C, 凝縮温度=" + std::to_string(Theta_ref_cnd_rtd) + "°C");
    calculationLogs_.push_back("　　　定格点冷媒: SC=" + std::to_string(Theta_ref_SC_rtd) + "°C, SH=" + 
                               std::to_string(Theta_ref_SH_rtd) + "°C, 理論効率=" + std::to_string(e_th_rtd_C));

    // ---- 評価点（実風量）：閉式で Θ_surf → 冷媒温度 → e_th_C ----
    // α_c(eval)：X_hs_in を使用
    const double alpha_c_eval = sensibleHTCoeffCooling_eval(V_hs_supply, X_hs_in);
    const double Theta_sur_eval = coolingSurfaceTempEval(Theta_hs_in, Theta_hs_out, V_hs_supply, alpha_c_eval);
    double Theta_ref_evp = std::max(Theta_sur_eval, -50.0);
    double Theta_ref_cnd = coolingCondensingTemp_eval(Theta, Theta_ref_evp);
    const double Theta_ref_SC = coolingSubcooling(Theta_ref_cnd);
    const double Theta_ref_SH = coolingSuperheating(Theta_ref_cnd);
    const double e_dash_th    = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Theta_ref_evp, Theta_ref_cnd, Theta_ref_SC, Theta_ref_SH);
    const double e_th_C = e_dash_th - 1.0;
    
    calculationLogs_.push_back("　　　評価点計算: 顕熱係数=" + std::to_string(alpha_c_eval) + "W/m2/K, 表面温度=" + 
                               std::to_string(Theta_sur_eval) + "°C");
    calculationLogs_.push_back("　　　評価点冷媒: 蒸発温度=" + std::to_string(Theta_ref_evp) + "°C, 凝縮温度=" + 
                               std::to_string(Theta_ref_cnd) + "°C, 理論効率=" + std::to_string(e_th_C));

    // ---- e_r の基準化は「定格理論効率」で実施（Python準拠） ----
    const double e_hs_rtd_C = q_hs_rtd_C / (P_hs_rtd_C - P_fan_rtd_C);
    double e_r_rtd_C = e_hs_rtd_C / e_th_rtd_C;
    e_r_rtd_C = std::clamp(e_r_rtd_C, 0.0, 1.0);

    const double e_r_min_C = e_r_rtd_C * 0.65;
    const double e_r_mid_C = e_r_rtd_C * 0.95;
    
    calculationLogs_.push_back("　　　効率比計算: e_hs_rtd=" + std::to_string(e_hs_rtd_C) + ", e_r_rtd=" + 
                               std::to_string(e_r_rtd_C) + ", e_r_min=" + std::to_string(e_r_min_C) + 
                               ", e_r_mid=" + std::to_string(e_r_mid_C));

    const double e_r_C = efficiencyRatioPiecewise(q_hs_C, q_hs_min_C, q_hs_mid_C, q_hs_rtd_C,
                                                  e_r_min_C, e_r_mid_C, e_r_rtd_C);
    calculationLogs_.push_back("　　　区分線形効率比: e_r_C=" + std::to_string(e_r_C));

    // 熱源機効率
    const double e_hs_C = e_th_C * e_r_C;
    calculationLogs_.push_back("　　　熱源機効率: e_hs_C=" + std::to_string(e_hs_C));

    // 圧縮機電力[kW]
    const double E_E_comp_C_kW = (q_hs_C > 0.0) ? (q_hs_C / e_hs_C) * 1e-3 : 0.0;
    calculationLogs_.push_back("　　　圧縮機電力: " + std::to_string(E_E_comp_C_kW) + "kW");

    // 総消費電力[kW]
    const double E_E_C_d_t_kW = E_E_comp_C_kW + E_E_fan_C_kW;
    calculationLogs_.push_back("　　　総消費電力: " + std::to_string(E_E_C_d_t_kW) + "kW (圧縮機:" + 
                               std::to_string(E_E_comp_C_kW) + "kW + 送風機:" + std::to_string(E_E_fan_C_kW) + "kW)");

    // COP
    const double COP = (E_E_C_d_t_kW > 0.0) ? (q_hs_C * 1e-3) / E_E_C_d_t_kW : 0.0;

    if (E_E_C_d_t_kW > 0.0) {
        result.COP   = COP;
        result.power = E_E_C_d_t_kW;
        result.valid = true;
        
        calculationLogs_.push_back("　　　最終結果: 総負荷=" + std::to_string(q_hs_C/1000.0) + "kW (顕熱:" + 
                                   std::to_string(q_hs_CS/1000.0) + "kW + 潜熱:" + std::to_string(q_hs_CL/1000.0) + 
                                   "kW), COP=" + std::to_string(COP) + ", 電力=" + std::to_string(E_E_C_d_t_kW) + "kW");
    } else {
        result.COP = 0.0;
        result.power = 0.0;
        result.valid = false;
        calculationLogs_.push_back("　　　エラー: 消費電力計算が無効です");
    }
    
    // ログメッセージを結果に含める
    result.logMessages.insert(result.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
    
    return result;
}

// =============== 暖房 ===============
COPResult DuctCentralModel::estimateHeatingCOP(const InputData& in) {
    COPResult result; calculationLogs_.clear();
    result.logMessages = preparationLogs_;

    calculationLogs_.push_back("　　　DuctCentral暖房COP推定開始: 外気=" + std::to_string(in.T_ex) + 
                               "°C, 室内=" + std::to_string(in.T_in) + "°C");

    const double q_hs_rtd_H  = getCapacity("heating","rtd");
    const double q_hs_mid_H  = getCapacity("heating","mid");
    const double q_hs_min_H  = getCapacity("heating","min");
    const double P_hs_rtd_H  = getPower   ("heating","rtd");
    const double V_fan_rtd_H = getVolume  ("V_inner","heating","rtd");
    const double V_fan_mid_H = getVolume  ("V_inner","heating","mid");
    const double P_fan_rtd_H = getFanPower("heating","rtd");
    // const double P_fan_mid_H = getFanPower("heating","mid"); // 未使用のためコメントアウト
    const double V_hs_dsgn_H = getVolume  ("V_inner","heating","dsgn");
    
    calculationLogs_.push_back("　　　暖房定格仕様: q_rtd=" + std::to_string(q_hs_rtd_H) + "W, q_mid=" + 
                               std::to_string(q_hs_mid_H) + "W, q_min=" + std::to_string(q_hs_min_H) + "W");
    calculationLogs_.push_back("　　　暖房電力仕様: P_rtd=" + std::to_string(P_hs_rtd_H) + "W, P_fan_rtd=" + 
                               std::to_string(P_fan_rtd_H) + "W");
    calculationLogs_.push_back("　　　暖房風量仕様: V_rtd=" + std::to_string(V_fan_rtd_H) + "m3/h, V_mid=" + 
                               std::to_string(V_fan_mid_H) + "m3/h, V_dsgn=" + std::to_string(V_hs_dsgn_H) + "m3/h");

    const double Theta       = in.T_ex;
    const double Theta_hs_in = in.T_in;
    const double X_hs_in     = in.X_in;
    double       q_hs_H      = in.Q_S;
    const double V_hs_supply = in.V_inner * 3600.0;
    
    calculationLogs_.push_back("　　　暖房入力条件: 外気温=" + std::to_string(Theta) + "°C, 室内温=" + 
                               std::to_string(Theta_hs_in) + "°C, 室内湿度=" + std::to_string(X_hs_in) + "kg/kg");
    calculationLogs_.push_back("　　　暖房負荷: " + std::to_string(q_hs_H) + "W, 送風量=" + std::to_string(V_hs_supply) + "m3/h");

    const double Theta_hs_out = Theta_hs_in + q_hs_H / (C_P_AIR * RHO_AIR * (V_hs_supply/3600.0));
    calculationLogs_.push_back("　　　吹出温度: " + std::to_string(Theta_hs_out) + "°C");

    // 霜着補正：相対湿度 h[%] = X_ex / X_sat * 100
    const double e_sat  = ae::saturation_vapor_pressure(Theta);           // Pa
    const double X_sat  = ae::absolute_humidity_from_vapor_pressure(e_sat); // kg/kg(DA)
    const double h_pct  = (X_sat > 0.0) ? (in.X_ex / X_sat * 100.0) : 0.0;
    const double C_df_H = ((Theta < 5.0) && (h_pct >= 80.0)) ? 0.77 : 1.0;
    
    calculationLogs_.push_back("　　　霜着判定: X_ex=" + std::to_string(in.X_ex) + "kg/kg, X_sat=" + 
                               std::to_string(X_sat) + "kg/kg, 相対湿度=" + std::to_string(h_pct) + "%");
    calculationLogs_.push_back("　　　霜着補正係数: C_df_H=" + std::to_string(C_df_H));

    // 負荷を 1/C_df_H 倍（必要出力が増える）
    q_hs_H = q_hs_H * (1.0 / C_df_H);
    calculationLogs_.push_back("　　　霜着補正後負荷: " + std::to_string(q_hs_H) + "W");

    // 送風機電力（q_hs_H > 0 のときのみ）
    const double E_E_fan_H_kW = (q_hs_H > 0.0)
        ? calculateFanPower(V_hs_supply, V_hs_dsgn_H, P_fan_rtd_H)
        : 0.0;
    calculationLogs_.push_back("　　　暖房送風機電力: " + std::to_string(E_E_fan_H_kW) + "kW");

    // ---- 中間点 ----
    const double alpha_c_mid = sensibleHTCoeffHeating(V_fan_mid_H);
    const double Theta_sur_mid = 20.0
        + (q_hs_mid_H / (2.0 * C_P_AIR * RHO_AIR * V_fan_mid_H)) * 3600.0
        + (q_hs_mid_H / (A_E_HEX * alpha_c_mid));
    double Tcnd_mid = heatingCondensingTemp_from_surface(Theta_sur_mid);
    double Tevp_mid = heatingEvaporatingTemp_mid_rtd(Tcnd_mid);
    const double SC_mid = heatingSubcooling(Tcnd_mid);
    const double SH_mid = heatingSuperheating(Tcnd_mid);
    const double e_th_mid_H = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Tevp_mid, Tcnd_mid, SC_mid, SH_mid);
    
    calculationLogs_.push_back("　　　暖房中間点: 顕熱係数=" + std::to_string(alpha_c_mid) + "W/m2/K, 表面温度=" + 
                               std::to_string(Theta_sur_mid) + "°C");
    calculationLogs_.push_back("　　　暖房中間冷媒: 凝縮温度=" + std::to_string(Tcnd_mid) + "°C, 蒸発温度=" + 
                               std::to_string(Tevp_mid) + "°C, 理論効率=" + std::to_string(e_th_mid_H));

    // ---- 定格点 ----
    const double alpha_c_rtd = sensibleHTCoeffHeating(V_fan_rtd_H);
    const double Theta_sur_rtd = 20.0
        + (q_hs_rtd_H / (2.0 * C_P_AIR * RHO_AIR * V_fan_rtd_H)) * 3600.0
        + (q_hs_rtd_H / (A_E_HEX * alpha_c_rtd));
    double Tcnd_rtd = heatingCondensingTemp_from_surface(Theta_sur_rtd);
    double Tevp_rtd = heatingEvaporatingTemp_mid_rtd(Tcnd_rtd);
    const double SC_rtd = heatingSubcooling(Tcnd_rtd);
    const double SH_rtd = heatingSuperheating(Tcnd_rtd);
    const double e_th_rtd_H = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Tevp_rtd, Tcnd_rtd, SC_rtd, SH_rtd);
    
    calculationLogs_.push_back("　　　暖房定格点: 顕熱係数=" + std::to_string(alpha_c_rtd) + "W/m2/K, 表面温度=" + 
                               std::to_string(Theta_sur_rtd) + "°C");
    calculationLogs_.push_back("　　　暖房定格冷媒: 凝縮温度=" + std::to_string(Tcnd_rtd) + "°C, 蒸発温度=" + 
                               std::to_string(Tevp_rtd) + "°C, 理論効率=" + std::to_string(e_th_rtd_H));

    // ---- 評価点（実風量） ----
    const double alpha_c_eval = sensibleHTCoeffHeating(V_hs_supply);
    const double Theta_sur_eval = heatingSurfaceTempEval(Theta_hs_in, Theta_hs_out, V_hs_supply, alpha_c_eval);
    double Tcnd = heatingCondensingTemp_from_surface(Theta_sur_eval);
    double Tevp = heatingEvaporatingTemp_eval(Theta, Tcnd);
    const double SC = heatingSubcooling(Tcnd);
    const double SH = heatingSuperheating(Tcnd);
    const double e_th_H = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
        Tevp, Tcnd, SC, SH);
    
    calculationLogs_.push_back("　　　暖房評価点: 顕熱係数=" + std::to_string(alpha_c_eval) + "W/m2/K, 表面温度=" + 
                               std::to_string(Theta_sur_eval) + "°C");
    calculationLogs_.push_back("　　　暖房評価冷媒: 凝縮温度=" + std::to_string(Tcnd) + "°C, 蒸発温度=" + 
                               std::to_string(Tevp) + "°C, 理論効率=" + std::to_string(e_th_H));

    // ---- e_r の基準は定格理論効率 ----
    const double e_hs_rtd_H = q_hs_rtd_H / (P_hs_rtd_H - P_fan_rtd_H);
    double e_r_rtd_H = e_hs_rtd_H / e_th_rtd_H;
    e_r_rtd_H = std::clamp(e_r_rtd_H, 0.0, 1.0);

    const double e_r_min_H = e_r_rtd_H * 0.65;
    const double e_r_mid_H = e_r_rtd_H * 0.95;
    
    calculationLogs_.push_back("　　　暖房効率比計算: e_hs_rtd=" + std::to_string(e_hs_rtd_H) + ", e_r_rtd=" + 
                               std::to_string(e_r_rtd_H) + ", e_r_min=" + std::to_string(e_r_min_H) + 
                               ", e_r_mid=" + std::to_string(e_r_mid_H));

    const double e_r_H = efficiencyRatioPiecewise(q_hs_H, q_hs_min_H, q_hs_mid_H, q_hs_rtd_H,
                                                  e_r_min_H, e_r_mid_H, e_r_rtd_H);
    calculationLogs_.push_back("　　　暖房区分線形効率比: e_r_H=" + std::to_string(e_r_H));

    // 熱源機効率
    const double e_hs_H = e_th_H * e_r_H;
    calculationLogs_.push_back("　　　暖房熱源機効率: e_hs_H=" + std::to_string(e_hs_H));

    // 圧縮機電力[kW]
    const double E_E_comp_H_kW = (q_hs_H > 0.0) ? (q_hs_H / e_hs_H) * 1e-3 : 0.0;
    calculationLogs_.push_back("　　　暖房圧縮機電力: " + std::to_string(E_E_comp_H_kW) + "kW");

    // 総消費電力[kW]
    const double E_E_H_d_t_kW = E_E_comp_H_kW + E_E_fan_H_kW;
    calculationLogs_.push_back("　　　暖房総消費電力: " + std::to_string(E_E_H_d_t_kW) + "kW (圧縮機:" + 
                               std::to_string(E_E_comp_H_kW) + "kW + 送風機:" + std::to_string(E_E_fan_H_kW) + "kW)");

    // COP
    const double COP = (E_E_H_d_t_kW > 0.0) ? (q_hs_H * 1e-3) / E_E_H_d_t_kW : 0.0;

    if (E_E_H_d_t_kW > 0.0) {
        result.COP   = COP;
        result.power = E_E_H_d_t_kW;
        result.valid = true;
        
        calculationLogs_.push_back("　　　暖房最終結果: 補正後負荷=" + std::to_string(q_hs_H/1000.0) + "kW, COP=" + 
                                   std::to_string(COP) + ", 電力=" + std::to_string(E_E_H_d_t_kW) + "kW");
    } else {
        result.COP = 0.0;
        result.power = 0.0;
        result.valid = false;
        calculationLogs_.push_back("　　　エラー: 暖房消費電力計算が無効です");
    }
    
    // ログメッセージを結果に含める
    result.logMessages.insert(result.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
    
    return result;
}

// =============== ユーティリティ実装 ===============
bool DuctCentralModel::isValidOperatingCondition(double outdoor_temp, double indoor_temp) const {
    return (outdoor_temp >= -20.0 && outdoor_temp <= 50.0 &&
            indoor_temp >= 15.0 && indoor_temp <= 35.0);
}

nlohmann::json DuctCentralModel::getModelParameters() const {
    nlohmann::json params;
    params["model_type"] = "DUCT_CENTRAL";
    params["A_F_HEX"] = A_F_HEX;
    params["A_E_HEX"] = A_E_HEX;
    return params;
}

double DuctCentralModel::calculatePowerConsumption(double cooling_load, double /*outdoor_temp*/, double /*indoor_temp*/) const {
    // Pythonの返り値仕様とは異なるため、ここは未使用想定。互換性のために簡易実装。
    // 必要なら estimateCoolingCOP を呼び出して実値を得る運用にしてください。
    if (cooling_load <= 0.0) return 0.0;
    return 0.0; // ダミー
}
double DuctCentralModel::calculateCoolingCapacity(double power_consumption, double /*outdoor_temp*/, double /*indoor_temp*/) const {
    if (power_consumption <= 0.0) return 0.0;
    return 0.0; // ダミー
}

// ---- 送風機電力[kW] ----
double DuctCentralModel::calculateFanPower(double v_supply_m3h, double v_design_m3h, double p_fan_rated_W) {
    if (v_supply_m3h <= 0.0 || v_design_m3h <= 0.0) return 0.0;
    return p_fan_rated_W * (v_supply_m3h / v_design_m3h) * 1e-3;
}

// ---- 伝熱係数 ----
double DuctCentralModel::latentHTCoeff(double v_flow_m3h) {
    const double a = std::max(v_flow_m3h, 400.0);
    return 0.050 * std::log((a/3600.0) / A_F_HEX) + 0.073; // [kg/(m2·s)]
}
double DuctCentralModel::sensibleHTCoeffCooling_mid_rtd(double v_flow_m3h) {
    // α_c = α'_c * (c_p_air + c_p_w*0.010376)
    const double alpha_dash = latentHTCoeff(v_flow_m3h);
    return alpha_dash * (C_P_AIR + (C_P_W * 0.010376 * 1000.0)); // C_P_W[kJ/kgK]→J/kgK
}
double DuctCentralModel::sensibleHTCoeffCooling_eval(double v_flow_m3h, double x_in_kgkg) {
    // α_c = α'_c * (c_p_air + c_p_w*X_in)
    const double alpha_dash = latentHTCoeff(v_flow_m3h);
    return alpha_dash * (C_P_AIR + (C_P_W * x_in_kgkg * 1000.0)); // C_P_W[kJ/kgK]→J/kgK
}
double DuctCentralModel::sensibleHTCoeffHeating(double v_flow_m3h) {
    const double a = (v_flow_m3h/3600.0) / A_F_HEX;
    return (-0.0017 * a * a + 0.044 * a + 0.0271) * 1000.0; // [W/(m2·K)]
}

// ---- 表面温度 ----
double DuctCentralModel::coolingSurfaceTempEval(double theta_in, double theta_out,
                                                double v_flow_m3h, double alpha_c_Wm2K) {
    // (32)
    return ((theta_in + theta_out) / 2.0)
         - (C_P_AIR * RHO_AIR * v_flow_m3h * (theta_in - theta_out) / 3600.0) / (A_E_HEX * alpha_c_Wm2K);
}
double DuctCentralModel::heatingSurfaceTempEval(double theta_in, double theta_out,
                                                double v_flow_m3h, double alpha_c_Wm2K) {
    // (31)
    return ((theta_in + theta_out) / 2.0)
         + (C_P_AIR * RHO_AIR * v_flow_m3h * (theta_out - theta_in) / 3600.0) / (A_E_HEX * alpha_c_Wm2K);
}

// ---- 冷媒温度（冷房） ----
double DuctCentralModel::coolingCondensingTemp_mid_rtd(double theta_evp) {
    double t = std::max(35.0 + 27.4 - 1.35 * theta_evp, 35.0);
    t = std::min(t, 65.0);
    t = std::max(t, theta_evp + 5.0);
    return t;
}
double DuctCentralModel::coolingCondensingTemp_eval(double theta_ex, double theta_evp) {
    double t = std::max(theta_ex + 27.4 - 1.35 * theta_evp, theta_ex);
    t = std::min(t, 65.0);
    t = std::max(t, theta_evp + 5.0);
    return t;
}
double DuctCentralModel::coolingSubcooling(double theta_cnd) {
    return std::max(0.772 * theta_cnd - 25.6, 0.0);
}
double DuctCentralModel::coolingSuperheating(double theta_cnd) {
    return std::max(0.194 * theta_cnd - 3.86, 0.0);
}

// ---- 冷媒温度（暖房） ----
double DuctCentralModel::heatingCondensingTemp_from_surface(double theta_surface) {
    return (theta_surface > 65.0) ? 65.0 : theta_surface;
}
double DuctCentralModel::heatingEvaporatingTemp_mid_rtd(double theta_cnd) {
    double t = 7.0 - (0.100 * theta_cnd + 2.95);
    if (t < -50.0) t = -50.0;
    if (t > theta_cnd - 5.0) t = theta_cnd - 5.0;
    return t;
}
double DuctCentralModel::heatingEvaporatingTemp_eval(double theta_ex, double theta_cnd) {
    double t = theta_ex - (0.100 * theta_cnd + 2.95);
    if (t < -50.0) t = -50.0;
    if (t > theta_cnd - 5.0) t = theta_cnd - 5.0;
    return t;
}
double DuctCentralModel::heatingSubcooling(double theta_cnd) {
    return 0.245 * theta_cnd - 1.72;
}
double DuctCentralModel::heatingSuperheating(double theta_cnd) {
    return 4.49 - 0.036 * theta_cnd;
}

// ---- 効率比（区分線形） ----
double DuctCentralModel::efficiencyRatioPiecewise(double q, double q_min, double q_mid, double q_rtd,
                                                  double e_r_min, double e_r_mid, double e_r_rtd) {
    if (q <= q_min) {
        return e_r_min - (q_min - q) * (e_r_min / q_min);
    }
    if (q > q_min && q <= q_mid) {
        return e_r_mid - (q_mid - q) * ((e_r_mid - e_r_min) / (q_mid - q_min));
    }
    if (q > q_mid && q <= q_rtd) {
        return e_r_rtd - (q_rtd - q) * ((e_r_rtd - e_r_mid) / (q_rtd - q_mid));
    }
    if (q > q_rtd && e_r_rtd > 0.4) {
        return std::max(e_r_rtd - (q - q_rtd) * (e_r_rtd / q_rtd), 0.4);
    }
    return e_r_rtd;
}

// ---- 飽和水蒸気圧の式(5a/5b)と飽和絶対湿度(3) ----
double DuctCentralModel::saturation_vapor_pressure_from_formula(double x_degC) {
    const double T = x_degC + 273.16; // [K]
    // 係数 (Pythonのコメントに合わせる)
    const double a1 = -6096.9385, a2 = 21.2409642,  a3 = -0.02711193,   a4 = 0.00001673952, a5 = 2.433502;
    const double b1 = -6024.5282, b2 = 29.32707,    b3 =  0.010613863, b4 = -0.000013198825, b5 = -0.49382577;

    double k;
    if (x_degC > 0.0) k = a1 / T + a2 + a3 * T + a4 * T * T + a5 * std::log(T);
    else              k = b1 / T + b2 + b3 * T + b4 * T * T + b5 * std::log(T);

    return std::exp(k); // [Pa]
}
double DuctCentralModel::X_saturated_from_surfaceT(double x_degC) {
    const double Pvs = saturation_vapor_pressure_from_formula(x_degC); // Pa
    return 0.622 * (Pvs / (F_ATM - Pvs));  // (式3)
}

// ---- 冷房（中間/定格）: 二分法で Θ_surf を解く ----
double DuctCentralModel::solveCoolingSurfaceTempByBisect(double v_flow_m3h,
                                                         double q_target_W,
                                                         double /*theta_in_degC*/) const
{
    // α'_c, α_c(mid/rtd)
    const double alpha_dash = latentHTCoeff(v_flow_m3h); // [kg/(m2·s)]
    const double alpha_c    = sensibleHTCoeffCooling_mid_rtd(v_flow_m3h); // [W/(m2·K)]

    // f(x) = q_CS(x) + q_CL(x) - q_target
    auto func = [&](double x_degC) -> double {
        // q_CS(x)
        const double term1 = (3600.0 / (2.0 * C_P_AIR * RHO_AIR * v_flow_m3h));
        const double term2 = (1.0 / (A_E_HEX * alpha_c));
        const double q_CS  = (27.0 - x_degC) / (term1 + term2); // Pythonは定数 27

        // 飽和絶対湿度 X_surf(x)（式5a/5b→式3）
        const double X_surf = X_saturated_from_surfaceT(x_degC);

        // q_CL(x)
        const double denom = (3600.0 / (2.0 * L_WTR * 1000.0 * RHO_AIR * v_flow_m3h))
                           + (1.0 / (L_WTR * 1000.0 * A_E_HEX * alpha_dash));
        const double q_CL  = std::max((0.010376 - X_surf) / denom, 0.0);

        return q_CS + q_CL - q_target_W;
    };

    // 二分法
    double lo = -273.15, hi = 99.96; // Pythonと同じ探索範囲
    double flo = func(lo), fhi = func(hi);

    // 端点で符号が同じ場合に軽く拡張（実務的安定化）
    for (int k=0; k<4 && flo*fhi>0.0; ++k) {
        lo -= 20.0; hi += 20.0;
        flo = func(lo); fhi = func(hi);
    }
    if (flo * fhi > 0.0) {
        // どうしても括れなければ、中心を返す（運用上は括れる前提）
        return (lo + hi) * 0.5;
    }

    const double tol = 1e-6;
    for (int iter=0; iter<100; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double fmid = func(mid);
        if (std::abs(fmid) < 1e-8 || (hi - lo) < tol) return mid;
        if (flo * fmid <= 0.0) { hi = mid; fhi = fmid; }
        else                   { lo = mid; flo = fmid; }
    }
    return 0.5 * (lo + hi);
}

} // namespace acmodel
