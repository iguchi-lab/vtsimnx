#include "rac_model.h"
#include "aircon_constants.h"
#include "acmodel.h"
#include "../../archenv/include/archenv.h"
#include <sstream>
#include <iomanip>

namespace ae = archenv;
namespace acmodel {

// ========= コンストラクタ/デストラクタ =========
RACModel::RACModel(const nlohmann::json& config) : AirconSpec(config) {
    dualcompressor_ = spec_.value("dualcompressor", false);

    preparationLogs_.clear();
    std::ostringstream oss;
    oss << "　　　RACModel初期化: dualcompressor="
        << (dualcompressor_ ? "ON" : "OFF");
    preparationLogs_.push_back(oss.str());
    
    // 定格仕様のログ出力
    try {
        double q_rtd_C = qrtdC_W();
        double q_max_C = qmaxC_W();
        double p_rtd_C = prtdC_W();
        double q_rtd_H = qrtdH_W();
        double q_max_H = qmaxH_W();
        double p_rtd_H = prtdH_W();
        
        preparationLogs_.push_back("　　　冷房定格: 能力=" + std::to_string(q_rtd_C/1000.0) + "kW, 最大=" + 
                                   std::to_string(q_max_C/1000.0) + "kW, 電力=" + std::to_string(p_rtd_C/1000.0) + "kW");
        preparationLogs_.push_back("　　　暖房定格: 能力=" + std::to_string(q_rtd_H/1000.0) + "kW, 最大=" + 
                                   std::to_string(q_max_H/1000.0) + "kW, 電力=" + std::to_string(p_rtd_H/1000.0) + "kW");
        
        // 補正係数のログ出力
        preparationLogs_.push_back("　　　補正係数: C_AF_C=" + std::to_string(C_AF_C) + 
                                   ", C_HM_C=" + std::to_string(C_HM_C) + 
                                   ", C_AF_H=" + std::to_string(C_AF_H));
        preparationLogs_.push_back("　　　SHF設定: SHF_L_MIN_C=" + std::to_string(SHF_L_MIN_C));
        
        acmodel::log("　　　RACModel初期化完了");
    } catch (const std::exception& e) {
        preparationLogs_.push_back("　　　警告: 仕様データの一部取得に失敗: " + std::string(e.what()));
    }
}
RACModel::~RACModel() {}

// ========= 仕様アクセス（kW→W） =========
static double getNested_kW_to_W(const nlohmann::json& j,
                                const char* a, const char* b, const char* c)
{
    if (!j.contains(a) || !j.at(a).contains(b) || !j.at(a).at(b).contains(c)) {
        std::ostringstream oss;
        oss << "missing spec: " << a << "." << b << "." << c;
        throw std::runtime_error(oss.str());
    }
    return j.at(a).at(b).at(c).get<double>() * 1000.0;
}

double RACModel::qrtdC_W() const { return getNested_kW_to_W(spec_, "Q", "cooling", "rtd"); }
double RACModel::qmaxC_W() const { return getNested_kW_to_W(spec_, "Q", "cooling", "max"); }
double RACModel::prtdC_W() const { return getNested_kW_to_W(spec_, "P", "cooling", "rtd"); }
double RACModel::qrtdH_W() const { return getNested_kW_to_W(spec_, "Q", "heating", "rtd"); }
double RACModel::qmaxH_W() const { return getNested_kW_to_W(spec_, "Q", "heating", "max"); }
double RACModel::prtdH_W() const { return getNested_kW_to_W(spec_, "P", "heating", "rtd"); }

bool RACModel::dualCompressor() const { return dualcompressor_; }

// ========= p_i（Python eq8/9/23/24） =========
double RACModel::calc_p_i_eq8(int i, double q_rtd_C) const {
    const int r = idx_row(i);
    const int c = idx_col2(i);
    const double s_i = TABLE_3[r][c];
    const double t_i = TABLE_3[r][c + 1];
    return s_i * (q_rtd_C * 1e-3) + t_i;
}
double RACModel::calc_p_i_eq9(int i, double q_rtd_C) const {
    const int r = idx_row(i);
    const int c = idx_col1(i);
    const double pA = TABLE_4_A[r][c];
    const double pB = TABLE_4_B[r][c];
    const double pC = TABLE_4_C[r][c];
    if (q_rtd_C <= 2200.0) return pA;
    if (q_rtd_C <= 4000.0) return lerp(pA, pB, (q_rtd_C - 2200.0) / (4000.0 - 2200.0));
    if (q_rtd_C <  7100.0) return lerp(pB, pC, (q_rtd_C - 4000.0) / (7100.0 - 4000.0));
    return pC;
}
double RACModel::calc_p_i_eq23(int i, double q_rtd_C) const {
    const int r = idx_row(i);
    const int c = idx_col2(i);
    q_rtd_C = std::min(5600.0, q_rtd_C);
    const double s_i = TABLE_5[r][c];
    const double t_i = TABLE_5[r][c + 1];
    return s_i * (q_rtd_C * 1e-3) + t_i;
}
double RACModel::calc_p_i_eq24(int i, double q_rtd_C) const {
    const int r = idx_row(i);
    const int c = idx_col1(i);
    const double pA = TABLE_6_A[r][c];
    const double pB = TABLE_6_B[r][c];
    const double pC = TABLE_6_C[r][c];
    if (q_rtd_C <= 2200.0) return pA;
    if (q_rtd_C <= 4000.0) return lerp(pA, pB, (q_rtd_C - 2200.0) / (4000.0 - 2200.0));
    if (q_rtd_C <  7100.0) return lerp(pB, pC, (q_rtd_C - 4000.0) / (7100.0 - 4000.0));
    return pC;
}

// ========= a0..a4（Python eq7/eq22） =========
void RACModel::calc_a_eq7(double q_rtd_C, bool dual, double Theta_ex,
                          double& a0, double& a1, double& a2, double& a3, double& a4) const
{
    auto p = [&](int i){ return dual ? calc_p_i_eq9(i, q_rtd_C) : calc_p_i_eq8(i, q_rtd_C); };
    const double p42 = p(42), p41 = p(41), p40 = p(40);
    const double p32 = p(32), p31 = p(31), p30 = p(30);
    const double p22 = p(22), p21 = p(21), p20 = p(20);
    const double p12 = p(12), p11 = p(11), p10 = p(10);
    const double p02 = p( 2), p01 = p( 1), p00 = p( 0);

    a4 = p42 * Theta_ex*Theta_ex + p41 * Theta_ex + p40;
    a3 = p32 * Theta_ex*Theta_ex + p31 * Theta_ex + p30;
    a2 = p22 * Theta_ex*Theta_ex + p21 * Theta_ex + p20;
    a1 = p12 * Theta_ex*Theta_ex + p11 * Theta_ex + p10;
    a0 = p02 * Theta_ex*Theta_ex + p01 * Theta_ex + p00;
}
void RACModel::calc_a_eq22(double Theta_ex, double q_rtd_C, bool dual,
                           double& a0, double& a1, double& a2, double& a3, double& a4) const
{
    auto p = [&](int i){ return dual ? calc_p_i_eq24(i, q_rtd_C) : calc_p_i_eq23(i, q_rtd_C); };
    const double p42 = p(42), p41 = p(41), p40 = p(40);
    const double p32 = p(32), p31 = p(31), p30 = p(30);
    const double p22 = p(22), p21 = p(21), p20 = p(20);
    const double p12 = p(12), p11 = p(11), p10 = p(10);
    const double p02 = p( 2), p01 = p( 1), p00 = p( 0);

    a4 = p42 * Theta_ex*Theta_ex + p41 * Theta_ex + p40;
    a3 = p32 * Theta_ex*Theta_ex + p31 * Theta_ex + p30;
    a2 = p22 * Theta_ex*Theta_ex + p21 * Theta_ex + p20;
    a1 = p12 * Theta_ex*Theta_ex + p11 * Theta_ex + p10;
    a0 = p02 * Theta_ex*Theta_ex + p01 * Theta_ex + p00;
}

// ========= f(x,Theta) =========
double RACModel::f_H_Theta(double x, double q_rtd_C, double Theta_ex, bool dual) const {
    double a0,a1,a2,a3,a4;
    calc_a_eq7(q_rtd_C, dual, Theta_ex, a0,a1,a2,a3,a4);
    x = clip(x, 0.0, 1.0);
    return (((a4*x + a3)*x + a2)*x + a1)*x + a0;
}
double RACModel::f_C_Theta(double x, double Theta_ex, double q_rtd_C, bool dual) const {
    double a0,a1,a2,a3,a4;
    calc_a_eq22(Theta_ex, q_rtd_C, dual, a0,a1,a2,a3,a4);
    x = clip(x, 0.0, 1.0);
    return (((a4*x + a3)*x + a2)*x + a1)*x + a0;
}

// ========= 最大出力比（Python eq3/eq13） =========
void RACModel::calc_a_eq3(double q_r_max_H, double q_rtd_C, double& a2, double& a1, double& a0) {
    const double qr = q_rtd_C * 1e-3;
    const double b2 =  0.000181 * qr - 0.000184;
    const double b1 =  0.002322 * qr + 0.013904;
    const double b0 =  0.003556 * qr + 0.993431;
    const double c2 = -0.000173 * qr + 0.000367;
    const double c1 = -0.003980 * qr + 0.003983;
    const double c0 = -0.002870 * qr + 0.006376;
    a2 = b2 * q_r_max_H + c2;
    a1 = b1 * q_r_max_H + c1;
    a0 = b0 * q_r_max_H + c0;
}
double RACModel::calc_Q_r_max_H(double q_rtd_C, double q_r_max_H, double Theta_ex) {
    double a2,a1,a0; calc_a_eq3(q_r_max_H, q_rtd_C, a2,a1,a0);
    const double t = Theta_ex - 7.0;
    return a2*t*t + a1*t + a0;
}
void RACModel::calc_a_eq13(double q_r_max_C, double q_rtd_C, double& a2, double& a1, double& a0) {
    const double qr = q_rtd_C * 1e-3;
    const double b2 =  0.000812 * qr - 0.001480;
    const double b1 =  0.003527 * qr - 0.023000;
    const double b0 = -0.011490 * qr + 1.024328;
    const double c2 = -0.000350 * qr + 0.000800;
    const double c1 = -0.001280 * qr + 0.003621;
    const double c0 =  0.004772 * qr - 0.011170;
    a2 = b2 * q_r_max_C + c2;
    a1 = b1 * q_r_max_C + c1;
    a0 = b0 * q_r_max_C + c0;
}
double RACModel::calc_Q_r_max_C(double q_r_max_C, double q_rtd_C, double Theta_ex) {
    double a2,a1,a0; calc_a_eq13(q_r_max_C, q_rtd_C, a2,a1,a0);
    const double t = Theta_ex - 35.0;
    return a2*t*t + a1*t + a0;
}

// ========= モード分岐 =========
COPResult RACModel::estimateCOP(const std::string& mode, const InputData& input) {
    if (mode == "cooling")    return estimateCoolingCOP(input);
    if (mode == "heating")    return estimateHeatingCOP(input);

    // Python版：符号で自動分岐するため、ここでは InputData の Q_S を参照
    COPResult r; r.logMessages = preparationLogs_;
    if (input.Q_S < 0.0) return estimateCoolingCOP(input);
    if (input.Q_S > 0.0) return estimateHeatingCOP(input);
    r.valid = false; r.COP = 0.0; r.power = 0.0;
    r.logMessages.push_back("　　　Q_S=0 のため計算なし");
    return r;
}

// ========= 冷房 =========
COPResult RACModel::estimateCoolingCOP(const InputData& in) {
    COPResult r; r.logMessages = preparationLogs_; calculationLogs_.clear();

    calculationLogs_.push_back("　　　RACモデル冷房COP推定開始: 外気=" + std::to_string(in.T_ex) + 
                               "°C, 室内=" + std::to_string(in.T_in) + "°C");
    
    // ---- spec（W） ----
    const double q_rtd_C = qrtdC_W();
    const double q_max_C = qmaxC_W();
    const double p_rtd_C = prtdC_W();
    
    calculationLogs_.push_back("　　　定格仕様: q_rtd_C=" + std::to_string(q_rtd_C) + "W, q_max_C=" + 
                               std::to_string(q_max_C) + "W, p_rtd_C=" + std::to_string(p_rtd_C) + "W");

    const double Theta = in.T_ex; // 外気[℃]
    const bool   dual  = dualCompressor();
    
    calculationLogs_.push_back("　　　動作設定: 外気温度=" + std::to_string(Theta) + "°C, デュアルコンプレッサー=" + 
                               (dual ? "ON" : "OFF"));

    // ---- 補正比（室内機×熱交換）----
    const double ratio = C_AF_C * C_HM_C;
    calculationLogs_.push_back("　　　補正比計算: C_AF_C=" + std::to_string(C_AF_C) + ", C_HM_C=" + 
                               std::to_string(C_HM_C) + ", ratio=" + std::to_string(ratio));

    // ---- 負荷 [MJ/h]（Python準拠。Q_S<0で冷房）----
    // W → MJ/h 変換（W * 3.6/1000 = W * 0.0036）
    const double L_CS = in.Q_S * 0.0036; // W → MJ/h
    const double L_CL = in.Q_L * 0.0036; // W → MJ/h
    
    calculationLogs_.push_back("　　　負荷計算: Q_S=" + std::to_string(in.Q_S) + "W, Q_L=" + 
                               std::to_string(in.Q_L) + "W → L_CS=" + std::to_string(L_CS) + 
                               "MJ/h, L_CL=" + std::to_string(L_CL) + "MJ/h");

    // ---- 最大冷房出力 [MJ/h] ----
    const double q_r_max_C = q_max_C / q_rtd_C;                 // 無次元
    const double Q_r_max_C = calc_Q_r_max_C(q_r_max_C, q_rtd_C, Theta);
    double Q_max_C = Q_r_max_C * q_rtd_C * ratio * 3600.0 * 1e-6; // W→MJ/h
    
    calculationLogs_.push_back("　　　最大出力計算: q_r_max_C=" + std::to_string(q_r_max_C) + 
                               ", Q_r_max_C=" + std::to_string(Q_r_max_C) + 
                               ", Q_max_C=" + std::to_string(Q_max_C) + "MJ/h");

    // ---- 潜熱上限制約・顕潜分離（Pythonそのまま）----
    const double L_max_CL  = L_CS * ((1.0 - SHF_L_MIN_C) / SHF_L_MIN_C);
    calculationLogs_.push_back("　　　潜熱上限計算: SHF_L_MIN_C=" + std::to_string(SHF_L_MIN_C) + 
                               ", L_max_CL=" + std::to_string(L_max_CL) + "MJ/h");
    
    double L_dash_CL = 0.0;
    if (L_max_CL >= 0.0) {
        L_dash_CL = std::clamp(L_CL, 0.0, L_max_CL);
        calculationLogs_.push_back("　　　潜熱負荷調整: L_CL=" + std::to_string(L_CL) + 
                                   "MJ/h → L_dash_CL=" + std::to_string(L_dash_CL) + "MJ/h");
    } else {
        // NumPyのclipにおける「上限<下限」相当の安定化：0に潰す
        L_dash_CL = 0.0;
        calculationLogs_.push_back("　　　潜熱負荷調整: L_max_CL<0のためL_dash_CL=0に設定");
    }
    const double L_dash_C  = L_CS + L_dash_CL;
    const double SHF_dash  = (L_dash_C > 0.0) ? (L_CS / L_dash_C) : 0.0;
    
    calculationLogs_.push_back("　　　調整後負荷: L_dash_C=" + std::to_string(L_dash_C) + 
                               "MJ/h, SHF_dash=" + std::to_string(SHF_dash));

    const double Q_max_CS  = Q_max_C * SHF_dash;
    const double Q_max_CL  = (L_dash_CL >= 0.0)
                           ? std::clamp(Q_max_C * (1.0 - SHF_dash), 0.0, L_dash_CL)
                           : 0.0;
    
    calculationLogs_.push_back("　　　最大処理能力: Q_max_CS=" + std::to_string(Q_max_CS) + 
                               "MJ/h, Q_max_CL=" + std::to_string(Q_max_CL) + "MJ/h");

    const double Q_T_CS    = std::min(Q_max_CS, L_CS);
    const double Q_T_CL    = std::min(Q_max_CL, L_CL);
    
    calculationLogs_.push_back("　　　実処理負荷: Q_T_CS=" + std::to_string(Q_T_CS) + 
                               "MJ/h, Q_T_CL=" + std::to_string(Q_T_CL) + "MJ/h");

    // ---- 補正処理冷房負荷 [MJ/h] ----
    // 元式（Python実装）: Q_dash_T_C = (Q_T_CS + Q_T_CL) / (C_HM_C * C_AF_C)
    const double Q_dash_T_C = (Q_T_CS + Q_T_CL) * (1.0 / (C_HM_C * C_AF_C));
    calculationLogs_.push_back("　　　補正処理負荷: Q_dash_T_C=" + std::to_string(Q_dash_T_C) + "MJ/h");

    // ---- 部分負荷電力係数 → 電力[kW] ----
    const double load_ratio_x1 = Q_dash_T_C / (q_max_C * 3600.0 * 1e-6);
    const double x1 = f_C_Theta(load_ratio_x1, Theta, q_rtd_C, dual);
    
    const double load_ratio_x2 = 1.0 / q_r_max_C;
    const double x2 = f_C_Theta(load_ratio_x2, 35.0, q_rtd_C, dual);
    
    calculationLogs_.push_back("　　　部分負荷係数計算: load_ratio_x1=" + std::to_string(load_ratio_x1) + 
                               ", x1=" + std::to_string(x1) + ", load_ratio_x2=" + std::to_string(load_ratio_x2) + 
                               ", x2=" + std::to_string(x2));

    const double E_E_C_kW = (x1 / x2) * (p_rtd_C * 1e-3);
    calculationLogs_.push_back("　　　消費電力計算: E_E_C_kW=" + std::to_string(E_E_C_kW) + "kW");

    // ---- 出力 ----
    const double L_total = (L_CS + L_CL);    // MJ/h
    const double L_total_kW = kW_from_MJh(L_total);
    
    if (E_E_C_kW > 0.0) {
        r.COP   = L_total_kW / E_E_C_kW;
        r.power = E_E_C_kW;
        r.valid = true;
        
        calculationLogs_.push_back("　　　最終結果: 総負荷=" + std::to_string(L_total_kW) + "kW (Q_S=" + 
                                   std::to_string(in.Q_S) + "W, Q_L=" + std::to_string(in.Q_L) + 
                                   "W), COP=" + std::to_string(r.COP) + ", 電力=" + std::to_string(r.power) + "kW");
    } else {
        r.COP = 0.0;
        r.power = 0.0;
        r.valid = false;
        calculationLogs_.push_back("　　　エラー: 消費電力計算が無効です");
    }
    
    // ログメッセージを結果に含める
    r.logMessages.insert(r.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
    
    return r;
}

// ========= 暖房 =========
static inline double C_df_H_func(double Theta, double h_percent) {
    return ((Theta < 5.0) && (h_percent >= 80.0)) ? 0.77 : 1.0;
}

COPResult RACModel::estimateHeatingCOP(const InputData& in) {
    COPResult r; r.logMessages = preparationLogs_; calculationLogs_.clear();

    calculationLogs_.push_back("　　　RACモデル暖房COP推定開始: 外気=" + std::to_string(in.T_ex) + 
                               "°C, 室内=" + std::to_string(in.T_in) + "°C");
    
    // ---- spec（W）----
    const double q_rtd_C = qrtdC_W(); // ※暖房式にも q_rtd_C が入る（Python準拠）
    const double q_rtd_H = qrtdH_W();
    const double q_max_H = qmaxH_W();
    const double p_rtd_H = prtdH_W();
    
    calculationLogs_.push_back("　　　定格仕様: q_rtd_C=" + std::to_string(q_rtd_C) + "W, q_rtd_H=" + 
                               std::to_string(q_rtd_H) + "W, q_max_H=" + std::to_string(q_max_H) + 
                               "W, p_rtd_H=" + std::to_string(p_rtd_H) + "W");

    const double Theta = in.T_ex; // 外気[℃]
    const bool   dual  = dualCompressor();
    
    calculationLogs_.push_back("　　　動作設定: 外気温度=" + std::to_string(Theta) + "°C, デュアルコンプレッサー=" + 
                               (dual ? "ON" : "OFF"));

    // ---- 相対湿度 h[%] ----
    // ae.saturation_vapor_pressure(Theta) -> e_sat
    // ae.absolute_humidity_from_e(e_sat)  -> X_sat（飽和絶対湿度）
    const double e_sat  = ae::saturation_vapor_pressure(Theta);
    const double X_sat  = ae::absolute_humidity_from_vapor_pressure(e_sat);
    const double h_pct  = (X_sat > 0.0) ? (in.X_ex / X_sat * 100.0) : 0.0;
    
    calculationLogs_.push_back("　　　湿度計算: X_ex=" + std::to_string(in.X_ex) + "kg/kg, X_sat=" + 
                               std::to_string(X_sat) + "kg/kg, 相対湿度=" + std::to_string(h_pct) + "%");

    // ---- 着霜・室内機補正 ----
    const double C_df_H = C_df_H_func(Theta, h_pct);
    const double ratio = C_AF_H * C_df_H;
    
    calculationLogs_.push_back("　　　補正係数計算: C_AF_H=" + std::to_string(C_AF_H) + 
                               ", C_df_H=" + std::to_string(C_df_H) + " (着霜補正), ratio=" + 
                               std::to_string(ratio));

    // ---- 暖房負荷 [MJ/h]（Python準拠。Q_S>0をそのまま）----
    const double L_H = in.Q_S * 0.0036; // W → MJ/h (W * 3.6/1000)
    calculationLogs_.push_back("　　　暖房負荷: Q_S=" + std::to_string(in.Q_S) + "W → L_H=" + 
                               std::to_string(L_H) + "MJ/h");

    // ---- 最大暖房出力 [MJ/h] ----
    const double q_r_max_H = q_max_H / q_rtd_H;
    const double Q_r_max_H = calc_Q_r_max_H(q_rtd_C, q_r_max_H, Theta);
    double Q_max_H = Q_r_max_H * q_rtd_H * ratio * 3600.0 * 1e-6;
    
    calculationLogs_.push_back("　　　最大出力計算: q_r_max_H=" + std::to_string(q_r_max_H) + 
                               ", Q_r_max_H=" + std::to_string(Q_r_max_H) + 
                               ", Q_max_H=" + std::to_string(Q_max_H) + "MJ/h");

    // ---- 処理負荷 ----
    const double Q_T_H      = std::min(Q_max_H, L_H);
    const double Q_dash_T_H = Q_T_H * (1.0 / ratio);
    
    calculationLogs_.push_back("　　　処理負荷計算: Q_T_H=" + std::to_string(Q_T_H) + 
                               "MJ/h, Q_dash_T_H=" + std::to_string(Q_dash_T_H) + "MJ/h");

    // ---- 部分負荷電力係数 → 電力[kW] ----
    const double load_ratio_x1 = Q_dash_T_H / (q_max_H * 3600.0 * 1e-6);
    const double x1 = f_H_Theta(load_ratio_x1, q_rtd_C, Theta, dual);
    
    const double load_ratio_x2 = 1.0 / q_r_max_H;
    const double x2 = f_H_Theta(load_ratio_x2, q_rtd_C, 7.0, dual);
    
    calculationLogs_.push_back("　　　部分負荷係数計算: load_ratio_x1=" + std::to_string(load_ratio_x1) + 
                               ", x1=" + std::to_string(x1) + ", load_ratio_x2=" + std::to_string(load_ratio_x2) + 
                               ", x2=" + std::to_string(x2));

    const double E_E_H_kW = (x1 / x2) * (p_rtd_H * 1e-3);
    calculationLogs_.push_back("　　　消費電力計算: E_E_H_kW=" + std::to_string(E_E_H_kW) + "kW");

    // ---- 出力 ----
    const double L_H_kW = kW_from_MJh(L_H);
    
    if (E_E_H_kW > 0.0) {
        r.COP   = L_H_kW / E_E_H_kW;
        r.power = E_E_H_kW;
        r.valid = true;
        
        calculationLogs_.push_back("　　　最終結果: 暖房負荷=" + std::to_string(L_H_kW) + "kW (" + 
                                   std::to_string(in.Q_S) + "W), COP=" + std::to_string(r.COP) + 
                                   ", 電力=" + std::to_string(r.power) + "kW");
    } else {
        r.COP = 0.0;
        r.power = 0.0;
        r.valid = false;
        calculationLogs_.push_back("　　　エラー: 消費電力計算が無効です");
    }
    
    // ログメッセージを結果に含める
    r.logMessages.insert(r.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
    
    return r;
}

// ========= パラメータ出力 =========
nlohmann::json RACModel::getModelParameters() const {
    nlohmann::json params;
    params["model_name"] = getModelName();
    params["dualcompressor"] = dualcompressor_;
    return params;
}

double RACModel::calculatePowerConsumption(double cooling_load, double outdoor_temp, double /*indoor_temp*/) const {
    // 冷却負荷から消費電力を計算
    if (cooling_load <= 0) {
        return 0.0;
    }
    
    // 基本的な効率補正係数の計算
    double efficiency_factor = 1.0;
    if (outdoor_temp > 35.0) {
        efficiency_factor = 0.8;  // 高温時の効率低下
    } else if (outdoor_temp < 5.0) {
        efficiency_factor = 0.7;  // 低温時の効率低下
    }
    
    // 定格COPを使用して消費電力を計算
    double base_cop = getCOP("cooling", "rtd");
    if (base_cop <= 0) {
        base_cop = 3.0;  // デフォルトCOP
    }
    
    double effective_cop = base_cop * efficiency_factor;
    return cooling_load / effective_cop;
}

double RACModel::calculateCoolingCapacity(double power_consumption, double outdoor_temp, double /*indoor_temp*/) const {
    // 消費電力から冷却能力を計算
    if (power_consumption <= 0) {
        return 0.0;
    }
    
    // 基本的な効率補正係数の計算
    double efficiency_factor = 1.0;
    if (outdoor_temp > 35.0) {
        efficiency_factor = 0.8;  // 高温時の効率低下
    } else if (outdoor_temp < 5.0) {
        efficiency_factor = 0.7;  // 低温時の効率低下
    }
    
    // 定格COPを使用して冷却能力を計算
    double base_cop = getCOP("cooling", "rtd");
    if (base_cop <= 0) {
        base_cop = 3.0;  // デフォルトCOP
    }
    
    double effective_cop = base_cop * efficiency_factor;
    return power_consumption * effective_cop;
}

bool RACModel::isValidOperatingCondition(double outdoor_temp, double indoor_temp) const {
    // 運転条件の妥当性をチェック
    return (outdoor_temp >= -20.0 && outdoor_temp <= 50.0 &&
            indoor_temp >= 10.0 && indoor_temp <= 35.0);
}

} // namespace acmodel
