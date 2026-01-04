#include "criepi_model.h"
#include "aircon_constants.h"
#include "acmodel.h"
#include "../../archenv/include/archenv.h"  // archenvライブラリ統合
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace acmodel {

// archenvライブラリの名前空間エイリアス（Python版と同様）
namespace ae = archenv;

// Python版Constantsクラスと同等の定数定義
namespace PythonConstants {
    constexpr double BYPASS_FACTOR = 0.2;
    constexpr double ERROR_THRESHOLD = 1e-3;
    constexpr double MAX_TEMP = 50.0;
    const std::vector<std::string> MODES = {"cooling", "heating"};
    const std::vector<std::string> KEYS_CRIEPI = {"min", "rtd", "max"};
}

// Python版avoid_over_saturation関数をC++に移植
double avoidOverSaturation(double Td, double X, double threshold = PythonConstants::ERROR_THRESHOLD, 
                          double max_temp = PythonConstants::MAX_TEMP) {
    // Python版と同じアルゴリズム：等エンタルピー線に沿って温度を調整
    double enthalpy = ae::total_enthalpy_from_x(Td, X);
    double left = Td;
    double right = max_temp;

    while (right - left > threshold) {
        double mid = (left + right) / 2.0;
        
        // 同じエンタルピーでの新しい絶対湿度を計算
        // X_new = (enthalpy - C_P_AIR * mid) / (C_P_W * mid + L_WTR)
        double X_new = (enthalpy - ae::SPECIFIC_HEAT_AIR * mid) / 
                       (ae::SPECIFIC_HEAT_WATER_VAPOR * mid + ae::LATENT_HEAT_VAPORIZATION);
        
        // midでの飽和絶対湿度を計算
        double X_saturation = ae::absolute_humidity_from_vapor_pressure(ae::saturation_vapor_pressure(mid));

        if (X_saturation < X_new) {
            left = mid;
        } else {
            right = mid;
        }
    }

    // 最終判定：leftでの飽和絶対湿度 > X かどうかをチェック
    double X_sat_left = ae::absolute_humidity_from_vapor_pressure(ae::saturation_vapor_pressure(left));
    return (X_sat_left > X) ? left : right;
}

CRIEPIModel::CRIEPIModel(const nlohmann::json& config) : AirconSpec(config) {
    acmodel::log("　　　CRIEPIModel初期化: " + config.dump());
    prepareCRIEPIModel();
}

CRIEPIModel::~CRIEPIModel() {
    // デストラクタ
}

std::string CRIEPIModel::getInitializationSummary() const {
    // verbosity=1でも出せる「初期化の最終結果」だけを1行で返す
    // - Pc / 係数（cooling/heating）
    // - rtd の V_inner / V_outer（取得できる場合）
    auto getDoubleSafe = [&](const char* top, const char* mode, const char* key, double fallback) -> double {
        try {
            if (!spec_.contains(top)) return fallback;
            const auto& t = spec_.at(top);
            if (!t.contains(mode)) return fallback;
            const auto& m = t.at(mode);
            if (!m.contains(key)) return fallback;
            return m.at(key).get<double>();
        } catch (...) {
            return fallback;
        }
    };

    const double Vc_in  = getDoubleSafe("V_inner", "cooling", "rtd", 0.0);
    const double Vc_out = getDoubleSafe("V_outer", "cooling", "rtd", 0.0);
    const double Vh_in  = getDoubleSafe("V_inner", "heating", "rtd", 0.0);
    const double Vh_out = getDoubleSafe("V_outer", "heating", "rtd", 0.0);

    auto coeffStr = [&](const std::string& mode) -> std::string {
        auto it = coeffs_.find(mode);
        if (it == coeffs_.end() || it->second.size() < 3) return "N/A";
        std::ostringstream oss;
        oss << "[" << std::fixed << std::setprecision(6)
            << it->second[0] << "," << it->second[1] << "," << it->second[2] << "]";
        return oss.str();
    };
    auto pcStr = [&](const std::string& mode) -> std::string {
        auto it = Pc_.find(mode);
        if (it == Pc_.end()) return "N/A";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << it->second;
        return oss.str();
    };

    std::ostringstream summary;
    summary << "CRIEPI 初期化サマリ:"
            << " cooling Pc=" << pcStr("cooling") << "kW coeff=" << coeffStr("cooling")
            << " V_inner(rtd)=" << std::fixed << std::setprecision(6) << Vc_in
            << " V_outer(rtd)=" << std::fixed << std::setprecision(6) << Vc_out
            << " | heating Pc=" << pcStr("heating") << "kW coeff=" << coeffStr("heating")
            << " V_inner(rtd)=" << std::fixed << std::setprecision(6) << Vh_in
            << " V_outer(rtd)=" << std::fixed << std::setprecision(6) << Vh_out;
    return summary.str();
}

void CRIEPIModel::prepareCRIEPIModel() {
    // Python版AirconSpec::prepare_CRIEPI_model()と同等の処理
    
    for (const auto& mode : PythonConstants::MODES) {
        // Python版と同じく"rtd"キーを使用してV_inner, V_outerを取得
        double V_inner = spec_["V_inner"][mode]["rtd"].get<double>();
        double V_outer = spec_["V_outer"][mode]["rtd"].get<double>();
        acmodel::log("　　　データ取得: " + mode + " V_inner=" + std::to_string(V_inner) + ", V_outer=" + std::to_string(V_outer));

        for (const auto& rating : PythonConstants::KEYS_CRIEPI) {
            double Q = spec_["Q"][mode][rating].get<double>();
            double P = spec_["P"][mode][rating].get<double>();

            COP_map_[mode][rating] = Q / P;
            Q_map_[mode][rating] = Q; // kW単位のまま（Python版のcapacityと一致）
            
            // Python版_calculate_efficiency関数と同等の処理
            eta_th_map_[mode][rating] = calculateEfficiency(mode, Q, P, V_inner, V_outer);
            
            acmodel::log("　　　データ取得: " + mode + "_" + rating + " Q=" + std::to_string(Q) + "kW, P=" + std::to_string(P) + "kW, COP=" + std::to_string(Q/P));
            acmodel::log("　　　効率計算: " + mode + "_" + rating + " eta_th=" + std::to_string(eta_th_map_[mode][rating]));
        }
    }

    // Python版_solve_coefficients関数と同等の係数計算
    if (!COP_map_.empty()) {
        for (const auto& mode : PythonConstants::MODES) {
            if (COP_map_.find(mode) != COP_map_.end()) {
                auto [coeffs_mode, Pc_mode] = solveCoefficientsForMode(mode, COP_map_, Q_map_, eta_th_map_);
                coeffs_[mode] = coeffs_mode;
                Pc_[mode] = Pc_mode;
            }
        }
        
        std::string coeffs_msg = "　　　係数計算完了: ";
        for (const auto& mode : PythonConstants::MODES) {
            if (Pc_.find(mode) != Pc_.end()) {
                coeffs_msg += mode + " Pc=" + std::to_string(Pc_[mode]) + "kW, ";
            }
        }
        preparationLogs_.push_back(coeffs_msg);
        acmodel::log(coeffs_msg);
        
        // 係数の詳細情報も出力
        for (const auto& mode : PythonConstants::MODES) {
            if (coeffs_.find(mode) != coeffs_.end()) {
                acmodel::log("　　　" + mode + "係数: [" + std::to_string(coeffs_[mode][0]) + ", " + 
                             std::to_string(coeffs_[mode][1]) + ", " + std::to_string(coeffs_[mode][2]) + "]");
            }
        }
    } else {
        throw std::runtime_error("CRIEPIモデルの係数計算用データが不足しています");
    }
}

double CRIEPIModel::calculateEfficiency(const std::string& mode, double Q, double P, double V_inner, double V_outer) {
    // Python版_calculate_efficiency関数と同等の処理
    acmodel::log("　　　効率計算入力: mode=" + mode + ", Q=" + std::to_string(Q) + "kW, P=" + std::to_string(P) + "kW, V_inner=" + std::to_string(V_inner) + ", V_outer=" + std::to_string(V_outer));
                 
    if (mode == "cooling") {
        return calculateCoolingEfficiency(Q, P, V_inner, V_outer);
    } else if (mode == "heating") {
        return calculateHeatingEfficiency(Q, P, V_inner, V_outer);
    }
    
    throw std::runtime_error("CRIEPIモデルの熱効率計算: 不明なモード " + mode);
}

std::vector<double> CRIEPIModel::polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree) {
    // 最小二乗法による多項式フィッティング（numpy.polyfit互換）
    size_t n = x.size();
    if (n != y.size() || n < static_cast<size_t>(degree + 1)) {
        throw std::runtime_error("polyfitの入力データが不正です");
    }
    
    if (degree == 2 && n == 3) {
        // numpy.polyfitと同じアルゴリズム：Vandermonde行列を使用
        // A = [[x[0]^2, x[0], 1], [x[1]^2, x[1], 1], [x[2]^2, x[2], 1]]
        // b = [y[0], y[1], y[2]]
        // 解は [a, b, c] where ax^2 + bx + c = y
        
        // Vandermonde行列
        std::vector<std::vector<double>> A(3, std::vector<double>(3));
        std::vector<double> b(3);
        
        for (size_t i = 0; i < 3; i++) {
            A[i][0] = x[i] * x[i];  // x^2項
            A[i][1] = x[i];         // x項  
            A[i][2] = 1.0;          // 定数項
            b[i] = y[i];
        }
        
        // ガウス消去法で連立方程式を解く（numpy.linalg.solveと同等）
        // 前進消去
        for (int k = 0; k < 2; k++) {
            // ピボット選択
            int max_row = k;
            for (int i = k + 1; i < 3; i++) {
                if (std::abs(A[i][k]) > std::abs(A[max_row][k])) {
                    max_row = i;
                }
            }
            // 行交換
            if (max_row != k) {
                std::swap(A[k], A[max_row]);
                std::swap(b[k], b[max_row]);
            }
            
            // 消去
            for (int i = k + 1; i < 3; i++) {
                double factor = A[i][k] / A[k][k];
                for (int j = k; j < 3; j++) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
        
        // 後退代入
        std::vector<double> coeffs(3);
        coeffs[2] = b[2] / A[2][2];  // c
        coeffs[1] = (b[1] - A[1][2] * coeffs[2]) / A[1][1];  // b
        coeffs[0] = (b[0] - A[0][1] * coeffs[1] - A[0][2] * coeffs[2]) / A[0][0];  // a
        
        return coeffs;
    }
    
    throw std::runtime_error("polyfit: 未対応の次数です");
}

double CRIEPIModel::calculateCoolingEfficiency(double Q, double P, double V_inner, double V_outer) {
    // Python版_calc_cooling_efficiency関数と完全一致の実装
    
    // 単位換算：kW → W（Python版ではcapacity/powerはkW、内部計算でWを使用）
    double q = Q * 1000.0;  // W
    double power = P * 1000.0;  // W
    double i_v = V_inner;   // m³/s (C++版では既にm³/s単位)
    double o_v = V_outer;   // m³/s (C++版では既にm³/s単位)
    
    // 蒸発器側の空気質量流量計算 (C++: V_innerは既にm³/s単位)
    double M_evp = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(ae::jis::T_C_IN) * i_v;
    acmodel::log("　　　M_evp: " + std::to_string(M_evp) + ", ae::jis::T_C_IN: " + std::to_string(ae::jis::T_C_IN));
    
    // 蒸発温度計算 (Python: T_evp = T_C_IN - q / (M_evp * ae.air_specific_heat(X_C_IN)))
    double T_evp = ae::jis::T_C_IN - q / (M_evp * ae::air_specific_heat(ae::jis::X_C_IN));
    acmodel::log("　　　T_evp: " + std::to_string(T_evp) + ", ae::jis::X_C_IN: " + std::to_string(ae::jis::X_C_IN));
    
    // Python版avoid_over_saturation関数の呼び出し
    T_evp = avoidOverSaturation(T_evp, ae::jis::X_C_IN);
    acmodel::log("　　　T_evp adjusted: " + std::to_string(T_evp));
    
    // 凝縮器側の空気質量流量計算 (C++: V_outerは既にm³/s単位)
    double M_cnd = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(ae::jis::T_C_EX) * o_v;
    acmodel::log("　　　M_cnd: " + std::to_string(M_cnd));
    
    // 凝縮温度計算 (Python: T_cnd = T_C_EX + (q + P) / (M_cnd * ae.air_specific_heat(X_C_EX)))
    double T_cnd = ae::jis::T_C_EX + (q + power) / (M_cnd * ae::air_specific_heat(ae::jis::X_C_EX));
    acmodel::log("　　　T_cnd: " + std::to_string(T_cnd));

    // 効率計算 (Python: return (T_evp + 273.15) / (T_cnd - T_evp))
    return (T_evp + 273.15) / (T_cnd - T_evp);
}

double CRIEPIModel::calculateHeatingEfficiency(double Q, double P, double V_inner, double V_outer) {
    // Python版_calc_heating_efficiency関数と完全一致の実装
    
    // 単位換算：kW → W（Python版ではcapacity/powerはkW、内部計算でWを使用）
    double q = Q * 1000.0;  // W
    double power = P * 1000.0;  // W
    double i_v = V_inner;   // m³/s (C++版では既にm³/s単位)
    double o_v = V_outer;   // m³/s (C++版では既にm³/s単位)
    
    // 蒸発器側の空気質量流量計算 (C++: V_outerは既にm³/s単位)
    double M_evp = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(ae::jis::T_H_EX) * o_v;
    acmodel::log("　　　M_evp: " + std::to_string(M_evp));
    
    // 蒸発温度計算 (Python: T_evp = T_H_EX - (q - P) / (M_evp * ae.air_specific_heat(X_H_EX)))
    double T_evp = ae::jis::T_H_EX - (q - power) / (M_evp * ae::air_specific_heat(ae::jis::X_H_EX));
    acmodel::log("　　　T_evp: " + std::to_string(T_evp));
    
    // Python版avoid_over_saturation関数の呼び出し
    T_evp = avoidOverSaturation(T_evp, ae::jis::X_H_EX);
    acmodel::log("　　　T_evp adjusted: " + std::to_string(T_evp));
    
    // 凝縮器側の空気質量流量計算 (C++: V_innerは既にm³/s単位)
    double M_cnd = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(ae::jis::T_H_IN) * i_v;
    acmodel::log("　　　M_cnd: " + std::to_string(M_cnd));
    
    // 凝縮温度計算 (Python: T_cnd = T_H_IN + q / (M_cnd * ae.air_specific_heat(X_H_IN)))
    double T_cnd = ae::jis::T_H_IN + q / (M_cnd * ae::air_specific_heat(ae::jis::X_H_IN));
    acmodel::log("　　　T_cnd: " + std::to_string(T_cnd));

    // 効率計算 (Python: return (T_cnd + 273.15) / (T_cnd - T_evp))
    return (T_cnd + 273.15) / (T_cnd - T_evp);
}

std::pair<std::vector<double>, double> CRIEPIModel::solveCoefficients(
    const std::map<std::string, std::map<std::string, double>>& COP_map,
    const std::map<std::string, std::map<std::string, double>>& Q_map,
    const std::map<std::string, std::map<std::string, double>>& eta_th_map) {
    
    // 最初に見つかったモードのデータを使用（互換性のため）
    for (const auto& [mode, mode_map] : COP_map) {
        return solveCoefficientsForMode(mode, COP_map, Q_map, eta_th_map);
    }
    
    throw std::runtime_error("CRIEPIモデルの係数計算用データが不足しています");
}

std::pair<std::vector<double>, double> CRIEPIModel::solveCoefficientsForMode(
    const std::string& mode,
    const std::map<std::string, std::map<std::string, double>>& COP_map,
    const std::map<std::string, std::map<std::string, double>>& Q_map,
    const std::map<std::string, std::map<std::string, double>>& eta_th_map) {
    
    // Python版_solve_coefficients関数と完全一致の実装
    std::vector<double> coeffs(3, 0.0);  // 2次多項式の係数 (ax^2 + bx + c)
    double Pc = 0.0;
    
    const auto& COP = COP_map.at(mode);
    const auto& Q = Q_map.at(mode);
    const auto& eta_th = eta_th_map.at(mode);
    
    // Python版：連立方程式を解く X * [R_minrtd, Pc] = Y
    // X = [[1/eta_th["min"], 1/Q["min"]], [1/eta_th["rtd"], 1/Q["rtd"]]]
    // Y = [[1/COP["min"]], [1/COP["rtd"]]]
    
    // numpy.linalg.solve(X_matrix, Y_matrix)と同等の計算
    double a11 = 1.0 / eta_th.at("min");
    double a12 = 1.0 / Q.at("min");
    double a21 = 1.0 / eta_th.at("rtd");
    double a22 = 1.0 / Q.at("rtd");
    
    double b1 = 1.0 / COP.at("min");
    double b2 = 1.0 / COP.at("rtd");
    
    // 2x2連立方程式の解（クラメルの公式）
    double det = a11 * a22 - a12 * a21;
    if (std::abs(det) < 1e-12) {
        throw std::runtime_error("CRIEPIモデル係数計算: 行列が特異です");
    }
    
    double R_minrtd_inv = (a22 * b1 - a12 * b2) / det;  // solution[0][0]
    Pc = (a11 * b2 - a21 * b1) / det;                   // solution[1][0]
    
    double R_minrtd = 1.0 / R_minrtd_inv;
    
    // R_max = COP["max"] * Q["max"] / (Q["max"] - COP["max"] * Pc) / eta_th["max"]
    double R_max = COP.at("max") * Q.at("max") / (Q.at("max") - COP.at("max") * Pc) / eta_th.at("max");
    
    // 2次多項式フィッティング: R = ax^2 + bx + c
    // 点 (Q["min"], R_minrtd), (Q["rtd"], R_minrtd), (Q["max"], R_max)
    std::vector<double> x_values = {Q.at("min"), Q.at("rtd"), Q.at("max")};
    std::vector<double> y_values = {R_minrtd, R_minrtd, R_max};
    
    acmodel::log("　　　" + mode + "係数計算: R_minrtd=" + std::to_string(R_minrtd) + ", R_max=" + std::to_string(R_max) + ", Pc=" + std::to_string(Pc));
    acmodel::log("　　　" + mode + "x_values: [" + std::to_string(x_values[0]) + ", " + std::to_string(x_values[1]) + ", " + std::to_string(x_values[2]) + "]");
    acmodel::log("　　　" + mode + "y_values: [" + std::to_string(y_values[0]) + ", " + std::to_string(y_values[1]) + ", " + std::to_string(y_values[2]) + "]");
    
    // numpy.polyfit(x_values, y_values, 2)と同等の計算
    coeffs = polyfit(x_values, y_values, 2);
    
    return {coeffs, Pc};
}

COPResult CRIEPIModel::estimateCOP(const std::string& mode, const InputData& inputdata) {
    if (mode == "cooling") {
        return estimateCoolingCOP(inputdata);
    } else if (mode == "heating") {
        return estimateHeatingCOP(inputdata);
    }
    else{
        throw std::runtime_error("CRIEPIモデルのCOP推定用データが不足しています");
    }
}

COPResult CRIEPIModel::estimateCoolingCOP(const InputData& inputdata) {
    COPResult result;
    calculationLogs_.clear();
    
    // 初期化ログを含める
    result.logMessages = preparationLogs_;
    
    calculationLogs_.push_back("　　　冷房COP推定開始: 外気=" + std::to_string(inputdata.T_ex) + 
                               "°C, 室内=" + std::to_string(inputdata.T_in) + "°C");
    
    // 入力データの検証
    if (!isValidOperatingCondition(inputdata.T_ex, inputdata.T_in)) {
        calculationLogs_.push_back("　　　エラー: 運転条件が無効です");
        result.logMessages.insert(result.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
        result.valid = false;
        return result;
    }
    
    // 飽和超過を回避
    double T_ex_adjusted = avoidOverSaturation(inputdata.T_ex, inputdata.X_ex);
    if (T_ex_adjusted != inputdata.T_ex) {
        calculationLogs_.push_back("　　　飽和超過回避: " + std::to_string(inputdata.T_ex) + 
                                   "°C → " + std::to_string(T_ex_adjusted) + "°C");
    }
    
    // 係数を取得
    const auto& coeffs = coeffs_.at("cooling");
    double Pc = Pc_.at("cooling");
    
    // Python版と同じ反復計算によるCOP推定
    double Q = inputdata.Q / 1000.0;  // W → kW 変換（Python版と同じ単位系）
    double V_inner = inputdata.V_inner;
    double V_outer = inputdata.V_outer;
    
    // 係数Rを計算（2次多項式）- kW単位で計算
    double coeff_R = coeffs[0] * Q * Q + coeffs[1] * Q + coeffs[2];
    
    calculationLogs_.push_back("　　　係数R=" + std::to_string(coeff_R) + 
                               " (Q=" + std::to_string(Q) + "kW)");
    
    // 反復計算でCOPを求める
    double COP = 5.0; // 初期値
    const int max_iterations = 100;
    const double tolerance = 1e-3;
    
    double M_evp = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(inputdata.T_in) * V_inner;
    acmodel::log("　　　M_evp: " + std::to_string(M_evp));
    double M_cnd = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(T_ex_adjusted) * V_outer;
    acmodel::log("　　　M_cnd: " + std::to_string(M_cnd));
    
    double T_evp = inputdata.T_in - (Q * 1000.0) / (M_evp * ae::air_specific_heat(inputdata.X_in));
    acmodel::log("　　　T_evp: " + std::to_string(T_evp));
    T_evp = avoidOverSaturation(T_evp, inputdata.X_in);
    acmodel::log("　　　T_evp adjusted: " + std::to_string(T_evp));
    
    for (int i = 0; i < max_iterations; i++) {
        double T_cnd = T_ex_adjusted + ((Q * 1000.0) + (Q * 1000.0) / COP) / (M_cnd * ae::air_specific_heat(inputdata.X_ex));
        double Refrigeration_COP = coeff_R * (T_evp + 273.15) / (T_cnd - T_evp);
        double calc_COP = Refrigeration_COP * Q / (Q + Pc * Refrigeration_COP);
        
        if (std::abs(calc_COP - COP) < tolerance) {
            break;
        }
        COP = calc_COP;
        
        if (i == max_iterations - 1) {
            calculationLogs_.push_back("　　　警告: 反復計算が収束しませんでした");
        }
    }
    
    calculationLogs_.push_back("　　　反復計算完了: COP=" + std::to_string(COP));
    
    if (COP > 0) {
        result.COP = COP;
        result.power = (inputdata.Q / 1000.0) / COP; // kW単位
        result.valid = true;
        
        calculationLogs_.push_back("　　　最終結果: COP=" + std::to_string(result.COP) + 
                                   ", 電力=" + std::to_string(result.power) + "kW");
    } else {
        calculationLogs_.push_back("　　　エラー: COP計算が失敗しました");
        result.valid = false;
    }
    
    // ログメッセージを結果に含める
    result.logMessages.insert(result.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
    
    return result;
}

COPResult CRIEPIModel::estimateHeatingCOP(const InputData& inputdata) {
    COPResult result;
    calculationLogs_.clear();
    
    // 初期化ログを含める
    result.logMessages = preparationLogs_;
    
    calculationLogs_.push_back("　　　暖房COP推定開始: 外気=" + std::to_string(inputdata.T_ex) + 
                               "°C, 室内=" + std::to_string(inputdata.T_in) + "°C");
    
    // 入力データの検証
    if (!isValidOperatingCondition(inputdata.T_ex, inputdata.T_in)) {
        calculationLogs_.push_back("　　　エラー: 運転条件が無効です");
        result.logMessages.insert(result.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
        result.valid = false;
        return result;
    }
    
    // 飽和超過を回避
    double T_ex_adjusted = avoidOverSaturation(inputdata.T_ex, inputdata.X_ex);
    if (T_ex_adjusted != inputdata.T_ex) {
        calculationLogs_.push_back("　　　飽和超過回避: " + std::to_string(inputdata.T_ex) + 
                                   "°C → " + std::to_string(T_ex_adjusted) + "°C");
    }
    
    // 係数を取得
    const auto& coeffs = coeffs_.at("heating");
    double Pc = Pc_.at("heating");
    
    // Python版と同じ反復計算によるCOP推定（暖房時）
    double Q = inputdata.Q / 1000.0;  // W → kW 変換（Python版と同じ単位系）
    double V_inner = inputdata.V_inner;
    double V_outer = inputdata.V_outer;
    
    // 係数Rを計算（2次多項式）- kW単位で計算
    double coeff_R = coeffs[0] * Q * Q + coeffs[1] * Q + coeffs[2];
    
    calculationLogs_.push_back("　　　係数R=" + std::to_string(coeff_R) + 
                               " (Q=" + std::to_string(Q) + "kW)");
    
    // 反復計算でCOPを求める
    double COP = 5.0; // 初期値
    const int max_iterations = 100;
    const double tolerance = 1e-3;
    
    double M_evp = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(T_ex_adjusted) * V_outer;
    double M_cnd = (1 - PythonConstants::BYPASS_FACTOR) * ae::air_density(inputdata.T_in) * V_inner;
    
    double T_cnd = inputdata.T_in + (Q * 1000.0) / (M_cnd * ae::air_specific_heat(inputdata.X_in));
    
    for (int i = 0; i < max_iterations; i++) {
        double T_evp = T_ex_adjusted - ((Q * 1000.0) - (Q * 1000.0) / COP) / (M_evp * ae::air_specific_heat(inputdata.X_ex));
        T_evp = avoidOverSaturation(T_evp, inputdata.X_ex);
        
        double Refrigeration_COP = coeff_R * (T_cnd + 273.15) / (T_cnd - T_evp);
        double calc_COP = Refrigeration_COP * Q / (Q + Pc * Refrigeration_COP);
        
        if (std::abs(calc_COP - COP) < tolerance) {
            break;
        }
        COP = calc_COP;
        
        if (i == max_iterations - 1) {
            calculationLogs_.push_back("　　　警告: 反復計算が収束しませんでした");
        }
    }
    
    calculationLogs_.push_back("　　　反復計算完了: COP=" + std::to_string(COP));
    
    if (COP > 0) {
        result.COP = COP;
        result.power = (inputdata.Q / 1000.0) / COP; // kW単位
        result.valid = true;
        
        calculationLogs_.push_back("　　　最終結果: COP=" + std::to_string(result.COP) + 
                                   ", 電力=" + std::to_string(result.power) + "kW");
    } else {
        calculationLogs_.push_back("　　　エラー: COP計算が失敗しました");
        result.valid = false;
    }
    
    // ログメッセージを結果に含める
    result.logMessages.insert(result.logMessages.end(), calculationLogs_.begin(), calculationLogs_.end());
    
    return result;
}

// archenvライブラリと重複する関数は削除し、archenv関数を使用

double CRIEPIModel::calculatePowerConsumption(double cooling_load, double outdoor_temp, double indoor_temp) const {
    // 消費電力を計算（archenvライブラリ使用）
    InputData input;
    input.T_ex = outdoor_temp;
    input.T_in = indoor_temp;
    input.X_ex = ae::absolute_humidity(outdoor_temp, 60.0);  // 標準的な外気湿度60%
    input.X_in = ae::absolute_humidity(indoor_temp, 50.0);   // 標準的な室内湿度50%
    input.Q = cooling_load;
    input.V_inner = 500.0;  // 標準的な風量 [m³/s]
    input.V_outer = 800.0;  // 標準的な風量 [m³/s]
    
    COPResult result = const_cast<CRIEPIModel*>(this)->estimateCOP("cooling", input);
    
    if (result.valid) {
        return cooling_load / result.COP;
    }
    
    return 0.0;
}

double CRIEPIModel::calculateCoolingCapacity(double power_consumption, double outdoor_temp, double indoor_temp) const {
    // 冷却能力を計算（archenvライブラリ使用）
    InputData input;
    input.T_ex = outdoor_temp;
    input.T_in = indoor_temp;
    input.X_ex = ae::absolute_humidity(outdoor_temp, 60.0);  // 標準的な外気湿度60%
    input.X_in = ae::absolute_humidity(indoor_temp, 50.0);   // 標準的な室内湿度50%
    input.Q = power_consumption * 3000.0;  // 初期推定値（COP=3.0想定）
    input.V_inner = 500.0;  // 標準的な風量 [m³/s]
    input.V_outer = 800.0;  // 標準的な風量 [m³/s]
    
    COPResult result = const_cast<CRIEPIModel*>(this)->estimateCOP("cooling", input);
    
    if (result.valid) {
        return power_consumption * result.COP;
    }
    
    return 0.0;
}

bool CRIEPIModel::isValidOperatingCondition(double outdoor_temp, double indoor_temp) const {
    // 運転条件の妥当性をチェック
    return (outdoor_temp >= -20.0 && outdoor_temp <= 50.0 &&
            indoor_temp >= 10.0 && indoor_temp <= 35.0);
}

std::string CRIEPIModel::getModelName() const {
    return "CRIEPI";
}

nlohmann::json CRIEPIModel::getModelParameters() const {
    nlohmann::json params;
    params["model_name"] = getModelName();
    params["bypass_factor"] = PythonConstants::BYPASS_FACTOR;
    params["error_threshold"] = PythonConstants::ERROR_THRESHOLD;
    params["max_temp"] = PythonConstants::MAX_TEMP;
    
    // 係数情報
    if (!coeffs_.empty()) {
        params["coefficients"] = coeffs_;
        params["constant_power"] = Pc_;
    }
    
    return params;
}

} // namespace acmodel 