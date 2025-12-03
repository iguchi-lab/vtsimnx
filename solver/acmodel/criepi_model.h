#pragma once

#include "aircon_spec.h"
#include "refrigerant_calculator.h"
#include <map>
#include <vector>
#include <cmath>

namespace acmodel {

/**
 * @brief CRIEPIモデルを適用するエアコンクラス
 */
class CRIEPIModel : public AirconSpec {
public:
    /**
     * @brief コンストラクタ
     * @param config エアコン仕様のJSONオブジェクト
     */
    explicit CRIEPIModel(const nlohmann::json& config);
    
    /**
     * @brief デストラクタ
     */
    virtual ~CRIEPIModel();

    /**
     * @brief COP（成績係数）を推定する
     * @param mode 運転モード ("cooling" または "heating")
     * @param inputdata 入力データ
     * @return COP推定結果
     */
    COPResult estimateCOP(const std::string& mode, const InputData& inputdata) override;

    // 基底クラスの純粋仮想関数を実装
    double calculatePowerConsumption(double cooling_load, double outdoor_temp, double indoor_temp) const override;
    double calculateCoolingCapacity(double power_consumption, double outdoor_temp, double indoor_temp) const override;
    bool isValidOperatingCondition(double outdoor_temp, double indoor_temp) const override;
    std::string getModelName() const override;
    nlohmann::json getModelParameters() const override;

private:
    // CRIEPIモデル用定数
    static constexpr double BF = 0.2;              // バイパスファクター
    static constexpr double ERROR_THRESHOLD = 1e-3; // 温度の許容誤差
    static constexpr double MAX_TEMP = 50.0;        // 最大温度探索範囲
    
    // JIS標準条件（簡略化版）
    static constexpr double T_C_IN = 27.0;   // 冷房時室内温度
    static constexpr double T_C_EX = 35.0;   // 冷房時外気温度
    static constexpr double T_H_IN = 20.0;   // 暖房時室内温度
    static constexpr double T_H_EX = 7.0;    // 暖房時外気温度
    static constexpr double X_C_IN = 0.010366896873858594; // 冷房時室内絶対湿度
    static constexpr double X_C_EX = 0.014107436180745189; // 冷房時外気絶対湿度
    static constexpr double X_H_IN = 0.008523733193021446; // 暖房時室内絶対湿度
    static constexpr double X_H_EX = 0.005371979845343348; // 暖房時外気絶対湿度

    // CRIEPIモデル用係数とデータ
    std::map<std::string, std::vector<double>> coeffs_;  // 多項式係数
    std::map<std::string, double> Pc_;                  // 定数消費電力
    std::map<std::string, std::map<std::string, double>> COP_map_;   // COP値のマップ
    std::map<std::string, std::map<std::string, double>> Q_map_;     // 能力値のマップ
    std::map<std::string, std::map<std::string, double>> eta_th_map_;// 熱効率のマップ
    
    // ログメッセージ保存用
    std::vector<std::string> preparationLogs_;          // 初期化時のログ
    mutable std::vector<std::string> calculationLogs_;  // 計算時のログ（mutableでconst関数からも変更可能）

    /**
     * @brief CRIEPIモデルの準備を行い、係数を求める
     */
    void prepareCRIEPIModel();

    /**
     * @brief 熱効率を計算する
     * @param mode 運転モード
     * @param Q 能力 [W]
     * @param P 消費電力 [W]
     * @param V_inner 内部風量 [m³/s]
     * @param V_outer 外部風量 [m³/s]
     * @return 熱効率
     */
    double calculateEfficiency(const std::string& mode, double Q, double P, double V_inner, double V_outer);

    /**
     * @brief 冷房時の熱効率を計算
     * @param Q 能力 [W]
     * @param P 消費電力 [W]
     * @param V_inner 内部風量 [m³/s]
     * @param V_outer 外部風量 [m³/s]
     * @return 熱効率
     */
    double calculateCoolingEfficiency(double Q, double P, double V_inner, double V_outer);

    /**
     * @brief 暖房時の熱効率を計算
     * @param Q 能力 [W]
     * @param P 消費電力 [W]
     * @param V_inner 内部風量 [m³/s]
     * @param V_outer 外部風量 [m³/s]
     * @return 熱効率
     */
    double calculateHeatingEfficiency(double Q, double P, double V_inner, double V_outer);

    /**
     * @brief 係数RとPcを求める
     * @param COP_map COP値のマップ
     * @param Q_map 能力値のマップ
     * @param eta_th_map 熱効率のマップ
     * @return 係数ベクトルと定数消費電力のペア
     */
    std::pair<std::vector<double>, double> solveCoefficients(
        const std::map<std::string, std::map<std::string, double>>& COP_map,
        const std::map<std::string, std::map<std::string, double>>& Q_map,
        const std::map<std::string, std::map<std::string, double>>& eta_th_map);

    /**
     * @brief モード別の係数RとPcを求める
     * @param mode 運転モード
     * @param COP_map COP値のマップ
     * @param Q_map 能力値のマップ
     * @param eta_th_map 熱効率のマップ
     * @return 係数ベクトルと定数消費電力のペア
     */
    std::pair<std::vector<double>, double> solveCoefficientsForMode(
        const std::string& mode,
        const std::map<std::string, std::map<std::string, double>>& COP_map,
        const std::map<std::string, std::map<std::string, double>>& Q_map,
        const std::map<std::string, std::map<std::string, double>>& eta_th_map);

    /**
     * @brief 冷房時のCOPを推定し、消費電力も求める
     * @param inputdata 入力データ
     * @return COP推定結果
     */
    COPResult estimateCoolingCOP(const InputData& inputdata);

    /**
     * @brief 暖房時のCOPを推定し、消費電力も求める
     * @param inputdata 入力データ
     * @return COP推定結果
     */
    COPResult estimateHeatingCOP(const InputData& inputdata);

    // archenvライブラリと重複する関数宣言は削除済み（archenv関数を使用）


    /**
     * @brief 多項式フィッティング
     * @param x x座標値
     * @param y y座標値
     * @param degree 多項式の次数
     * @return 多項式の係数 [a, b, c] for ax^2 + bx + c
     */
    std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree);
};

} // namespace acmodel 