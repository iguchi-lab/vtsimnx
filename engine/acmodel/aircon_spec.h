#pragma once

#include <map>
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace acmodel {

/**
 * @brief エアコンの運転データを格納する構造体
 */
struct InputData {
    double T_in;      // 室内温度 [°C]
    double T_ex;      // 室外温度 [°C] 
    double X_in;      // 室内絶対湿度 [kg/kg(DA)]
    double X_ex;      // 室外絶対湿度 [kg/kg(DA)]
    double Q;         // 要求能力 [W]
    double Q_S;       // 顕熱負荷 [W]
    double Q_L;       // 潜熱負荷 [W]
    double V_inner;   // 内部風量 [m³/s]
    double V_outer;   // 外部風量 [m³/s]
    double V_vent = 0.0; // 換気風量 [m³/s]（DUCT_CENTRAL用、未入力時はモデル側既定値）
};

/**
 * @brief COP推定結果を格納する構造体
 */
struct COPResult {
    double COP;                  // 成績係数 [-]
    double power;                // 消費電力 [kW]
    bool valid;                  // 計算が有効かどうか
    std::vector<std::string> logMessages;  // ログメッセージ
    
    COPResult() : COP(0.0), power(0.0), valid(false) {}
    COPResult(double cop, double pwr) : COP(cop), power(pwr), valid(true) {}
};

/**
 * @brief エアコンの仕様を管理する基底クラス
 */
class AirconSpec {
public:
    /**
     * @brief コンストラクタ
     * @param spec エアコン仕様のJSONオブジェクト
     */
    explicit AirconSpec(const nlohmann::json& spec);

    /**
     * @brief デストラクタ
     */
    virtual ~AirconSpec() = default;

    /**
     * @brief COP（成績係数）を推定する純粋仮想関数
     * @param mode 運転モード ("cooling" または "heating")
     * @param inputdata 入力データ
     * @return COP推定結果
     */
    virtual COPResult estimateCOP(const std::string& mode, const InputData& inputdata) = 0;

    /**
     * @brief 電力消費量を計算する純粋仮想関数
     * @param cooling_load 冷房負荷 [W]
     * @param outdoor_temp 外気温度 [°C]
     * @param indoor_temp 室内温度 [°C]
     * @return 電力消費量 [W]
     */
    virtual double calculatePowerConsumption(double cooling_load, double outdoor_temp, double indoor_temp) const = 0;

    /**
     * @brief 冷却能力を計算する純粋仮想関数
     * @param power_consumption 消費電力 [W]
     * @param outdoor_temp 外気温度 [°C]
     * @param indoor_temp 室内温度 [°C]
     * @return 冷却能力 [W]
     */
    virtual double calculateCoolingCapacity(double power_consumption, double outdoor_temp, double indoor_temp) const = 0;

    /**
     * @brief 運転条件が有効かどうかチェックする純粋仮想関数
     * @param outdoor_temp 外気温度 [°C]
     * @param indoor_temp 室内温度 [°C]
     * @return 有効な運転条件の場合 true
     */
    virtual bool isValidOperatingCondition(double outdoor_temp, double indoor_temp) const = 0;

    /**
     * @brief モデル名を取得する純粋仮想関数
     * @return モデル名
     */
    virtual std::string getModelName() const = 0;

    /**
     * @brief モデルパラメータを取得する純粋仮想関数
     * @return モデルパラメータのJSON
     */
    virtual nlohmann::json getModelParameters() const = 0;

    /**
     * @brief モデル初期化の最終サマリ（verbosity=1 でも出したい要約）を返す
     * @return サマリ文字列（空なら出力しない）
     */
    virtual std::string getInitializationSummary() const { return ""; }

    /**
     * @brief 仕様データのバリデーションを行う
     * @return バリデーション結果（成功: true, 失敗: false）
     */
    bool validateSpec() const;

    /**
     * @brief COP（成績係数）を計算する
     */
    void calculateCOP();

protected:
    nlohmann::json spec_;                                        // エアコン仕様データ
    std::map<std::string, std::map<std::string, double>> COP_;   // 計算されたCOP値

    // 基本仕様パラメータ
    double rated_capacity;    // 定格能力 [W]
    double cop;              // 定格COP [-]
    double max_capacity;     // 最大能力 [W]
    std::string model_name;  // モデル名

    /**
     * @brief 指定されたモードとキーのCOP値を取得
     * @param mode 運転モード
     * @param key 能力キー
     * @return COP値（見つからない場合は0.0）
     */
    double getCOP(const std::string& mode, const std::string& key) const;

    /**
     * @brief 指定されたモードとキーの能力値を取得
     * @param mode 運転モード
     * @param key 能力キー
     * @return 能力値（見つからない場合は0.0）
     */
    double getCapacity(const std::string& mode, const std::string& key) const;

    /**
     * @brief 指定されたモードとキーの消費電力値を取得
     * @param mode 運転モード
     * @param key 能力キー
     * @return 消費電力値（見つからない場合は0.0）
     */
    double getPower(const std::string& mode, const std::string& key) const;

    /**
     * @brief 指定されたモードとキーの風量値を取得
     * @param volumeType 風量タイプ ("V_inner" または "V_outer")
     * @param mode 運転モード
     * @param key 能力キー
     * @return 風量値（見つからない場合は0.0）
     */
    double getVolume(const std::string& volumeType, const std::string& mode, const std::string& key) const;

    /**
     * @brief 指定されたモードとキーのファン消費電力値を取得
     * @param mode 運転モード
     * @param key 能力キー
     * @return ファン消費電力値（見つからない場合は0.0）
     */
    double getFanPower(const std::string& mode, const std::string& key) const;
};

} // namespace acmodel 