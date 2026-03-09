#pragma once

#include "vtsim_solver.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <utility>

#include "acmodel/acmodel.h"

// 前方宣言
class ThermalNetwork;
class VentilationNetwork;

/**
 * @brief エアコン制御の結果を表す構造体
 */
struct AirconControlResult {
    bool stateChanged;
    bool on;
    std::string logMessage;
};

/**
 * @brief エアコンバリデーションデータを表す構造体
 */
struct AirconValidationData {
    double outdoorTemp;
    double indoorTemp;
    double airconTemp;
    double setTemp;
    // 湿度（絶対湿度）: calc_flag.x が有効な場合に使用（無効時は0でよい）
    double outdoorX = 0.0;
    double indoorX = 0.0;
};

/**
 * @brief エアコン関連の処理を管理するクラス
 */
class AirconController {
private:
    // エアコンモデルを保存するマップ (エアコンキー -> acmodelインスタンス)
    std::unordered_map<std::string, std::unique_ptr<acmodel::AirconSpec>> airconModels;
    int logVerbosity_ = 1;

    struct RuntimeContext {
        AirconValidationData validData;
        double heatCapacity = 0.0;
        double airFlowRate = 0.0;
        std::string operationMode;
    };

    // 出力用：エアコンキー順キャッシュ（airconModels のキーを昇順で保持）
    mutable bool airconKeysCacheInitialized_ = false;
    mutable std::vector<std::string> airconKeysOrdered_;

    // 能力超過時 nullopt 用の二分探索 bracket（タイムステップごとにクリア）
    // first=T_low, second=T_high。暖房時は setpoint を下げるので T_high を更新、冷房時は T_low を更新。
    mutable std::unordered_map<std::string, std::pair<double, double>> capacityLimitBracket_;

    /**
     * @brief エアコンの基本データをバリデーションして取得する
     * @throws std::runtime_error 必要なデータが見つからない場合
     */
    AirconValidationData validateAirconData(const std::string& airconKey,
                                            ThermalNetwork& thermalNetwork,
                                            const VertexProperties& nodeProps) const;

    RuntimeContext prepareRuntimeContext(const std::string& airconKey,
                                         ThermalNetwork& thermalNetwork,
                                         const VertexProperties& nodeProps,
                                         const FlowRateMap& flowRates) const;

    /** 1台分の電力[W]とCOPを推定する。logDetail が true のときのみ WARN と詳細1行を出力。失敗時は例外。 */
    std::pair<double, double> estimatePowerAndCOPForAircon(const std::string& airconKey,
                                                           ThermalNetwork& thermalNetwork,
                                                           const VertexProperties& nodeProps,
                                                           const FlowRateMap& flowRates,
                                                           std::ostream& logs,
                                                           bool logDetail) const;

public:
    // === モデル管理 ===
    void initializeModels(ThermalNetwork& thermalNetwork, std::ostream& logs, int logVerbosity);
    acmodel::AirconSpec* getModel(const std::string& airconKey) const;
    ~AirconController();

    // === テスト用（本番コードの挙動は変えない）===
    // モデルを外部から登録できる注入口。単体テストでモックモデルを差し込むために使用する。
    void registerModelForTesting(const std::string& airconKey, std::unique_ptr<acmodel::AirconSpec> model);
    void clearModelsForTesting();

    // === 計算関数 ===
    // エアコンの処理熱量（顕熱）を計算する。
    // - 暖房/冷房どちらでも「処理熱量は +W（大きさ）」として扱う（COP推定/能力チェック用）
    // - mode は "heating" / "cooling"（prepareRuntimeContext が決める）
    double calculateHeatCapacity(ThermalNetwork& thermalNetwork,
                                 const std::string& mode,
                                 const std::string& inNode,
                                 const std::string& airconNode,
                                 const FlowRateMap& flowRates) const;

    // === 制御関数 ===
    template<typename NodeType>
    AirconControlResult controlAircon(const NodeType& nodeProps, double currentTemp,
                                      double targetTemp, double tolerance, [[maybe_unused]] std::ostream& logs) const {
        AirconControlResult result{false, nodeProps.on, ""};
        const std::string targetName = nodeProps.set_node.empty() ? nodeProps.key : nodeProps.set_node;
        if (nodeProps.current_mode == "OFF") {
            if (nodeProps.on) {
                result.stateChanged = true;
                result.on = false;
            }
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン: モードOFFのため制御対象外"
                << " (現在 " << currentTemp << "°C, 目標 " << targetTemp << "°C)";
            result.logMessage = oss.str();
            return result;
        }
        bool shouldBeOn = false;
        // NOTE:
        // 熱ソルバ側で set_node を fixed-row(T=pre_temp) にするため、
        // ONにした直後の反復では currentTemp が「ちょうど目標」になりやすい。
        // ここで「目標になったから即OFF」をやると、外側ループで ON/OFF が交互になり収束しない。
        // そのため、誤差帯(±tolerance)内では状態を維持（deadband）する。
        const double diff = currentTemp - targetTemp;
        const bool withinBand = (std::abs(diff) <= tolerance);

        if (nodeProps.current_mode == "HEATING") {
            // 暖房:
            // - 目標未満（-tolより低い）ならON
            // - 目標超過（+tolより高い）ならOFF
            // - ±tol内なら現状態を維持（収束扱い）
            if (withinBand) shouldBeOn = nodeProps.on;
            else shouldBeOn = (diff < 0.0);
        } else if (nodeProps.current_mode == "COOLING") {
            // 冷房:
            // - 目標超過（+tolより高い）ならON
            // - 目標未満（-tolより低い）ならOFF
            // - ±tol内なら現状態を維持（収束扱い）
            if (withinBand) shouldBeOn = nodeProps.on;
            else shouldBeOn = (diff > 0.0);
        } else if (nodeProps.current_mode == "AUTO") {
            // AUTO:
            // - ±tol内なら現状態を維持（収束扱い）
            // - 外れたらON（方向は後段のoperationMode判定等に委ねる）
            shouldBeOn = withinBand ? nodeProps.on : true;
        } else {
            throw std::runtime_error("エアコンのモードが不正です: " + nodeProps.current_mode);
        }
        if (shouldBeOn != nodeProps.on) {
            result.stateChanged = true;
            result.on = shouldBeOn;
            const char* transition = nodeProps.on ? "ON→OFF" : "OFF→ON";
            const char* action = result.on ? "起動" : "停止";
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン " << transition << " (" << action << ")"
                << " : 現在 " << currentTemp << "°C"
                << ", 目標 " << targetTemp << "°C";
            result.logMessage = oss.str();
        } else {
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン: "
                << (shouldBeOn ? "運転継続" : "停止維持")
                << " (現在 " << currentTemp << "°C, 目標 " << targetTemp << "°C)";
            result.logMessage = oss.str();
        }
        return result;
    }

    bool controlAllAircons(ThermalNetwork& thermalNetwork,
                           double tolerance,
                           std::ostream& logFile) const;

    bool checkAndAdjustCapacity(ThermalNetwork& thermalNetwork, VentilationNetwork& ventNetwork,
                                const SimulationConstants& constants,
                                const FlowRateMap& flowRates,
                                std::ostream& logFile,
                                int& totalIterations) const;

    // === データ収集・ログ ===
    const std::vector<std::string>& getAirconKeys() const;

    std::vector<double> collectAirconDataValues(ThermalNetwork& thermalNetwork,
                                                const FlowRateMap& flowRates,
                                                const std::string& dataType) const;

    std::vector<double> calculatePowerValues(ThermalNetwork& thermalNetwork,
                                             const FlowRateMap& flowRates,
                                             std::ostream& logFile) const;

    std::vector<double> calculateCOPValues(ThermalNetwork& thermalNetwork,
                                           const FlowRateMap& flowRates,
                                           std::ostream& logFile) const;

    // === 設定 ===
    void applyPreset(ThermalNetwork& thermalNetwork, std::ostream& logFile) const;

    /** 能力超過時の二分探索 bracket をクリア（タイムステップ先頭で呼ぶ） */
    void clearCapacityLimitBracket() const;
};


