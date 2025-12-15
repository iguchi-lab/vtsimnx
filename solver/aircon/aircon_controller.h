#pragma once

#include "vtsim_solver.h"
#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <map>
#include <stdexcept>
#include <sstream>

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

public:
    // === モデル管理 ===
    void initializeModels(ThermalNetwork& thermalNetwork, std::ostream& logs, int logVerbosity);
    acmodel::AirconSpec* getModel(const std::string& airconKey) const;
    ~AirconController();

    // === 計算関数 ===
    double calculateHeatCapacity(ThermalNetwork& thermalNetwork,
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
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン: モードOFFのため制御対象外"
                << " (現在 " << currentTemp << "°C, 目標 " << targetTemp << "°C)";
            result.logMessage = oss.str();
            return result;
        }
        bool shouldBeOn = false;
        if (nodeProps.current_mode == "HEATING") {
            shouldBeOn = (currentTemp < targetTemp + tolerance);
        } else if (nodeProps.current_mode == "COOLING") {
            shouldBeOn = (currentTemp > targetTemp - tolerance);
        } else if (nodeProps.current_mode == "AUTO") {
            shouldBeOn = (currentTemp < targetTemp + tolerance || currentTemp > targetTemp - tolerance);
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
};


