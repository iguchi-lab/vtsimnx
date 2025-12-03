#pragma once

#include "vtsim_solver.h"
#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <map>
#include <stdexcept>

// acmodel への依存は前方宣言で最小化（外部環境が未整備でもビルド可能にする）
namespace acmodel { class AirconSpec; }

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
#ifdef AC_MODEL_AVAILABLE
    std::unordered_map<std::string, std::unique_ptr<acmodel::AirconSpec>> airconModels;
#else
    std::unordered_map<std::string, void*> airconModels;
#endif

    /**
     * @brief エアコンの基本データをバリデーションして取得する
     * @throws std::runtime_error 必要なデータが見つからない場合
     */
    AirconValidationData validateAirconData(const std::string& airconKey,
                                            const VertexProperties& nodeProps,
                                            const TemperatureMap& temperatureMap) const;

public:
    // === モデル管理 ===
    void initializeModels(ThermalNetwork& thermalNetwork, std::ostream& logs);
    acmodel::AirconSpec* getModel(const std::string& airconKey) const;

    // === 計算関数 ===
    double calculateHeatCapacity(const std::string& inNode, const std::string& airconNode,
                                 const FlowRateMap& flowRates,
                                 const TemperatureMap& temperatureMap) const;

    // === 制御関数 ===
    template<typename NodeType>
    AirconControlResult controlAircon(const NodeType& nodeProps, double currentTemp,
                                      double targetTemp, double tolerance, [[maybe_unused]] std::ostream& logs) const {
        AirconControlResult result{false, nodeProps.on, ""};
        if (nodeProps.current_mode == "OFF") {
            result.logMessage = std::string("　エアコンOFF: ") + nodeProps.set_node + " ON/OFF=" + (nodeProps.on ? "ON" : "OFF") +
                                ", set_node.current_t=" + std::to_string(currentTemp) + "°C, aircon.pre_temp=" + std::to_string(targetTemp) + "°C";
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
            result.logMessage = std::string("　エアコン") + (nodeProps.on ? "ON->OFF" : "OFF->ON") + ": " + nodeProps.set_node +
                                " ON/OFF=" + (result.on ? "ON" : "OFF") + ", set_node.current_t=" + std::to_string(currentTemp) +
                                "°C, aircon.pre_temp=" + std::to_string(targetTemp) + "°C";
        } else {
            result.logMessage = std::string("　エアコン制御") + (shouldBeOn ? "完了" : "不要") + ": " + nodeProps.set_node +
                                " ON/OFF=" + (nodeProps.on ? "ON" : "OFF") + ", set_node.current_t=" + std::to_string(currentTemp) +
                                "°C, aircon.pre_temp=" + std::to_string(targetTemp) + "°C";
        }
        return result;
    }

    bool controlAllAircons(ThermalNetwork& thermalNetwork,
                           const TemperatureMap& temperatureMap,
                           double tolerance,
                           std::ostream& logFile) const;

    bool checkAndAdjustCapacity(ThermalNetwork& thermalNetwork, VentilationNetwork& ventNetwork,
                                const SimulationConstants& constants,
                                const FlowRateMap& flowRates,
                                const TemperatureMap& temperatureMap, std::ostream& logFile,
                                int& totalIterations) const;

    // === データ収集・ログ ===
    AirconDataMap collectAirconData(ThermalNetwork& thermalNetwork,
                                    const FlowRateMap& flowRates,
                                    const TemperatureMap& temperatureMap,
                                    const std::string& dataType) const;

    AirconDataMap calculatePower(ThermalNetwork& thermalNetwork,
                                 const FlowRateMap& flowRates,
                                 const TemperatureMap& temperatureMap,
                                 std::ostream& logFile) const;

    AirconDataMap calculateCOP(ThermalNetwork& thermalNetwork,
                               const FlowRateMap& flowRates,
                               const TemperatureMap& temperatureMap,
                               std::ostream& logFile) const;

    // === 設定 ===
    void applyPreset(ThermalNetwork& thermalNetwork, std::ostream& logFile) const;
};


