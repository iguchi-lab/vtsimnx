#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "../archenv/include/archenv.h"

#ifdef AC_MODEL_AVAILABLE
#include "acmodel/acmodel.h"
#endif

#include <cmath>
#include <stdexcept>

// 外部で定義される連成計算
extern std::tuple<PressureMap, FlowRateMap, FlowBalanceMap,
                  TemperatureMap, HeatRateMap, HeatBalanceMap>
performCoupledCalculation(VentilationNetwork& ventNetwork, ThermalNetwork& thermalNetwork,
                          const SimulationConstants& constants, std::ostream& logs,
                          int& totalIterations);

namespace {
    // 動作モード判定（"cooling" / "heating"）
    inline std::string determineOperationMode(const VertexProperties& nodeProps,
                                              const AirconValidationData& validData) {
        if (nodeProps.current_mode == "HEATING") return "heating";
        if (nodeProps.current_mode == "AUTO") {
            return (validData.indoorTemp > validData.airconTemp) ? "cooling" : "heating";
        }
        return "cooling";
    }

    // 風量（絶対値）を取得
    inline double computeAirFlowRate(const VertexProperties& nodeProps,
                                     const FlowRateMap& flowRates) {
        if (nodeProps.in_node.empty()) return 0.0;
        auto flowKey = std::make_pair(nodeProps.in_node, nodeProps.key);
        auto it = flowRates.find(flowKey);
        if (it == flowRates.end()) return 0.0;
        return std::abs(it->second);
    }
}

AirconValidationData AirconController::validateAirconData(const std::string& /*airconKey*/,
                                                          const VertexProperties& nodeProps,
                                                          const TemperatureMap& temperatureMap) const {
    AirconValidationData data;

    if (nodeProps.outside_node.empty()) {
        throw std::runtime_error("outside_node が設定されていません");
    }
    auto outdoorIt = temperatureMap.find(nodeProps.outside_node);
    if (outdoorIt == temperatureMap.end()) {
        throw std::runtime_error("outside_node '" + nodeProps.outside_node + "' の温度が見つかりません");
    }
    data.outdoorTemp = outdoorIt->second;

    if (nodeProps.in_node.empty()) {
        throw std::runtime_error("in_node が設定されていません");
    }
    auto indoorIt = temperatureMap.find(nodeProps.in_node);
    if (indoorIt == temperatureMap.end()) {
        throw std::runtime_error("in_node '" + nodeProps.in_node + "' の温度が見つかりません");
    }
    data.indoorTemp = indoorIt->second;

    auto airconIt = temperatureMap.find(nodeProps.key);
    if (airconIt == temperatureMap.end()) {
        throw std::runtime_error("エアコンノード '" + nodeProps.key + "' の温度が見つかりません");
    }
    data.airconTemp = airconIt->second;

    if (!nodeProps.set_node.empty()) {
        auto setIt = temperatureMap.find(nodeProps.set_node);
        if (setIt == temperatureMap.end()) {
            throw std::runtime_error("set_node '" + nodeProps.set_node + "' の温度が見つかりません");
        }
        data.setTemp = setIt->second;
    }
    return data;
}

void AirconController::initializeModels(ThermalNetwork& thermalNetwork, std::ostream& logs) {
#ifdef AC_MODEL_AVAILABLE
    logs << "--エアコンモデル初期化開始\n";

    // acmodelのログをファイルへ
    acmodel::setLogger([&logs](const std::string& message) {
        logs << message << "\n";
    });

    const auto& graph = thermalNetwork.getGraph();
    auto vertex_range = boost::vertices(graph);
    int initializedCount = 0;

    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& nodeProps = graph[vertex];
        if (nodeProps.type == "aircon") {
            try {
                airconModels[nodeProps.key] = acmodel::AirconModelFactory::createModel(
                    nodeProps.model, nodeProps.ac_spec);
                logs << "---エアコンモデル初期化完了: " << nodeProps.key
                        << " (タイプ: " << nodeProps.model << ", 仕様: " << nodeProps.ac_spec.dump() << ")\n";
                initializedCount++;
            } catch (const std::exception& e) {
                logs << "---[ERROR] エアコンモデル初期化に失敗 " << nodeProps.key << ": " << e.what() << "\n";
            }
        }
    }
    logs << "--エアコンモデル初期化完了: " << initializedCount << "台\n";
#else
    (void)thermalNetwork;
    logs << "--エアコンモデル初期化は無効化されています（AC_MODEL_AVAILABLE未定義）。\n";
#endif
}

acmodel::AirconSpec* AirconController::getModel(const std::string& airconKey) const {
#ifdef AC_MODEL_AVAILABLE
    auto it = airconModels.find(airconKey);
    if (it != airconModels.end()) return it->second.get();
    return nullptr;
#else
    (void)airconKey;
    return nullptr;
#endif
}

double AirconController::calculateHeatCapacity(const std::string& inNode, const std::string& airconNode,
                                               const FlowRateMap& flowRates,
                                               const TemperatureMap& temperatureMap) const {
    if (inNode.empty()) return 0.0;
    auto inTempIt = temperatureMap.find(inNode);
    auto airconTempIt = temperatureMap.find(airconNode);
    auto flowKey = std::make_pair(inNode, airconNode);
    auto flowIt = flowRates.find(flowKey);
    if (inTempIt == temperatureMap.end() || airconTempIt == temperatureMap.end() || flowIt == flowRates.end()) return 0.0;
    const double tempDiff = std::abs(inTempIt->second - airconTempIt->second);
    const double massFlowRate = std::abs(flowIt->second);
    return archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * massFlowRate * tempDiff;
}

bool AirconController::controlAllAircons(ThermalNetwork& thermalNetwork,
                                         const TemperatureMap& temperatureMap,
                                         double tolerance,
                                         std::ostream& logs) const {
    auto& graph = thermalNetwork.getGraph();
    bool allAirconControlled = true;
    for (const auto& [airconKey, _] : airconModels) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        auto tempIt = temperatureMap.find(nodeProps.set_node);
        if (tempIt != temperatureMap.end()) {
            double currentTemp = tempIt->second;
            double targetTemp = nodeProps.current_pre_temp;
            auto controlResult = controlAircon(nodeProps, currentTemp, targetTemp, tolerance, logs);
            logs << controlResult.logMessage << "\n";
            if (controlResult.stateChanged) {
                allAirconControlled = false;
                nodeProps.on = controlResult.on;
                auto setNodeIt = thermalNetwork.getKeyToVertex().find(nodeProps.set_node);
                if (setNodeIt != thermalNetwork.getKeyToVertex().end()) {
                    if (controlResult.on) {
                        graph[setNodeIt->second].current_t = targetTemp;
                        graph[setNodeIt->second].calc_t = false;
                    } else {
                        graph[setNodeIt->second].calc_t = true;
                    }
                }
            }
        }
    }
    return allAirconControlled;
}

bool AirconController::checkAndAdjustCapacity(ThermalNetwork& thermalNetwork, VentilationNetwork& ventNetwork,
                                              const SimulationConstants& constants,
                                              const FlowRateMap& flowRates,
                                              const TemperatureMap& temperatureMap, std::ostream& logs,
                                              int& totalIterations) const {
#ifdef AC_MODEL_AVAILABLE
    auto& graph = thermalNetwork.getGraph();
    bool anyAdjustmentMade = false;
    for (const auto& [airconKey, _] : airconModels) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        try {
            if (!nodeProps.on) continue;
            auto validData = validateAirconData(airconKey, nodeProps, temperatureMap);
            std::string actualMode = nodeProps.current_mode;
            if (nodeProps.current_mode == "AUTO") {
                actualMode = (validData.indoorTemp > validData.airconTemp) ? "COOLING" : "HEATING";
            }
            double currentHeatCapacity = calculateHeatCapacity(nodeProps.in_node, nodeProps.key, flowRates, temperatureMap);
            if (nodeProps.ac_spec.empty()) {
                throw std::runtime_error("仕様(ac_spec)が初期化されていません");
            }
            bool capacityFound = false;
            double maxHeatCapacity = 0.0;
            std::string modeKey = (actualMode == "COOLING") ? "cooling" : "heating";
            if (nodeProps.ac_spec.contains("Q") &&
                nodeProps.ac_spec["Q"].contains(modeKey) &&
                nodeProps.ac_spec["Q"][modeKey].contains("rtd")) {
                double capacityFromSpec = nodeProps.ac_spec["Q"][modeKey]["rtd"].get<double>();
                maxHeatCapacity = capacityFromSpec * 1000.0;
                capacityFound = true;
                logs << "　　エアコン最大処理熱量を仕様から取得: " << nodeProps.key
                        << " モード=" << actualMode
                        << ", 能力=" << capacityFromSpec << "kW"
                        << ", 最大処理熱量=" << maxHeatCapacity << "W\n";
            }
            if (nodeProps.ac_spec.contains("max_heat_capacity")) {
                maxHeatCapacity = nodeProps.ac_spec["max_heat_capacity"].get<double>();
                capacityFound = true;
                logs << "　　エアコン最大処理熱量をmax_heat_capacityから取得: " << nodeProps.key
                        << " 最大処理熱量=" << maxHeatCapacity << "W\n";
            }
            if (!capacityFound) {
                throw std::runtime_error(actualMode + "能力情報が仕様から取得できません (Q." + modeKey + ".rtd または max_heat_capacity が必要)");
            }
            if (maxHeatCapacity > 0 && currentHeatCapacity > maxHeatCapacity) {
                logs << "　エアコン処理熱量超過検出: " << nodeProps.key
                        << " 現在=" << currentHeatCapacity << "W"
                        << " 最大=" << maxHeatCapacity << "W\n";

                double originalSetTemp = nodeProps.current_pre_temp;
                double currentSetTemp = validData.setTemp;
                double searchMin, searchMax;
                const double searchRange = 5.0;
                if (actualMode == "HEATING") {
                    searchMin = currentSetTemp - searchRange;
                    searchMax = currentSetTemp;
                } else if (actualMode == "COOLING") {
                    searchMin = currentSetTemp;
                    searchMax = currentSetTemp + searchRange;
                } else {
                    searchMin = currentSetTemp - searchRange;
                    searchMax = currentSetTemp + searchRange;
                }
                const double tempTolerance = 1e-9;
                const int maxSearchIterations = 50;
                int searchIteration = 0;
                double optimalSetTemp = originalSetTemp;
                logs << "　二分探索開始: 探索範囲=" << searchMin << "°C～" << searchMax << "°C (実際の動作モード: " << actualMode << ")\n";
                while (searchIteration < maxSearchIterations && (searchMax - searchMin) > tempTolerance) {
                    searchIteration++;
                    double midTemp = (searchMin + searchMax) / 2.0;
                    nodeProps.current_pre_temp = midTemp;
                    auto setNodeIt = thermalNetwork.getKeyToVertex().find(nodeProps.set_node);
                    if (setNodeIt != thermalNetwork.getKeyToVertex().end()) {
                        graph[setNodeIt->second].current_t = midTemp;
                    }
                    auto [testPressureMap, testFlowRates, testVentBalance, testTempMap, testHeatRates, testHeatBalance] =
                        performCoupledCalculation(ventNetwork, thermalNetwork, constants, logs, totalIterations);
                    double testHeatCapacity = calculateHeatCapacity(nodeProps.in_node, nodeProps.key, testFlowRates, testTempMap);
                    logs << "　探索" << searchIteration << ": 設定温度=" << midTemp
                            << "°C, 処理熱量=" << testHeatCapacity << "W\n";
                    if (testHeatCapacity <= maxHeatCapacity) {
                        optimalSetTemp = midTemp;
                        if (actualMode == "HEATING") {
                            searchMin = midTemp;
                        } else if (actualMode == "COOLING") {
                            searchMax = midTemp;
                        } else {
                            searchMin = midTemp;
                        }
                    } else {
                        if (actualMode == "HEATING") {
                            searchMax = midTemp;
                        } else if (actualMode == "COOLING") {
                            searchMin = midTemp;
                        } else {
                            searchMax = midTemp;
                        }
                    }
                }
                nodeProps.current_pre_temp = optimalSetTemp;
                auto setNodeIt2 = thermalNetwork.getKeyToVertex().find(nodeProps.set_node);
                if (setNodeIt2 != thermalNetwork.getKeyToVertex().end()) {
                    graph[setNodeIt2->second].current_t = optimalSetTemp;
                }
                logs << "　　設定温度調整完了: " << nodeProps.key
                        << " 元の設定=" << originalSetTemp << "°C"
                        << " 調整後=" << optimalSetTemp << "°C"
                        << " (探索回数=" << /* searchIteration not accessible here */ 0 << ", 動作モード: " << actualMode << ")\n";
                anyAdjustmentMade = true;
            }
        } catch (const std::exception& e) {
            logs << "　　エラー: エアコン " << airconKey << " の容量調整でエラーが発生 - " << e.what() << "\n";
        }
    }
    return anyAdjustmentMade;
#else
    (void)thermalNetwork; (void)ventNetwork; (void)constants; (void)flowRates; (void)temperatureMap; (void)logs; (void)totalIterations;
    return false;
#endif
}

AirconDataMap AirconController::collectAirconData(ThermalNetwork& thermalNetwork,
                                                  const FlowRateMap& flowRates,
                                                  const TemperatureMap& temperatureMap,
                                                  const std::string& dataType) const {
    AirconDataMap airconData;
    for (const auto& [airconKey, _] : airconModels) {
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (dataType == "airconTemp") {
            auto airconTempIt = temperatureMap.find(nodeProps.key);
            if (airconTempIt != temperatureMap.end()) {
                airconData[airconKey] = airconTempIt->second;
            }
        } else if (dataType == "inTemp") {
            if (!nodeProps.in_node.empty()) {
                auto inTempIt = temperatureMap.find(nodeProps.in_node);
                if (inTempIt != temperatureMap.end()) {
                    airconData[airconKey] = inTempIt->second;
                }
            }
        } else if (dataType == "setTemp") {
            if (!nodeProps.set_node.empty()) {
                auto setTempIt = temperatureMap.find(nodeProps.set_node);
                if (setTempIt != temperatureMap.end()) {
                    airconData[airconKey] = setTempIt->second;
                }
            }
        } else if (dataType == "flow") {
            if (!nodeProps.in_node.empty()) {
                auto flowKey = std::make_pair(nodeProps.in_node, nodeProps.key);
                auto flowIt = flowRates.find(flowKey);
                if (flowIt != flowRates.end()) {
                    airconData[airconKey] = std::abs(flowIt->second);
                }
            }
        } else if (dataType == "heatCapacity") {
            double heatCapacity = calculateHeatCapacity(nodeProps.in_node, nodeProps.key, flowRates, temperatureMap);
            if (heatCapacity > 0.0) {
                airconData[airconKey] = heatCapacity;
            }
        }
    }
    return airconData;
}

AirconDataMap AirconController::calculatePower(ThermalNetwork& thermalNetwork,
                                               const FlowRateMap& flowRates,
                                               const TemperatureMap& temperatureMap,
                                               std::ostream& logs) const {
    AirconDataMap airconPowerData;
#ifdef AC_MODEL_AVAILABLE
    for (const auto& [airconKey, _] : airconModels) {
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        try {
            if (!nodeProps.on) {
                airconPowerData[airconKey] = 0.0;
                continue;
            }
            auto validData = validateAirconData(airconKey, nodeProps, temperatureMap);
            double heatCapacity = calculateHeatCapacity(nodeProps.in_node, nodeProps.key, flowRates, temperatureMap);
            double airFlowRate = computeAirFlowRate(nodeProps, flowRates);
            logs << "　　エアコン電力計算: " << airconKey << "\n";
            std::string operationMode = "cooling";
            if (nodeProps.current_mode == "HEATING") {
                operationMode = "heating";
            } else if (nodeProps.current_mode == "AUTO") {
                operationMode = (validData.indoorTemp > validData.airconTemp) ? "cooling" : "heating";
            }
            auto* acModel = getModel(airconKey);
            if (!acModel) {
                throw std::runtime_error("初期化済みモデルが見つかりません");
            }
            acmodel::InputData inputData;
            inputData.T_ex = validData.outdoorTemp;
            inputData.T_in = validData.indoorTemp;
            inputData.X_ex = 0.0076;
            inputData.X_in = 0.0064;
            inputData.Q = heatCapacity;
            inputData.Q_S = heatCapacity;
            inputData.Q_L = 0.0;
            inputData.V_inner = airFlowRate;
            inputData.V_outer = 25.5 / 60.0;
            auto result = acModel->estimateCOP(operationMode, inputData);
            for (const auto& logMsg : result.logMessages) {
                logs << logMsg << "\n";
            }
            if (!result.valid) {
                throw std::runtime_error("電力計算が失敗しました");
            }
            double powerConsumption = result.power * 1000.0;
            logs << "　　エアコン電力計算: " << airconKey
                    << " モード=" << operationMode
                    << ", 処理熱量=" << heatCapacity << "W"
                    << ", 風量=" << airFlowRate << "m³/s"
                    << ", 外気温=" << validData.outdoorTemp << "°C"
                    << ", 室内温=" << validData.indoorTemp << "°C"
                    << ", COP=" << result.COP
                    << ", 電力=" << powerConsumption << "W\n";
            airconPowerData[airconKey] = powerConsumption;
        } catch (const std::exception& e) {
            logs << "　　エラー: エアコン " << airconKey << " - " << e.what() << "\n";
            airconPowerData[airconKey] = 0.0;
        }
    }
#else
    (void)thermalNetwork; (void)flowRates; (void)temperatureMap;
    logs << "ACモデル未リンクのため、電力計算はスキップされました。\n";
#endif
    return airconPowerData;
}

AirconDataMap AirconController::calculateCOP(ThermalNetwork& thermalNetwork,
                                             const FlowRateMap& flowRates,
                                             const TemperatureMap& temperatureMap,
                                             std::ostream& logs) const {
    AirconDataMap airconCOPData;
#ifdef AC_MODEL_AVAILABLE
    for (const auto& [airconKey, _] : airconModels) {
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        try {
            if (!nodeProps.on) {
                airconCOPData[airconKey] = 0.0;
                continue;
            }
            auto validData = validateAirconData(airconKey, nodeProps, temperatureMap);
            const double heatCapacity = calculateHeatCapacity(nodeProps.in_node, nodeProps.key, flowRates, temperatureMap);
            const double airFlowRate = computeAirFlowRate(nodeProps, flowRates);
            const std::string operationMode = determineOperationMode(nodeProps, validData);
            auto& acModelPtr = airconModels.at(airconKey);
            auto result = acModelPtr->estimateCOP(operationMode, {validData.indoorTemp, validData.outdoorTemp, 0.0064, 0.0076,
                                                                  heatCapacity, heatCapacity, 0.0, airFlowRate, 25.5 / 60.0});
            double cop = 0.0;
            if (result.valid) {
                cop = result.COP;
            }
            airconCOPData[airconKey] = cop;
        } catch (const std::exception& e) {
            logs << "　　エラー: エアコン " << airconKey
                    << " のCOP計算中にエラーが発生しました: " << e.what() << "\n";
            airconCOPData[airconKey] = 0.0;
        }
    }
#else
    (void)thermalNetwork; (void)flowRates; (void)temperatureMap;
    logs << "ACモデル未リンクのため、COP計算はスキップされました。\n";
#endif
    return airconCOPData;
}

void AirconController::applyPreset(ThermalNetwork& thermalNetwork, std::ostream& logs) const {
    [[maybe_unused]] int appliedCount = 0;
    auto& graph = thermalNetwork.getGraph();
    for (const auto& [airconKey, _] : airconModels) {
        try {
            auto& nodeProps = thermalNetwork.getNode(airconKey);
            nodeProps.on = false;
            if (!nodeProps.set_node.empty()) {
                auto setNodeIt = thermalNetwork.getKeyToVertex().find(nodeProps.set_node);
                if (setNodeIt != thermalNetwork.getKeyToVertex().end()) {
                    auto& setNodeProps = graph[setNodeIt->second];
                    setNodeProps.calc_t = true;
                    logs << "　エアコン設定（初期化）: " << nodeProps.set_node
                            << " ON/OFF=" << (nodeProps.on ? "ON" : "OFF") << "\n";
                } else {
                    logs << "警告: set_node " << nodeProps.set_node << " が熱回路網に見つかりません\n";
                }
            }
            appliedCount++;
        } catch (const std::exception& e) {
            logs << "　　エラー: エアコン " << airconKey << " の初期化でエラーが発生 - " << e.what() << "\n";
        }
    }
}


