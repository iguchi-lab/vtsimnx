#include "simulation_runner.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "aircon/aircon_controller.h"
#include "utils/utils.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <tuple>
#include <fstream>
#include <ostream>
#include <string>

// 圧力変化量を計算
double calculatePressureChange(const PressureMap& oldPressures, const PressureMap& newPressures) {
    double maxChange = 0.0;
    for (const auto& [name, newPress] : newPressures) {
        auto it = oldPressures.find(name);
        double oldPress = (it != oldPressures.end()) ? it->second : 0.0;
        double change = std::abs(newPress - oldPress);
        maxChange = std::max(maxChange, change);
    }
    return maxChange;
}

// 温度変化量を計算
double calculateTemperatureChange(const TemperatureMap& oldTemperatures, const TemperatureMap& newTemperatures) {
    double maxChange = 0.0;
    for (const auto& [name, newTemp] : newTemperatures) {
        auto it = oldTemperatures.find(name);
        double oldTemp = (it != oldTemperatures.end()) ? it->second : 0.0;
        double change = std::abs(newTemp - oldTemp);
        maxChange = std::max(maxChange, change);
    }
    return maxChange;
}

// 換気・熱計算の連成を行う関数
std::tuple<PressureMap, FlowRateMap, FlowBalanceMap, TemperatureMap, HeatRateMap, HeatBalanceMap>
performCoupledCalculation(VentilationNetwork& ventNetwork,
                          ThermalNetwork& thermalNetwork,
                          const SimulationConstants& constants,
                          std::ostream& logs,
                          int& totalIterations,
                          TimingList& timings,
                          const std::string& meta) {
    PressureMap     pressureMap, prevPressureMap;
    FlowRateMap     flowRates;
    FlowBalanceMap  flowBalance;
    TemperatureMap  temperatureMap, prevTemperatureMap;
    HeatRateMap     heatRates;
    HeatBalanceMap  heatBalance;

    int iterationCount = 0;

    do {
        iterationCount++;
        totalIterations++;

        // 前回の値を保存
        prevPressureMap    = pressureMap;
        prevTemperatureMap = temperatureMap;

        ScopedLogSection iterationScope(
            logs,
            "圧力-熱計算 連成反復 " + std::to_string(iterationCount) + ":");

        // 換気計算
        if (constants.pressureCalc) {
            ScopedLogSection pressureScope(logs, "圧力計算");
            {
                ScopedTimer timer(timings, "pressure_solve_iteration", meta);
                std::tie(pressureMap, flowRates, flowBalance) =
                    ventNetwork.calculatePressure(constants, logs);
            }
            ventNetwork.updateCalculationResults(pressureMap, flowRates);

            if (iterationCount == 1 && !ventNetwork.getLastPressureConverged()) {
                writeLog(logs, "エラー: フォールバック後も未収束のため停止します（最終通常解の再試行は無効化）");
                throw std::runtime_error("Disabled final normal solve: stopping after fallback non-convergence");
            }
            // 熱計算が有効な場合、換気計算結果を熱回路網に同期
            if (constants.temperatureCalc) {
                thermalNetwork.syncFlowRatesFromVentilationNetwork(ventNetwork);
            }
        }

        // 熱計算
        if (constants.temperatureCalc) {
            ScopedLogSection thermalScope(logs, "熱計算");
            {
                ScopedTimer timer(timings, "thermal_solve_iteration", meta);
                std::tie(temperatureMap, heatRates, heatBalance) =
                    thermalNetwork.calculateTemperature(constants, logs);
            }

            // 計算結果（温度と熱流量）を更新
            ventNetwork.updateNodeTemperatures(temperatureMap);
            thermalNetwork.updateCalculationResults(temperatureMap, heatRates);
        }

        // 連成計算が必要かどうかをチェック（両方がtrueの場合のみ連成）
        bool needsCoupledCalculation = constants.pressureCalc && constants.temperatureCalc;
        
        // 連成計算が不要な場合、1回の計算後にループを抜ける
        if (!needsCoupledCalculation) {
            writeLog(logs, "圧力-熱連成計算は不要です（圧力または熱計算のみ）");
            break;
        }

        // 変化量を計算してログ出力
        double pressureChange = calculatePressureChange(prevPressureMap, pressureMap);
        double temperatureChange = calculateTemperatureChange(prevTemperatureMap, temperatureMap);
        writeLog(
            logs,
            "圧力変化量: " + std::to_string(pressureChange) +
                " Pa, 温度変化量: " + std::to_string(temperatureChange) + " K");
        
        // 収束判定（2回目以降の反復でのみ実行）
        if (iterationCount > 1) {
            if (pressureChange < constants.convergenceTolerance &&
                temperatureChange < constants.convergenceTolerance) {
                writeLog(logs, "圧力-熱計算 連成計算が収束しました (" +
                                        std::to_string(iterationCount) + "回)");
                break;
            }
        }

        // 最大反復回数に達した場合、エラーを投げて停止
        if (iterationCount >= constants.maxInnerIteration) {
            writeLog(logs, "圧力-熱計算 連成計算が最大反復回数に達しました");
            throw std::runtime_error("Maximum iteration count reached: stopping after maximum iteration count");
        }
    } while (true);

    return {pressureMap, flowRates, flowBalance, temperatureMap, heatRates, heatBalance};
}

void runSimulation(VentilationNetwork& ventNetwork,
                   ThermalNetwork& thermalNetwork,
                   AirconController& airconController,
                   const SimulationConstants& constants,
                   SimulationResults& results,
                   std::ostream& logs,
                   TimingList& timings,
                   const std::string& meta) {

    int totalIterations = 0; // 総反復回数を記録

    // エアコンの設定を適用
    airconController.applyPreset(thermalNetwork, logs);

    // 連成計算の実行
    PressureMap     pressureMap;
    FlowRateMap     flowRates;
    FlowBalanceMap  flowBalance;
    TemperatureMap  temperatureMap;
    HeatRateMap     heatRates;
    HeatBalanceMap  heatBalance;

    for (auto iteration = 0; iteration < static_cast<int>(constants.maxInnerIteration); iteration++) {
        std::string loopLabel = "圧力-温度連成計算-エアコン制御ループ " +
                                std::to_string(iteration + 1) + ":";
        bool loopConverged = false;
        {
            ScopedLogSection coupledScope(logs, loopLabel);
            {
                ScopedTimer timer(timings, "performCoupledCalculation", meta + ",iteration=" + std::to_string(iteration + 1));
                std::tie(pressureMap, flowRates, flowBalance, temperatureMap, heatRates, heatBalance) =
                    performCoupledCalculation(ventNetwork, thermalNetwork, constants, logs, totalIterations, timings, meta + ",iteration=" + std::to_string(iteration + 1));
            }

            // エアコン制御ロジック（連成計算後）
            bool allAirconControlled;
            {
                ScopedTimer timer(timings, "aircon_control", meta + ",iteration=" + std::to_string(iteration + 1));
                allAirconControlled = airconController.controlAllAircons(
                    thermalNetwork, temperatureMap, constants.thermalTolerance, logs);
            }

            if (allAirconControlled) {
                // 処理熱量チェックと設定温度調整
                bool adjustmentMade;
                {
                    ScopedTimer timer(timings, "aircon_capacity_adjust", meta + ",iteration=" + std::to_string(iteration + 1));
                    adjustmentMade = airconController.checkAndAdjustCapacity(
                        thermalNetwork, ventNetwork, constants, flowRates, temperatureMap, logs, totalIterations);
                }
                if (adjustmentMade) {
                    writeLog(logs, "エアコン制御の修正が行われました。再計算を実行します。");
                    continue;
                }
                loopConverged = true;
            }
        }
        if (loopConverged) {
            writeLog(logs,
                     "圧力-温度連成計算-エアコン制御ループ " +
                         std::to_string(iteration + 1) + " が収束しました。");
            break;   // 全てのエアコンが制御完了の場合、反復を終了
        }
    }

    // 結果を履歴に追加（1タイムステップ分のデータをまとめて追加）
    TimestepResult timestepResult;
    
    if (constants.pressureCalc) {
        timestepResult.pressureMap = pressureMap;
        timestepResult.flowRateMap = ventNetwork.collectFlowRates();
        timestepResult.flowBalanceMap = flowBalance;
    }

    if (constants.temperatureCalc) {
        timestepResult.temperatureMap = temperatureMap;
        timestepResult.heatRateMap = thermalNetwork.collectHeatRates();
        timestepResult.heatBalanceMap = heatBalance;
        
        timestepResult.airconInletTempMap = airconController.collectAirconData(thermalNetwork, flowRates, temperatureMap, "inTemp");
        timestepResult.airconOutletTempMap = airconController.collectAirconData(thermalNetwork, flowRates, temperatureMap, "airconTemp");
        timestepResult.airconFlowMap = airconController.collectAirconData(thermalNetwork, flowRates, temperatureMap, "flow");
        timestepResult.airconSensibleHeatMap = airconController.collectAirconData(thermalNetwork, flowRates, temperatureMap, "sensibleHeatCapacity");
        timestepResult.airconLatentHeatMap = airconController.collectAirconData(thermalNetwork, flowRates, temperatureMap, "latentHeatCapacity");
        timestepResult.airconPowerMap = airconController.calculatePower(thermalNetwork, flowRates, temperatureMap, logs);
        timestepResult.airconCOPMap = airconController.calculateCOP(thermalNetwork, flowRates, temperatureMap, logs);
    }
    
    // 1回のpush_backで全データを追加
    results.timestepHistory.push_back(timestepResult);

    writeLog(logs,
             "タイムステップ終了  総連成反復回数: " + std::to_string(totalIterations),
             true);
}
