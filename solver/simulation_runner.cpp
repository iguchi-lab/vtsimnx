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
#include <cstdint>
#include <cstring>
#include <boost/range/iterator_range.hpp>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {
// float の指数部が 0xFF（Inf/NaN）なら 0 にする（SIMD 向け）
static inline void sanitizeFiniteInplace(std::vector<float>& v) {
    if (v.empty()) return;
#if defined(__AVX2__)
    const __m256i expMask = _mm256_set1_epi32(0x7f800000);
    const __m256i expAll1 = _mm256_set1_epi32(0x7f800000);
    const __m256  zeros = _mm256_setzero_ps();
    size_t i = 0;
    const size_t n = v.size();
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(v.data() + i);
        __m256i xi = _mm256_castps_si256(x);
        __m256i exp = _mm256_and_si256(xi, expMask);
        __m256i isInfNaN = _mm256_cmpeq_epi32(exp, expAll1);
        // isInfNaN の lane が 0xFFFFFFFF のところを 0 にする
        __m256 mask = _mm256_castsi256_ps(isInfNaN);
        __m256 cleaned = _mm256_blendv_ps(x, zeros, mask);
        _mm256_storeu_ps(v.data() + i, cleaned);
    }
    for (; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v[i], sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) v[i] = 0.0f;
    }
#else
    for (float& f : v) {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) f = 0.0f;
    }
#endif
}

static inline void convertDoublesToF32(std::vector<float>& dst, const std::vector<double>& src) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
    sanitizeFiniteInplace(dst);
}
} // namespace

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
static double calculateTemperatureChangeByVertex(const Graph& graph, const std::vector<double>& prevTemps) {
    double maxChange = 0.0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t idx = static_cast<size_t>(v);
        if (idx >= prevTemps.size()) continue;
        double change = std::abs(graph[v].current_t - prevTemps[idx]);
        maxChange = std::max(maxChange, change);
    }
    return maxChange;
}

// 換気・熱計算の連成を行う関数
std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>
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
    std::vector<double> prevTempsByVertex;

    int iterationCount = 0;

    do {
        iterationCount++;
        totalIterations++;

        // 前回の値を保存
        prevPressureMap    = pressureMap;
        if (constants.temperatureCalc) {
            const auto& g = thermalNetwork.getGraph();
            const size_t vCount = static_cast<size_t>(boost::num_vertices(g));
            prevTempsByVertex.resize(vCount);
            for (auto v : boost::make_iterator_range(boost::vertices(g))) {
                prevTempsByVertex[static_cast<size_t>(v)] = g[v].current_t;
            }
        }

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
                thermalNetwork.calculateTemperature(constants, logs);
            }

            // pressureCalc=false の場合、換気側で温度（密度）を参照する計算が走らないため更新不要
            if (constants.pressureCalc) {
                ventNetwork.updateNodeTemperaturesFromThermalNetwork(thermalNetwork);
            }
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
        double temperatureChange = 0.0;
        if (constants.temperatureCalc) {
            temperatureChange = calculateTemperatureChangeByVertex(thermalNetwork.getGraph(), prevTempsByVertex);
        }
        writeLog(
            logs,
            "圧力変化量: " + std::to_string(pressureChange) +
                " Pa, 温度変化量: " + std::to_string(temperatureChange) + " K");
        
        // 収束判定（2回目以降の反復でのみ実行）
        if (iterationCount > 1) {
            double pTol = (constants.couplingPressureTolerance > 0.0)
                              ? constants.couplingPressureTolerance
                              : constants.convergenceTolerance;
            double tTol = (constants.couplingTemperatureTolerance > 0.0)
                              ? constants.couplingTemperatureTolerance
                              : constants.convergenceTolerance;
            if (pressureChange < pTol && temperatureChange < tTol) {
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

    return {pressureMap, flowRates, flowBalance};
}

void runSimulation(VentilationNetwork& ventNetwork,
                   ThermalNetwork& thermalNetwork,
                   AirconController& airconController,
                   const SimulationConstants& constants,
                   TimestepResult& timestepResultOut,
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

    for (auto iteration = 0; iteration < static_cast<int>(constants.maxInnerIteration); iteration++) {
        std::string loopLabel = "圧力-温度連成計算-エアコン制御ループ " +
                                std::to_string(iteration + 1) + ":";
        bool loopConverged = false;
        {
            ScopedLogSection coupledScope(logs, loopLabel);
            {
                ScopedTimer timer(timings, "performCoupledCalculation", meta + ",iteration=" + std::to_string(iteration + 1));
                std::tie(pressureMap, flowRates, flowBalance) =
                    performCoupledCalculation(ventNetwork, thermalNetwork, constants, logs, totalIterations, timings, meta + ",iteration=" + std::to_string(iteration + 1));
            }

            // エアコン制御ロジック（連成計算後）
            bool allAirconControlled;
            {
                ScopedTimer timer(timings, "aircon_control", meta + ",iteration=" + std::to_string(iteration + 1));
                allAirconControlled = airconController.controlAllAircons(
                    thermalNetwork, constants.thermalTolerance, logs);
            }

            if (allAirconControlled) {
                // 処理熱量チェックと設定温度調整
                bool adjustmentMade;
                {
                    ScopedTimer timer(timings, "aircon_capacity_adjust", meta + ",iteration=" + std::to_string(iteration + 1));
                    adjustmentMade = airconController.checkAndAdjustCapacity(
                        thermalNetwork, ventNetwork, constants, flowRates, logs, totalIterations);
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

    // 1タイムステップ分の結果を構築（呼び出し側で即座に書き出す想定）
    TimestepResult timestepResult;
    
    if (constants.pressureCalc) {
        convertDoublesToF32(timestepResult.pressure, ventNetwork.collectPressureValues());
        convertDoublesToF32(timestepResult.flowRate, ventNetwork.collectFlowRateValues());
    }

    if (constants.temperatureCalc) {
        convertDoublesToF32(timestepResult.temperature, thermalNetwork.collectTemperatureValues());
        convertDoublesToF32(timestepResult.heatRate, thermalNetwork.collectHeatRateValues());
        
        convertDoublesToF32(timestepResult.airconSensibleHeat,
                            airconController.collectAirconDataValues(thermalNetwork, flowRates, "sensibleHeatCapacity"));
        convertDoublesToF32(timestepResult.airconLatentHeat,
                            airconController.collectAirconDataValues(thermalNetwork, flowRates, "latentHeatCapacity"));
        convertDoublesToF32(timestepResult.airconPower,
                            airconController.calculatePowerValues(thermalNetwork, flowRates, logs));
        convertDoublesToF32(timestepResult.airconCOP,
                            airconController.calculateCOPValues(thermalNetwork, flowRates, logs));
    }
    
    timestepResultOut = std::move(timestepResult);

    writeLog(logs,
             "タイムステップ終了  総連成反復回数: " + std::to_string(totalIterations),
             true);
}
