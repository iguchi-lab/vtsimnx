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

static double calculateMaxAbsDiff(const std::vector<double>& oldValues, const std::vector<double>& newValues) {
    const size_t n = std::min(oldValues.size(), newValues.size());
    double maxChange = 0.0;
    for (size_t i = 0; i < n; ++i) {
        maxChange = std::max(maxChange, std::abs(newValues[i] - oldValues[i]));
    }
    // サイズ不一致は設計上想定しないが、念のため差分を「大きい」とみなす
    if (oldValues.size() != newValues.size()) {
        maxChange = std::max(maxChange, std::numeric_limits<double>::infinity());
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

namespace {

static inline bool needsCoupledCalculation(const SimulationConstants& constants) {
    // 両方が true のときのみ連成
    return constants.pressureCalc && constants.temperatureCalc;
}

static inline double couplingPressureTol(const SimulationConstants& constants) {
    return (constants.couplingPressureTolerance > 0.0)
               ? constants.couplingPressureTolerance
               : constants.convergenceTolerance;
}

static inline double couplingTemperatureTol(const SimulationConstants& constants) {
    return (constants.couplingTemperatureTolerance > 0.0)
               ? constants.couplingTemperatureTolerance
               : constants.convergenceTolerance;
}

static inline void capturePrevTempsByVertex(const Graph& graph, std::vector<double>& prevTempsByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    prevTempsByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        prevTempsByVertex[static_cast<size_t>(v)] = graph[v].current_t;
    }
}

struct AirconStepResult {
    bool shouldRecompute = false; // 設定温度等の調整が入り、同じ外側反復をやり直すべき
    bool allControlled = false;   // 全エアコン制御完了
};

static AirconStepResult runAirconControlAndAdjust(AirconController& airconController,
                                                  ThermalNetwork& thermalNetwork,
                                                  VentilationNetwork& ventNetwork,
                                                  const SimulationConstants& constants,
                                                  const FlowRateMap& flowRates,
                                                  std::ostream& logs,
                                                  int& totalIterations,
                                                  TimingList& timings,
                                                  const std::string& meta) {
    AirconStepResult r;
    bool allAirconControlled = false;

    // 1. 現在の温度でエアコン出力を決定し、各ノードの heat_source をリセットする
    {
        auto& graph = thermalNetwork.getGraph();
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            graph[v].heat_source = 0.0;
        }

        ScopedTimer timer(timings, "aircon_control", meta);
        allAirconControlled = airconController.controlAllAircons(
            thermalNetwork, constants.thermalTolerance, logs);
    }

    // 2. エアコンが ON の場合、必要に応じて追加の処理（現状は行列側でA案として処理されるため、ここでの heat_source 設定は不要）
    {
        // A案（行列の行入れ替え）を採用するため、以前追加した heat_source への Gain 投入は削除します。
    }

    if (!allAirconControlled) {
        r.allControlled = false;
        r.shouldRecompute = true;
        return r;
    }

    bool adjustmentMade = false;
    {
        ScopedTimer timer(timings, "aircon_capacity_adjust", meta);
        adjustmentMade = airconController.checkAndAdjustCapacity(
            thermalNetwork, ventNetwork, constants, flowRates, logs, totalIterations);
    }

    if (adjustmentMade) {
        r.shouldRecompute = true;
        r.allControlled = false;
        return r;
    }

    r.shouldRecompute = false;
    r.allControlled = true;
    return r;
}

static void buildTimestepResult(const SimulationConstants& constants,
                                VentilationNetwork& ventNetwork,
                                ThermalNetwork& thermalNetwork,
                                AirconController& airconController,
                                const FlowRateMap& flowRates,
                                std::ostream& logs,
                                TimestepResult& timestepResultOut) {
    TimestepResult timestepResult;

    if (constants.pressureCalc) {
        convertDoublesToF32(timestepResult.pressure, ventNetwork.collectPressureValues());
        convertDoublesToF32(timestepResult.flowRate, ventNetwork.collectFlowRateValues());
    }

    if (constants.temperatureCalc) {
        convertDoublesToF32(timestepResult.temperature, thermalNetwork.collectTemperatureValues());
        convertDoublesToF32(timestepResult.temperatureCapacity, thermalNetwork.collectTemperatureValuesCapacity());
        convertDoublesToF32(timestepResult.temperatureLayer, thermalNetwork.collectTemperatureValuesLayer());
        convertDoublesToF32(timestepResult.heatRateAdvection, thermalNetwork.collectHeatRateValuesAdvection());
        convertDoublesToF32(timestepResult.heatRateHeatGeneration, thermalNetwork.collectHeatRateValuesHeatGeneration());
        convertDoublesToF32(timestepResult.heatRateSolarGain, thermalNetwork.collectHeatRateValuesSolarGain());
        convertDoublesToF32(timestepResult.heatRateNocturnalLoss, thermalNetwork.collectHeatRateValuesNocturnalLoss());
        convertDoublesToF32(timestepResult.heatRateConvection, thermalNetwork.collectHeatRateValuesConvection());
        convertDoublesToF32(timestepResult.heatRateConduction, thermalNetwork.collectHeatRateValuesConduction());
        convertDoublesToF32(timestepResult.heatRateRadiation, thermalNetwork.collectHeatRateValuesRadiation());
        convertDoublesToF32(timestepResult.heatRateCapacity, thermalNetwork.collectHeatRateValuesCapacity());

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
}

} // namespace

// 換気・熱計算の連成を行う関数
std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>
performCoupledCalculation(VentilationNetwork& ventNetwork,
                          ThermalNetwork& thermalNetwork,
                          const SimulationConstants& constants,
                          std::ostream& logs,
                          int& totalIterations,
                          TimingList& timings,
                          const std::string& meta) {
    const bool logEnabled = (constants.logVerbosity > 0);
    PressureMap     pressureMap;
    FlowRateMap     flowRates;
    FlowBalanceMap  flowBalance;
    std::vector<double> prevTempsByVertex;
    std::vector<double> prevPressuresByKey;

    int iterationCount = 0;

    do {
        iterationCount++;
        totalIterations++;

        // 前回の値を保存
        if (constants.pressureCalc) {
            // map(string) をホットパスから外し、キー順固定の vector で差分を取る
            prevPressuresByKey = ventNetwork.collectPressureValues();
        }
        if (constants.temperatureCalc) {
            capturePrevTempsByVertex(thermalNetwork.getGraph(), prevTempsByVertex);
        }

        std::unique_ptr<ScopedLogSection> iterationScope;
        if (logEnabled) {
            iterationScope = std::make_unique<ScopedLogSection>(
                logs,
                "圧力-熱計算 連成反復 " + std::to_string(iterationCount) + ":");
        }

        // 換気計算
        if (constants.pressureCalc) {
            std::unique_ptr<ScopedLogSection> pressureScope;
            if (logEnabled) pressureScope = std::make_unique<ScopedLogSection>(logs, "圧力計算");
            {
                ScopedTimer timer(timings, "pressure_solve_iteration", meta);
                std::tie(pressureMap, flowRates, flowBalance) =
                    ventNetwork.calculatePressure(constants, logs);
            }
            ventNetwork.updateCalculationResults(pressureMap, flowRates);

            if (iterationCount == 1 && !ventNetwork.getLastPressureConverged()) {
                if (logEnabled) {
                    writeLog(logs, "エラー: フォールバック後も未収束のため停止します（最終通常解の再試行は無効化）");
                }
                throw std::runtime_error("Disabled final normal solve: stopping after fallback non-convergence");
            }
            // 熱計算が有効な場合、換気計算結果を熱回路網に同期
            if (constants.temperatureCalc) {
                thermalNetwork.syncFlowRatesFromVentilationNetwork(ventNetwork);
            }
        }

        // 熱計算
        if (constants.temperatureCalc) {
            // pressureCalc=false の場合でも fixed_flow 等で flow_rate が入るため、移流用に同期する
            if (!constants.pressureCalc) {
                thermalNetwork.syncFlowRatesFromVentilationNetwork(ventNetwork);
            }
            std::unique_ptr<ScopedLogSection> thermalScope;
            if (logEnabled) thermalScope = std::make_unique<ScopedLogSection>(logs, "熱計算");
            {
                ScopedTimer timer(timings, "thermal_solve_iteration", meta);
                thermalNetwork.calculateTemperature(constants, logs);
            }

            // pressureCalc=false の場合、換気側で温度（密度）を参照する計算が走らないため更新不要
            if (constants.pressureCalc) {
                ventNetwork.updateNodeTemperaturesFromThermalNetwork(thermalNetwork);
            }
        }

        // 連成計算が不要な場合、1回の計算後にループを抜ける
        if (!needsCoupledCalculation(constants)) {
            if (logEnabled) writeLog(logs, "圧力-熱連成計算は不要です（圧力または熱計算のみ）");
            break;
        }

        // 変化量を計算してログ出力
        const double pressureChange = calculateMaxAbsDiff(prevPressuresByKey, ventNetwork.collectPressureValues());
        double temperatureChange = calculateTemperatureChangeByVertex(thermalNetwork.getGraph(), prevTempsByVertex);
        if (logEnabled) {
            writeLog(
                logs,
                "圧力変化量: " + std::to_string(pressureChange) +
                    " Pa, 温度変化量: " + std::to_string(temperatureChange) + " K");
        }
        
        // 収束判定（2回目以降の反復でのみ実行）
        if (iterationCount > 1) {
            const double pTol = couplingPressureTol(constants);
            const double tTol = couplingTemperatureTol(constants);
            if (pressureChange < pTol && temperatureChange < tTol) {
                if (logEnabled) {
                    writeLog(logs, "圧力-熱計算 連成計算が収束しました (" +
                                            std::to_string(iterationCount) + "回)");
                }
                break;
            }
        }

        // 最大反復回数に達した場合、エラーを投げて停止
        if (iterationCount >= constants.maxInnerIteration) {
            if (logEnabled) writeLog(logs, "圧力-熱計算 連成計算が最大反復回数に達しました");
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

    const bool logEnabled = (constants.logVerbosity > 0);
    int totalIterations = 0; // 総反復回数を記録

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
            // pressureCalc=false の場合でも、aircon制御（処理熱量/風量/COP計算）が流量を参照できるように
            // VentilationNetwork の確定 flow_rate（fixed_flow 等）から FlowRateMap を生成する。
            if (!constants.pressureCalc) {
                flowRates = ventNetwork.collectFlowRateMap();
            }

            // エアコン制御ロジック（連成計算後）
            const auto airconRes = runAirconControlAndAdjust(
                airconController,
                thermalNetwork,
                ventNetwork,
                constants,
                flowRates,
                logs,
                totalIterations,
                timings,
                meta + ",iteration=" + std::to_string(iteration + 1));
            if (airconRes.shouldRecompute) {
                if (logEnabled) writeLog(logs, "エアコン制御の修正が行われました。再計算を実行します。");
                continue;
            }
            // 収束判定:
            // - aircon が安定していても、熱計算が未収束なら「収束しました」とは扱わない
            const bool thermalOk = thermalNetwork.getLastThermalConverged();
            if (!thermalOk) {
                // 無限ループや「誤って収束扱い」を避けるため、未収束になった時点でエラー終了する
                if (logEnabled) {
                    std::ostringstream oss;
                    oss << "　エラー: 熱計算が未収束のため停止します (method="
                        << thermalNetwork.getLastThermalMethod()
                        << ", RMSE=" << std::scientific << std::setprecision(6) << thermalNetwork.getLastThermalRmseBalance()
                        << ", maxBalance=" << std::scientific << std::setprecision(6) << thermalNetwork.getLastThermalMaxBalance()
                        << ", loop=" << (iteration + 1)
                        << ")";
                    writeLog(logs, oss.str());
                }
                throw std::runtime_error("Thermal solver did not converge: stopping to avoid infinite loop");
            }
            loopConverged = airconRes.allControlled;
        }
        if (loopConverged) {
            if (logEnabled) {
                writeLog(logs,
                         "圧力-温度連成計算-エアコン制御ループ " +
                             std::to_string(iteration + 1) + " が収束しました。");
            }
            break;   // 全てのエアコンが制御完了の場合、反復を終了
        }
    }

    // 1タイムステップ分の結果を構築（呼び出し側で即座に書き出す想定）
    buildTimestepResult(constants, ventNetwork, thermalNetwork, airconController, flowRates, logs, timestepResultOut);

    if (logEnabled) {
        writeLog(logs,
                 "タイムステップ終了  総連成反復回数: " + std::to_string(totalIterations),
                 true);
    }
}
