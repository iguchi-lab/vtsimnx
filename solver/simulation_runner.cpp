#include "simulation_runner.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "aircon/aircon_controller.h"
#include "core/humidity/humidity_solver.h"
#include "transport/concentration_solver.h"
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

static inline bool humidityCouplingActive(const SimulationConstants& constants) {
    return constants.humidityCalc && constants.moistureCouplingEnabled;
}

static inline int activeCouplingStateCount(const SimulationConstants& constants) {
    int n = 0;
    if (constants.pressureCalc) ++n;
    if (constants.temperatureCalc) ++n;
    if (humidityCouplingActive(constants)) ++n;
    return n;
}

static inline bool needsInnerCoupledIteration(const SimulationConstants& constants) {
    // 連成対象状態量が2つ以上なら、内側反復で収束させる
    return activeCouplingStateCount(constants) >= 2;
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

static inline double couplingHumidityTol(const SimulationConstants& constants) {
    return (constants.couplingHumidityTolerance > 0.0)
               ? constants.couplingHumidityTolerance
               : constants.convergenceTolerance;
}

static inline void capturePrevTempsByVertex(const Graph& graph, std::vector<double>& prevTempsByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    prevTempsByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        prevTempsByVertex[static_cast<size_t>(v)] = graph[v].current_t;
    }
}

// タイムステップ開始時点の絶対湿度 x を保存する。
// エアコン制御ループが複数回まわっても「毎回タイムステップ開始時から x を積分し直す」ために使用する。
static inline void captureXPrevByVertex(const Graph& graph, std::vector<double>& xPrev) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    xPrev.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        xPrev[static_cast<size_t>(v)] = graph[v].current_x;
    }
}

// タイムステップ開始時点の材料側含湿状態 w を保存する。
static inline void captureWPrevByVertex(const Graph& graph, std::vector<double>& wPrev) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    wPrev.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        wPrev[static_cast<size_t>(v)] = graph[v].current_w;
    }
}

static inline void capturePrevHumidityByVertex(const Graph& graph, std::vector<double>& prevHumidityByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    prevHumidityByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        prevHumidityByVertex[static_cast<size_t>(v)] = graph[v].current_x;
    }
}

static inline void captureHeatSourceByVertex(const Graph& graph, std::vector<double>& heatSourceByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    heatSourceByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        heatSourceByVertex[static_cast<size_t>(v)] = graph[v].heat_source;
    }
}

static inline void restoreHeatSourceByVertex(Graph& graph, const std::vector<double>& heatSourceByVertex) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        if (i < heatSourceByVertex.size()) {
            graph[v].heat_source = heatSourceByVertex[i];
        }
    }
}

static inline double calculateHumidityChangeByVertex(const Graph& graph, const std::vector<double>& prevHumidityByVertex) {
    double maxChange = 0.0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t idx = static_cast<size_t>(v);
        if (idx >= prevHumidityByVertex.size()) continue;
        if (!graph[v].calc_x) continue;
        const double change = std::abs(graph[v].current_x - prevHumidityByVertex[idx]);
        maxChange = std::max(maxChange, change);
    }
    return maxChange;
}

static inline void relaxHumidityByVertex(Graph& graph,
                                         VentilationNetwork& ventNetwork,
                                         const std::vector<double>& prevHumidityByVertex,
                                         double relaxation) {
    if (!(relaxation > 0.0) || relaxation >= 1.0) return;
    const auto& vKeyToV = ventNetwork.getKeyToVertex();
    auto& vGraph = ventNetwork.getGraph();
    const double alpha = std::min(1.0, std::max(0.0, relaxation));
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t idx = static_cast<size_t>(v);
        if (idx >= prevHumidityByVertex.size()) continue;
        if (!graph[v].calc_x) continue;
        const double mixed = prevHumidityByVertex[idx] + alpha * (graph[v].current_x - prevHumidityByVertex[idx]);
        graph[v].current_x = mixed;
        graph[v].current_w = mixed;
        auto itV = vKeyToV.find(graph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_x = mixed;
        }
    }
}

// 保存した x_prev を thermal/vent 両グラフへ復元する。
// これにより、どのループ反復でも湿度計算は「タイムステップ開始時点から」スタートする。
static inline void restoreXPrevToGraph(Graph& graph, VentilationNetwork& ventNetwork,
                                       const std::vector<double>& xPrev) {
    const auto& vKeyToV = ventNetwork.getKeyToVertex();
    auto& vGraph = ventNetwork.getGraph();
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        if (i < xPrev.size()) {
            graph[v].current_x = xPrev[i];
            // vent 側にも反映（humidity_solver が vGraph も更新するため）
            auto itV = vKeyToV.find(graph[v].key);
            if (itV != vKeyToV.end()) {
                vGraph[itV->second].current_x = xPrev[i];
            }
        }
    }
}

// 保存した w_prev を thermal 側へ復元する。
static inline void restoreWPrevToGraph(Graph& graph, const std::vector<double>& wPrev) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        if (i < wPrev.size()) {
            graph[v].current_w = wPrev[i];
        }
    }
}

struct CoupledDelta {
    double pressureChange = 0.0;     // [Pa]
    double temperatureChange = 0.0;  // [K]
    double humidityChange = 0.0;     // [kg/kg(DA)]
};

// 連成計算（pressure/thermal）1回分の「確定データ」をまとめる
struct CoupledStepData {
    PressureMap pressureMap;
    FlowRateMap flowRates;
    FlowBalanceMap flowBalance;
};

static inline CoupledDelta computeCoupledDelta(const SimulationConstants& constants,
                                               VentilationNetwork& ventNetwork,
                                               ThermalNetwork& thermalNetwork,
                                               const std::vector<double>& prevPressuresByKey,
                                               const std::vector<double>& prevTempsByVertex) {
    CoupledDelta d{};
    if (constants.pressureCalc) {
        d.pressureChange = calculateMaxAbsDiff(prevPressuresByKey, ventNetwork.collectPressureValues());
    }
    if (constants.temperatureCalc) {
        d.temperatureChange = calculateTemperatureChangeByVertex(thermalNetwork.getGraph(), prevTempsByVertex);
    }
    return d;
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
    }
    // 換気回路網を構築している場合は風量を出力する（圧力収束計算をしない固定流量のみのときも固定値を出力）
    if (constants.pressureCalc || constants.temperatureCalc || constants.humidityCalc || constants.concentrationCalc) {
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

    if (constants.humidityCalc) {
        convertDoublesToF32(timestepResult.humidityX, thermalNetwork.collectHumidityValues());
    }
    if (constants.concentrationCalc) {
        convertDoublesToF32(timestepResult.concentrationC, thermalNetwork.collectConcentrationValues());
    }

    timestepResultOut = std::move(timestepResult);
}

} // namespace

// 内部: 連成計算 1 回分の結果を struct で返す（前方宣言）
static CoupledStepData
performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                              ThermalNetwork& thermalNetwork,
                              const SimulationConstants& constants,
                              std::ostream& logs,
                              int& totalIterations,
                              TimingList& timings,
                              const std::string& meta);

// 換気・熱計算の連成を行う関数（公開API互換: tuple を返す）
std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>
performCoupledCalculation(VentilationNetwork& ventNetwork,
                          ThermalNetwork& thermalNetwork,
                          const SimulationConstants& constants,
                          std::ostream& logs,
                          int& totalIterations,
                          TimingList& timings,
                          const std::string& meta) {
    CoupledStepData step = performCoupledStepCalculation(
        ventNetwork, thermalNetwork, constants, logs, totalIterations, timings, meta);
    return {step.pressureMap, step.flowRates, step.flowBalance};
}

// 換気・熱計算の「1回分」を実行する（runSimulation 側で収束反復を制御する）
static CoupledStepData
performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                              ThermalNetwork& thermalNetwork,
                              const SimulationConstants& constants,
                              std::ostream& logs,
                              int& totalIterations,
                              TimingList& timings,
                              const std::string& meta) {
    (void)totalIterations; // runSimulation 側で反復回数を管理する
    const bool logEnabled = (constants.logVerbosity > 0);
    CoupledStepData step;

        // 換気計算
        if (constants.pressureCalc) {
            std::unique_ptr<ScopedLogSection> pressureScope;
            if (logEnabled) pressureScope = std::make_unique<ScopedLogSection>(logs, "圧力計算");
            {
                ScopedTimer timer(timings, "pressure_solve_iteration", meta);
                std::tie(step.pressureMap, step.flowRates, step.flowBalance) =
                    ventNetwork.calculatePressure(constants, logs);
            }
            ventNetwork.updateCalculationResults(step.pressureMap, step.flowRates);

        // runSimulation 側の1回目チェックと同じ条件で止めたいので、ここでは totalIterations を見ない
        // （未収束フラグは solve 後に network 側に保持される）
        }

        // 熱計算
        if (constants.temperatureCalc) {
            // pressureCalc=false の場合でも fixed_flow 等で flow_rate が入るため、移流用に同期する
            if (!constants.pressureCalc) {
                thermalNetwork.syncFlowRatesFromVentilationNetwork(ventNetwork);
        } else {
            // 熱計算が有効な場合、換気計算結果を熱回路網に同期
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

    return step;
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

    // 連成計算の実行（1回分の結果をまとめて保持）
    CoupledStepData step;

    // タイムステップ開始時点の絶対湿度を保存する。
    // エアコン制御ループが複数回まわる場合に、毎回同じ出発点から x を積分し直すために必要。
    // （ループ回数に関わらず計算結果が冪等になる）
    std::vector<double> xPrevByVertex;
    std::vector<double> wPrevByVertex;
    if (constants.humidityCalc) {
        captureXPrevByVertex(thermalNetwork.getGraph(), xPrevByVertex);
        captureWPrevByVertex(thermalNetwork.getGraph(), wPrevByVertex);
    }

    for (auto iteration = 0; iteration < static_cast<int>(constants.maxInnerIteration); iteration++) {
        if (iteration == 0) {
            airconController.clearCapacityLimitBracket();
        }
        // タイムステップ内の各外側反復は、熱源を初期化してから開始する。
        for (auto v : boost::make_iterator_range(boost::vertices(thermalNetwork.getGraph()))) {
            thermalNetwork.getGraph()[v].heat_source = 0.0;
        }
        std::string loopLabel = "圧力-温度連成計算-エアコン制御ループ " +
                                std::to_string(iteration + 1) + ":";
        bool loopConverged = false;
        {
            ScopedLogSection coupledScope(logs, loopLabel);
            {
                // 連成反復（air -> thermal -> moisture -> latent_feedback の収束まで回す）
                std::vector<double> prevTempsByVertex;
                std::vector<double> prevPressuresByKey;
                std::vector<double> prevHumidityByVertex;
                std::vector<double> baseHeatSourceByVertex;
                CoupledDelta lastDelta{};
                double lastLatentAppliedW = 0.0;
                core::humidity::HumiditySolveStats lastHumiditySolveStats{};
                int coupledIter = 0;
                while (true) {
                    ++coupledIter;
                    ++totalIterations;
                    const bool humidityActive = humidityCouplingActive(constants);
                    if (coupledIter == 1) {
                        captureHeatSourceByVertex(thermalNetwork.getGraph(), baseHeatSourceByVertex);
                    }

                    // 前回の値を保存
                    if (constants.pressureCalc) {
                        prevPressuresByKey = ventNetwork.collectPressureValues();
                    }
                    if (constants.temperatureCalc) {
                        capturePrevTempsByVertex(thermalNetwork.getGraph(), prevTempsByVertex);
                    }
                    if (humidityActive) {
                        capturePrevHumidityByVertex(thermalNetwork.getGraph(), prevHumidityByVertex);
                    }

                    std::unique_ptr<ScopedLogSection> iterScope;
                    if (logEnabled) {
                        iterScope = std::make_unique<ScopedLogSection>(
                            logs,
                            "空気-熱-湿気 連成反復 " + std::to_string(coupledIter) + ":");
                    }

                    {
                        ScopedTimer timer(timings, "performCoupledCalculation",
                                          meta + ",iteration=" + std::to_string(iteration + 1));
                        step = performCoupledStepCalculation(ventNetwork, thermalNetwork, constants, logs, totalIterations, timings,
                                                             meta + ",iteration=" + std::to_string(iteration + 1));
                    }
                    if (!constants.pressureCalc) {
                        step.flowRates = ventNetwork.collectFlowRateMap();
                    }

                    if (humidityActive) {
                        // 同一タイムステップ内反復なので、毎回 x_prev / w_prev に戻して再評価する。
                        restoreXPrevToGraph(thermalNetwork.getGraph(), ventNetwork, xPrevByVertex);
                        restoreWPrevToGraph(thermalNetwork.getGraph(), wPrevByVertex);
                        lastHumiditySolveStats = core::humidity::updateHumidityIfEnabled(
                            constants, ventNetwork, thermalNetwork, step.flowRates, logs, timings,
                            meta + ",iteration=" + std::to_string(iteration + 1) +
                                       ",coupledIter=" + std::to_string(coupledIter));
                        if (logEnabled && !lastHumiditySolveStats.converged) {
                            writeLog(
                                logs,
                                "湿気ソルバ未収束(内側反復継続): iter=" + std::to_string(lastHumiditySolveStats.iterations) +
                                    ", maxDiff=" + std::to_string(lastHumiditySolveStats.finalMaxDiff) +
                                    ", active=" + std::to_string(lastHumiditySolveStats.activeVertices));
                        }
                        relaxHumidityByVertex(thermalNetwork.getGraph(), ventNetwork, prevHumidityByVertex, constants.humidityRelaxation);
                    }

                    // 潜熱フィードバックは内側反復ごとに「基準熱源 + 今回分」で再構成する。
                    // これにより反復回数依存の熱源積み上がりを防ぐ。
                    restoreHeatSourceByVertex(thermalNetwork.getGraph(), baseHeatSourceByVertex);
                    const auto latentStats = airconController.applyLatentFeedbackToThermal(
                        thermalNetwork, step.flowRates, constants.latentRelaxation, logs);
                    lastLatentAppliedW = latentStats.maxAppliedHeatW;

                    // 1回目で pressure が未収束なら停止（従来と同じ）
                    if (constants.pressureCalc && coupledIter == 1 && !ventNetwork.getLastPressureConverged()) {
                        if (logEnabled) {
                            writeLog(logs, "エラー: フォールバック後も未収束のため停止します（最終通常解の再試行は無効化）");
                        }
                        throw std::runtime_error("Disabled final normal solve: stopping after fallback non-convergence");
                    }

                    // 連成計算が不要な場合、1回の計算後に抜ける
                    if (!needsInnerCoupledIteration(constants)) {
                        if (logEnabled) writeLog(logs, "内側連成反復は不要です（有効状態量が1つ以下）");
                        break;
                    }

                    // 変化量を計算してログ出力
                    auto delta = computeCoupledDelta(constants, ventNetwork, thermalNetwork,
                                                     prevPressuresByKey, prevTempsByVertex);
                    if (humidityActive) {
                        delta.humidityChange = calculateHumidityChangeByVertex(thermalNetwork.getGraph(), prevHumidityByVertex);
                    }
                    lastDelta = delta;
                    if (logEnabled) {
                        writeLog(
                            logs,
                            "圧力変化量: " + std::to_string(delta.pressureChange) +
                                " Pa, 温度変化量: " + std::to_string(delta.temperatureChange) +
                                " K, 湿気変化量: " + std::to_string(delta.humidityChange) +
                                " kg/kg(DA), 潜熱反映: " + std::to_string(latentStats.maxAppliedHeatW) +
                                " W, 湿気反復: " + std::to_string(lastHumiditySolveStats.iterations) +
                                ", 湿気残差: " + std::to_string(lastHumiditySolveStats.finalMaxDiff));
                    }

                    // 収束判定（2回目以降）
                    if (coupledIter > 1) {
                        const double pTol = couplingPressureTol(constants);
                        const double tTol = couplingTemperatureTol(constants);
                        const double xTol = couplingHumidityTol(constants);
                        const bool pOk = !constants.pressureCalc || (delta.pressureChange < pTol);
                        const bool tOk = !constants.temperatureCalc || (delta.temperatureChange < tTol);
                        const bool xOk = !humidityActive || (delta.humidityChange < xTol);
                        if (pOk && tOk && xOk) {
                            if (logEnabled) {
                                writeLog(logs, "空気-熱-湿気 連成計算が収束しました (" +
                                                    std::to_string(coupledIter) + "回)");
                            }
                            break;
                        }
                    }

                    if (coupledIter >= static_cast<int>(constants.maxInnerIteration)) {
                        if (logEnabled) {
                            const double pTol = couplingPressureTol(constants);
                            const double tTol = couplingTemperatureTol(constants);
                            const double xTol = couplingHumidityTol(constants);
                            const double pRatio = constants.pressureCalc ? (lastDelta.pressureChange / std::max(1e-30, pTol)) : 0.0;
                            const double tRatio = constants.temperatureCalc ? (lastDelta.temperatureChange / std::max(1e-30, tTol)) : 0.0;
                            const double xRatio = humidityActive ? (lastDelta.humidityChange / std::max(1e-30, xTol)) : 0.0;

                            std::string dominant = "none";
                            double domRatio = -1.0;
                            if (constants.pressureCalc && pRatio > domRatio) { domRatio = pRatio; dominant = "pressure"; }
                            if (constants.temperatureCalc && tRatio > domRatio) { domRatio = tRatio; dominant = "temperature"; }
                            if (humidityActive && xRatio > domRatio) { domRatio = xRatio; dominant = "humidity"; }

                            std::ostringstream oss;
                            oss << "連成計算が最大反復回数に到達: iter=" << coupledIter
                                << ", dominant=" << dominant
                                << ", pressure=" << lastDelta.pressureChange << "/" << pTol
                                << ", temperature=" << lastDelta.temperatureChange << "/" << tTol
                                << ", humidity=" << lastDelta.humidityChange << "/" << xTol
                                << ", latentApplied=" << lastLatentAppliedW << " W"
                                << ", humidityIter=" << lastHumiditySolveStats.iterations
                                << ", humidityResidual=" << lastHumiditySolveStats.finalMaxDiff;
                            writeLog(logs, oss.str());
                        }
                        throw std::runtime_error("Maximum iteration count reached: stopping after maximum iteration count");
                    }
                }
            }
            // pressureCalc=false の場合でも、aircon制御（処理熱量/風量/COP計算）が流量を参照できるように
            // VentilationNetwork の確定 flow_rate（fixed_flow 等）から FlowRateMap を生成する。
            if (!constants.pressureCalc) {
                step.flowRates = ventNetwork.collectFlowRateMap();
            }
            if (constants.humidityCalc && !constants.moistureCouplingEnabled) {
                // 連成OFF時は従来互換: 外側ループごとに1回のみ湿気更新
                restoreXPrevToGraph(thermalNetwork.getGraph(), ventNetwork, xPrevByVertex);
                restoreWPrevToGraph(thermalNetwork.getGraph(), wPrevByVertex);
                (void)core::humidity::updateHumidityIfEnabled(constants, ventNetwork, thermalNetwork, step.flowRates, logs, timings,
                                                              meta + ",iteration=" + std::to_string(iteration + 1));
            }

            // エアコン制御ロジック（連成計算後）
            const auto airconRes = runAirconControlAndAdjust(
                airconController,
                thermalNetwork,
                ventNetwork,
                constants,
                step.flowRates,
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

    // 濃度（c）更新：エアコン制御が完了した後でOK（エアコン制御には影響しない想定）
    transport::updateConcentrationIfEnabled(constants, ventNetwork, thermalNetwork, logs, timings, meta);

    // 1タイムステップ分の結果を構築（呼び出し側で即座に書き出す想定）
    buildTimestepResult(constants, ventNetwork, thermalNetwork, airconController, step.flowRates, logs, timestepResultOut);

    if (logEnabled) {
        writeLog(logs,
                 "タイムステップ終了  総連成反復回数: " + std::to_string(totalIterations),
                 true);
    }
}
