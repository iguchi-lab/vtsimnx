#include "simulation_runner.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "aircon/aircon_controller.h"
#include "core/humidity/humidity_solver.h"
#include "simulation_runner_helpers.h"
#include "transport/concentration_solver.h"
#include "utils/utils.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <fstream>
#include <ostream>
#include <string>
#include <boost/range/iterator_range.hpp>

// 連成計算（pressure/thermal）1回分の「確定データ」をまとめる
struct CoupledStepData {
    PressureMap pressureMap;
    FlowRateMap flowRates;
    FlowBalanceMap flowBalance;
};

static CoupledStepData
performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                              ThermalNetwork& thermalNetwork,
                              const SimulationConstants& constants,
                              std::ostream& logs,
                              int& totalIterations,
                              TimingList& timings,
                              const std::string& meta);

namespace {
using namespace simulation::detail;

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

static void runCoupledInnerLoop(VentilationNetwork& ventNetwork,
                                ThermalNetwork& thermalNetwork,
                                AirconController& airconController,
                                const SimulationConstants& constants,
                                std::ostream& logs,
                                TimingList& timings,
                                const std::string& meta,
                                bool logEnabled,
                                int outerIteration,
                                const std::vector<double>& xPrevByVertex,
                                const std::vector<double>& wPrevByVertex,
                                CoupledStepData& step,
                                int& totalIterations) {
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
                              meta + ",iteration=" + std::to_string(outerIteration + 1));
            step = performCoupledStepCalculation(ventNetwork, thermalNetwork, constants, logs, totalIterations, timings,
                                                 meta + ",iteration=" + std::to_string(outerIteration + 1));
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
                meta + ",iteration=" + std::to_string(outerIteration + 1) +
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
                runCoupledInnerLoop(ventNetwork,
                                    thermalNetwork,
                                    airconController,
                                    constants,
                                    logs,
                                    timings,
                                    meta,
                                    logEnabled,
                                    iteration,
                                    xPrevByVertex,
                                    wPrevByVertex,
                                    step,
                                    totalIterations);
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
