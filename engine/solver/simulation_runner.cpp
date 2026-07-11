#include "simulation_runner.h"
#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "network/contaminant_network.h"
#include "aircon/aircon_controller.h"
#include "core/humidity/humidity_solver.h"
#include "simulation_coupled_step.h"
#include "simulation_coupling_control.h"
#include "simulation_runner_helpers.h"
#include "transport/concentration_solver.h"
#include "utils/utils.h"

#include <limits>
#include <cmath>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <boost/range/iterator_range.hpp>

namespace {
using namespace simulation::detail;

struct SharedNodeStateArgs {
    Graph& nodeGraph;
    ConstNodeStateView nodeState;
};

static SharedNodeStateArgs makeSharedNodeStateArgs(ThermalNetwork& thermalNetwork) {
    return SharedNodeStateArgs{
        thermalNetwork.getGraph(),
        static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView(),
    };
}

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

    // 0. DUCT_CENTRAL の処理熱量連動風量を補正（変更が入ったら外側ループをやり直し）
    {
        ScopedTimer timer(timings, "aircon_duct_flow_adjust", meta);
        const bool ductFlowAdjusted = airconController.checkAndAdjustDuctCentralAirflow(
            thermalNetwork, ventNetwork, flowRates, logs);
        if (ductFlowAdjusted) {
            r.allControlled = false;
            r.shouldRecompute = true;
            return r;
        }
    }

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
                                HumidityNetwork& humidityNetwork,
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
    (void)airconController;
    // 連成反復（air -> thermal -> moisture -> latent_feedback の収束まで回す）
    std::vector<double> prevTempsByVertex;
    std::vector<double> prevPressuresByKey;
    std::vector<double> prevHumidityByVertex;
    std::vector<double> baseHeatSourceByVertex;
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
            const auto sharedNodeState = makeSharedNodeStateArgs(thermalNetwork);
            // 同一タイムステップ内反復なので、毎回 x_prev / w_prev に戻して再評価する。
            restoreXPrevToGraph(sharedNodeState.nodeGraph, ventNetwork, xPrevByVertex);
            restoreWPrevToGraph(sharedNodeState.nodeGraph, wPrevByVertex);
            lastHumiditySolveStats = core::humidity::updateHumidityIfEnabled(
                constants,
                ventNetwork,
                sharedNodeState.nodeGraph,
                sharedNodeState.nodeState,
                humidityNetwork,
                step.flowRates, logs, timings,
                meta + ",iteration=" + std::to_string(outerIteration + 1) +
                           ",coupledIter=" + std::to_string(coupledIter));
            logHumiditySolverNotConverged(logs, logEnabled, lastHumiditySolveStats);
            relaxHumidityByVertex(thermalNetwork.getGraph(), ventNetwork, prevHumidityByVertex, constants.humidityRelaxation);
        }

        // 方針B: 除湿潜熱は熱ネットワークへフィードバックしない。
        // 熱源は毎反復で基準値へ戻し、湿気→熱の注入は行わない。
        restoreHeatSourceByVertex(thermalNetwork.getGraph(), baseHeatSourceByVertex);
        const double latentAppliedThisIter = 0.0;
        lastLatentAppliedW = latentAppliedThisIter;

        // 変化量を計算
        auto delta = computeCoupledDelta(constants, ventNetwork, thermalNetwork,
                                         prevPressuresByKey, prevTempsByVertex);
        if (humidityActive) {
            delta.humidityChange = calculateHumidityChangeByVertex(thermalNetwork.getGraph(), prevHumidityByVertex);
        }

        const InnerCouplingEval eval = evaluateInnerCoupling(
            constants,
            humidityActive,
            coupledIter,
            delta,
            ventNetwork.getLastPressureConverged());

        // ログ: 打ち切り系は評価後、変化量ログは Continue / BreakConverged / ThrowMax の前に出す
        // 従来順序:
        // 1) pressure non-conv → log + throw (変化量ログなし)
        // 2) no need → log + break (変化量ログなし)
        // 3) delta log
        // 4) converged → log + break
        // 5) max iter → log + throw
        if (eval.action == InnerCouplingAction::ThrowPressureNonConvergence) {
            logPressureFallbackStop(logs, logEnabled);
            throw std::runtime_error("Disabled final normal solve: stopping after fallback non-convergence");
        }
        if (eval.action == InnerCouplingAction::BreakNoNeed) {
            logInnerCouplingNotNeeded(logs, logEnabled);
            break;
        }

        logInnerCouplingDelta(logs, logEnabled, delta, latentAppliedThisIter, lastHumiditySolveStats);

        if (eval.action == InnerCouplingAction::BreakConverged) {
            logInnerCouplingConverged(logs, logEnabled, coupledIter);
            break;
        }
        if (eval.action == InnerCouplingAction::ThrowMaxIteration) {
            logInnerCouplingMaxIteration(logs, logEnabled, coupledIter, eval,
                                         lastLatentAppliedW, lastHumiditySolveStats);
            throw std::runtime_error("Maximum iteration count reached: stopping after maximum iteration count");
        }
        // Continue: next iteration
    }
}

static void buildTimestepResult(const SimulationConstants& constants,
                                VentilationNetwork& ventNetwork,
                                ThermalNetwork& thermalNetwork,
                                HumidityNetwork& humidityNetwork,
                                ContaminantNetwork& contaminantNetwork,
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
        convertDoublesToF32(
            timestepResult.humidityX,
            humidityNetwork.collectOutputValues(static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView()));
        convertDoublesToF32(
            timestepResult.humidityFlux,
            ventNetwork.collectHumidityFluxValues());
    }
    if (constants.concentrationCalc) {
        convertDoublesToF32(
            timestepResult.concentrationC,
            contaminantNetwork.collectOutputValues(static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView()));
        convertDoublesToF32(
            timestepResult.concentrationFlux,
            ventNetwork.collectConcentrationFluxValues());
    }

    timestepResultOut = std::move(timestepResult);
}

} // namespace

void runSimulation(VentilationNetwork& ventNetwork,
                   ThermalNetwork& thermalNetwork,
                   HumidityNetwork& humidityNetwork,
                   ContaminantNetwork& contaminantNetwork,
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
                                    humidityNetwork,
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
                const auto sharedNodeState = makeSharedNodeStateArgs(thermalNetwork);
                // 連成OFF時は従来互換: 外側ループごとに1回のみ湿気更新
                restoreXPrevToGraph(sharedNodeState.nodeGraph, ventNetwork, xPrevByVertex);
                restoreWPrevToGraph(sharedNodeState.nodeGraph, wPrevByVertex);
                (void)core::humidity::updateHumidityIfEnabled(constants, ventNetwork, sharedNodeState.nodeGraph,
                                                              sharedNodeState.nodeState,
                                                              humidityNetwork,
                                                              step.flowRates, logs, timings,
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
                logAirconRecompute(logs, logEnabled);
                continue;
            }
            // 収束判定:
            // - aircon が安定していても、熱計算が未収束なら「収束しました」とは扱わない
            const bool thermalOk = thermalNetwork.getLastThermalConverged();
            if (!thermalOk) {
                // 無限ループや「誤って収束扱い」を避けるため、未収束になった時点でエラー終了する
                logThermalNotConverged(logs, logEnabled,
                                       thermalNetwork.getLastThermalMethod(),
                                       thermalNetwork.getLastThermalRmseBalance(),
                                       thermalNetwork.getLastThermalMaxBalance(),
                                       iteration + 1);
                throw std::runtime_error("Thermal solver did not converge: stopping to avoid infinite loop");
            }
            loopConverged = airconRes.allControlled;
        }
        if (loopConverged) {
            logOuterLoopConverged(logs, logEnabled, iteration + 1);
            break;   // 全てのエアコンが制御完了の場合、反復を終了
        }
    }

    // 濃度（c）更新：エアコン制御が完了した後でOK（エアコン制御には影響しない想定）
    const auto sharedNodeState = makeSharedNodeStateArgs(thermalNetwork);
    transport::updateConcentrationIfEnabled(constants,
                                            ventNetwork,
                                            sharedNodeState.nodeGraph,
                                            sharedNodeState.nodeState,
                                            contaminantNetwork,
                                            logs,
                                            timings,
                                            meta);

    // 1タイムステップ分の結果を構築（呼び出し側で即座に書き出す想定）
    buildTimestepResult(constants, ventNetwork, thermalNetwork, humidityNetwork, contaminantNetwork,
                       airconController, step.flowRates, logs, timestepResultOut);

    logTimestepFinished(logs, logEnabled, totalIterations);
}
