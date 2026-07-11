#include "simulation_coupling_control.h"

#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace simulation {
namespace detail {

InnerCouplingEval evaluateInnerCoupling(const SimulationConstants& constants,
                                        bool humidityActive,
                                        int coupledIter,
                                        const CoupledDelta& delta,
                                        bool pressureConvergedAfterFirstSolve) {
    InnerCouplingEval eval;
    eval.delta = delta;
    eval.pressureTol = couplingPressureTol(constants);
    eval.temperatureTol = couplingTemperatureTol(constants);
    eval.humidityTol = couplingHumidityTol(constants);

    // 1回目で pressure が未収束なら停止（従来と同じ）
    if (constants.pressureCalc && coupledIter == 1 && !pressureConvergedAfterFirstSolve) {
        eval.action = InnerCouplingAction::ThrowPressureNonConvergence;
        return eval;
    }

    // 連成計算が不要な場合、1回の計算後に抜ける
    if (!needsInnerCoupledIteration(constants)) {
        eval.action = InnerCouplingAction::BreakNoNeed;
        return eval;
    }

    // 収束判定（2回目以降）
    if (coupledIter > 1) {
        const bool pOk = !constants.pressureCalc || (delta.pressureChange < eval.pressureTol);
        const bool tOk = !constants.temperatureCalc || (delta.temperatureChange < eval.temperatureTol);
        const bool xOk = !humidityActive || (delta.humidityChange < eval.humidityTol);
        if (pOk && tOk && xOk) {
            eval.action = InnerCouplingAction::BreakConverged;
            return eval;
        }
    }

    if (coupledIter >= static_cast<int>(constants.maxInnerIteration)) {
        const double pRatio =
            constants.pressureCalc ? (delta.pressureChange / std::max(1e-30, eval.pressureTol)) : 0.0;
        const double tRatio =
            constants.temperatureCalc ? (delta.temperatureChange / std::max(1e-30, eval.temperatureTol)) : 0.0;
        const double xRatio = humidityActive ? (delta.humidityChange / std::max(1e-30, eval.humidityTol)) : 0.0;

        std::string dominant = "none";
        double domRatio = -1.0;
        if (constants.pressureCalc && pRatio > domRatio) {
            domRatio = pRatio;
            dominant = "pressure";
        }
        if (constants.temperatureCalc && tRatio > domRatio) {
            domRatio = tRatio;
            dominant = "temperature";
        }
        if (humidityActive && xRatio > domRatio) {
            dominant = "humidity";
        }
        eval.dominant = dominant;
        eval.action = InnerCouplingAction::ThrowMaxIteration;
        return eval;
    }

    eval.action = InnerCouplingAction::Continue;
    return eval;
}

void logHumiditySolverNotConverged(std::ostream& logs,
                                   bool logEnabled,
                                   const core::humidity::HumiditySolveStats& stats) {
    if (!logEnabled || stats.converged) return;
    writeLog(
        logs,
        "湿気ソルバ未収束(内側反復継続): iter=" + std::to_string(stats.iterations) +
            ", maxDiff=" + std::to_string(stats.finalMaxDiff) +
            ", active=" + std::to_string(stats.activeVertices));
}

void logPressureFallbackStop(std::ostream& logs, bool logEnabled) {
    if (!logEnabled) return;
    writeLog(logs, "エラー: フォールバック後も未収束のため停止します（最終通常解の再試行は無効化）");
}

void logInnerCouplingNotNeeded(std::ostream& logs, bool logEnabled) {
    if (!logEnabled) return;
    writeLog(logs, "内側連成反復は不要です（有効状態量が1つ以下）");
}

void logInnerCouplingDelta(std::ostream& logs,
                           bool logEnabled,
                           const CoupledDelta& delta,
                           double latentAppliedW,
                           const core::humidity::HumiditySolveStats& humidityStats) {
    if (!logEnabled) return;
    writeLog(
        logs,
        "圧力変化量: " + std::to_string(delta.pressureChange) +
            " Pa, 温度変化量: " + std::to_string(delta.temperatureChange) +
            " K, 湿気変化量: " + std::to_string(delta.humidityChange) +
            " kg/kg(DA), 潜熱反映: " + std::to_string(latentAppliedW) +
            " W, 湿気反復: " + std::to_string(humidityStats.iterations) +
            ", 湿気残差: " + std::to_string(humidityStats.finalMaxDiff));
}

void logInnerCouplingConverged(std::ostream& logs, bool logEnabled, int coupledIter) {
    if (!logEnabled) return;
    writeLog(logs, "空気-熱-湿気 連成計算が収束しました (" + std::to_string(coupledIter) + "回)");
}

void logInnerCouplingMaxIteration(std::ostream& logs,
                                  bool logEnabled,
                                  int coupledIter,
                                  const InnerCouplingEval& eval,
                                  double latentAppliedW,
                                  const core::humidity::HumiditySolveStats& humidityStats) {
    if (!logEnabled) return;
    std::ostringstream oss;
    oss << "連成計算が最大反復回数に到達: iter=" << coupledIter
        << ", dominant=" << eval.dominant
        << ", pressure=" << eval.delta.pressureChange << "/" << eval.pressureTol
        << ", temperature=" << eval.delta.temperatureChange << "/" << eval.temperatureTol
        << ", humidity=" << eval.delta.humidityChange << "/" << eval.humidityTol
        << ", latentApplied=" << latentAppliedW << " W"
        << ", humidityIter=" << humidityStats.iterations
        << ", humidityResidual=" << humidityStats.finalMaxDiff;
    writeLog(logs, oss.str());
}

void logAirconRecompute(std::ostream& logs, bool logEnabled) {
    if (!logEnabled) return;
    writeLog(logs, "エアコン制御の修正が行われました。再計算を実行します。");
}

void logThermalNotConverged(std::ostream& logs,
                            bool logEnabled,
                            const std::string& method,
                            double rmseBalance,
                            double maxBalance,
                            int loopIndex1Based) {
    if (!logEnabled) return;
    std::ostringstream oss;
    oss << "　エラー: 熱計算が未収束のため停止します (method="
        << method
        << ", RMSE=" << std::scientific << std::setprecision(6) << rmseBalance
        << ", maxBalance=" << std::scientific << std::setprecision(6) << maxBalance
        << ", loop=" << loopIndex1Based
        << ")";
    writeLog(logs, oss.str());
}

void logOuterLoopConverged(std::ostream& logs, bool logEnabled, int loopIndex1Based) {
    if (!logEnabled) return;
    writeLog(logs,
             "圧力-温度連成計算-エアコン制御ループ " +
                 std::to_string(loopIndex1Based) + " が収束しました。");
}

void logTimestepFinished(std::ostream& logs, bool logEnabled, int totalIterations) {
    if (!logEnabled) return;
    writeLog(logs,
             "タイムステップ終了  総連成反復回数: " + std::to_string(totalIterations),
             true);
}

} // namespace detail
} // namespace simulation
