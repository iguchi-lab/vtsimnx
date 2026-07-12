#include "simulation_coupling_control.h"

#include "simulation_error.h"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>

namespace simulation {
namespace detail {

InnerCouplingEval evaluateInnerCoupling(const SimulationConstants& constants,
                                        bool humidityActive,
                                        bool latentActive,
                                        std::size_t coupledIter,
                                        std::size_t minCouplingIterations,
                                        const CoupledDelta& delta,
                                        bool pressureConvergedAfterFirstSolve) {
    InnerCouplingEval eval;
    eval.delta = delta;
    eval.pressureTol = couplingPressureTol(constants);
    eval.temperatureTol = couplingTemperatureTol(constants);
    eval.humidityTol = couplingHumidityTol(constants);
    const double absLat = std::max(0.0, constants.couplingLatentAbsoluteToleranceW);
    const double relLat = std::max(0.0, constants.couplingLatentRelativeTolerance);
    eval.latentTol = absLat; // ログ用の絶対項

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

    const std::size_t minIters = std::max<std::size_t>(1, minCouplingIterations);
    const auto latentOk = [&]() -> bool {
        if (!latentActive) return true;
        const double tol = absLat + relLat * std::max(0.0, delta.latentHeatScale);
        return delta.latentHeatChange <= tol;
    };

    // 収束判定（minIters 回目以降）
    if (coupledIter >= minIters) {
        const bool pOk = !constants.pressureCalc || (delta.pressureChange < eval.pressureTol);
        const bool tOk = !constants.temperatureCalc || (delta.temperatureChange < eval.temperatureTol);
        const bool xOk = !humidityActive || (delta.humidityChange < eval.humidityTol);
        const bool qOk = latentOk();
        if (pOk && tOk && xOk && qOk) {
            eval.action = InnerCouplingAction::BreakConverged;
            return eval;
        }
    }

    const std::size_t maxCoupling = effectiveMaxCouplingIterations(constants);
    if (coupledIter >= maxCoupling) {
        const double pRatio =
            constants.pressureCalc ? (delta.pressureChange / std::max(1e-30, eval.pressureTol)) : 0.0;
        const double tRatio =
            constants.temperatureCalc ? (delta.temperatureChange / std::max(1e-30, eval.temperatureTol)) : 0.0;
        const double xRatio = humidityActive ? (delta.humidityChange / std::max(1e-30, eval.humidityTol)) : 0.0;
        const double qRatio = latentActive ? (delta.latentHeatChange / std::max(1e-30, std::max(absLat, 1e-30))) : 0.0;

        std::string dominant = "none";
        double domRatio = -1.0;
        auto consider = [&](const char* name, double ratio, bool active) {
            if (!active) return;
            if (ratio > domRatio) {
                domRatio = ratio;
                dominant = name;
            }
        };
        consider("pressure", pRatio, constants.pressureCalc);
        consider("temperature", tRatio, constants.temperatureCalc);
        consider("humidity", xRatio, humidityActive);
        consider("latent", qRatio, latentActive);
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
    writeDomainLog(
        logs,
        "湿気",
        "ソルバ未収束(停止): iter=" + std::to_string(stats.iterations) +
            ", relativeResidual=" + std::to_string(stats.finalRelativeResidual) +
            ", active=" + std::to_string(stats.activeVertices));
}

void logPressureFallbackStop(std::ostream& logs, bool logEnabled) {
    if (!logEnabled) return;
    writeDomainLog(logs, "連成", "[ERROR] フォールバック後も未収束のため停止します（最終通常解の再試行は無効化）");
}

void logInnerCouplingNotNeeded(std::ostream& logs, bool logEnabled) {
    if (!logEnabled) return;
    writeDomainLog(logs, "連成", "内側連成反復は不要です（有効状態量が1つ以下）");
}

void logInnerCouplingDelta(std::ostream& logs,
                           bool logEnabled,
                           const CoupledDelta& delta,
                           double latentAppliedW,
                           const core::humidity::HumiditySolveStats& humidityStats) {
    if (!logEnabled) return;
    writeDomainLog(
        logs,
        "連成",
        "圧力変化量: " + std::to_string(delta.pressureChange) +
            " Pa, 温度変化量: " + std::to_string(delta.temperatureChange) +
            " K, 湿気変化量: " + std::to_string(delta.humidityChange) +
            " kg/kg(DA), 潜熱変化量: " + std::to_string(delta.latentHeatChange) +
            " W, 潜熱反映: " + std::to_string(latentAppliedW) +
            " W, 湿気反復: " + std::to_string(humidityStats.iterations) +
            ", 湿気相対残差: " + std::to_string(humidityStats.finalRelativeResidual));
}

void logInnerCouplingConverged(std::ostream& logs, bool logEnabled, std::size_t coupledIter) {
    if (!logEnabled) return;
    writeDomainLog(logs, "連成", "空気-熱-湿気 連成計算が収束しました (" + std::to_string(coupledIter) + "回)");
}

void logInnerCouplingMaxIteration(std::ostream& logs,
                                  bool logEnabled,
                                  std::size_t coupledIter,
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
        << ", humidityRelativeResidual=" << humidityStats.finalRelativeResidual;
    writeDomainLog(logs, "連成", oss.str());
}

void logAirconRecompute(std::ostream& logs, bool logEnabled) {
    if (!logEnabled) return;
    writeDomainLog(logs, "空調", "制御の修正が行われました。再計算を実行します。");
}

void logThermalNotConverged(std::ostream& logs,
                            bool logEnabled,
                            const std::string& method,
                            double rmseBalance,
                            double maxBalance,
                            int loopIndex1Based) {
    if (!logEnabled) return;
    std::ostringstream oss;
    oss << "[ERROR] 熱計算が未収束のため停止します (method="
        << method
        << ", RMSE=" << std::scientific << std::setprecision(6) << rmseBalance
        << ", maxBalance=" << std::scientific << std::setprecision(6) << maxBalance
        << ", loop=" << loopIndex1Based
        << ")";
    writeDomainLog(logs, "熱", oss.str());
}

void logOuterLoopConverged(std::ostream& logs, bool logEnabled, int loopIndex1Based) {
    if (!logEnabled) return;
    writeDomainLog(logs, "連成",
             "圧力-温度連成計算-エアコン制御ループ " +
                 std::to_string(loopIndex1Based) + " が収束しました。");
}

void logTimestepFinished(std::ostream& logs, bool logEnabled, int totalIterations) {
    if (!logEnabled) return;
    writeDomainLog(logs, "連成",
             "タイムステップ終了  総連成反復回数: " + std::to_string(totalIterations),
             true);
}

void ensureOuterAirconLoopConverged(bool outerLoopConverged) {
    if (outerLoopConverged) return;
    throw simulation::Error(
        simulation::ErrorCode::AirconMaxIterations,
        "Aircon control did not converge within the iteration limit");
}

} // namespace detail
} // namespace simulation
