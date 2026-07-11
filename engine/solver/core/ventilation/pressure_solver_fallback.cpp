#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/edge_mutation_guard.h"
#include "core/ventilation/pressure_balance.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <iomanip>
#include <limits>
#include <sstream>

// =============================================================================
// Fallbackループ（orchestrator）
// =============================================================================

std::optional<PressureSolver::SolverResult> PressureSolver::runFallbackLoop(
        const SimulationConstants& constants,
        SolverSetup& setup,
        ceres::Solver::Summary& summary) {
    auto& nodeNames = setup.nodeNames;
    auto& pressures = setup.pressures;

    writeLog(logFile_, "圧力計算フォールバック");
    FallbackLogger fallbackLog = [&](int indent, const std::string& message) {
        std::string prefix;
        for (int i = 0; i < indent; ++i) {
            prefix += "  ";
        }
        writeLog(logFile_, prefix + message);
    };
    auto formatScientific = [](double value) {
        std::ostringstream os;
        os << std::scientific << std::setprecision(6) << value;
        return os.str();
    };

    network_.setLastPressureConverged(false);
    fallbackLog(0, "エラー: 圧力計算が収束しませんでした");

    std::string terminationType;
    switch(summary.termination_type) {
        case ceres::NO_CONVERGENCE:
            terminationType = "NO_CONVERGENCE (最大反復回数到達)";
            break;
        case ceres::FAILURE:
            terminationType = "FAILURE (計算失敗)";
            break;
        case ceres::USER_FAILURE:
            terminationType = "USER_FAILURE (ユーザー関数エラー)";
            break;
        default:
            terminationType = "UNKNOWN (" + std::to_string(static_cast<int>(summary.termination_type)) + ")";
    }
    fallbackLog(1, "終了理由: " + terminationType);

    PressureMap currentPressures = extractPressures(pressures, nodeNames);
    FallbackOuterState state;
    state.prevPressureMapFB = currentPressures;

    if (constants.logVerbosity >= 1) {
        fallbackLog(0, "[Fallback] スーパーノード化 + 外気ギャップ固定流量化を適用します");
    }

    Graph& g = network_.getGraph();
    ventilation::EdgeMutationGuard edgeGuard(g);
    SupernodePartition partition = detectSupernodePartition(constants, currentPressures);

    const auto tols = ventilation::makePressureSolverTolerances(constants);
    const int maxOuter = 5;
    const int minOuter = 2;

    for (int outer = 1; outer <= maxOuter; ++outer) {
        edgeGuard.restore();

        const std::string outerTag = "[外部反復 " + std::to_string(outer) + "/" + std::to_string(maxOuter) + "]";
        std::string prevCostText = "prev=-";
        if (!std::isinf(state.lastCostOuter)) {
            prevCostText = "prev=" + formatScientific(state.lastCostOuter);
        }
        fallbackLog(0, outerTag + " 開始 | " + prevCostText);

        fallbackLog(1, "[A] スーパーノード代表圧フェーズ開始" +
                           std::string(outer >= 2 ? " | source=B(prev)" : " | source=A(current)"));
        StageASolveResult stageA = solveStageAReduced(
            constants, g, partition, state.prevPressureMapFB, fallbackLog);
        if (!ventilation::canProceedToFallbackStageB(stageA.ok, /*interfaceFreezeSkipped=*/false)) {
            fallbackLog(0, "[Fallback] Stage A 未収束のため外部反復を打ち切り");
            break;
        }

        const auto freeze = freezeInterfaceFlows(
            g,
            edgeGuard,
            partition,
            stageA.mapping,
            stageA.pressureMap,
            state.prevPressureMapFB,
            outer,
            fallbackLog);
        if (!ventilation::canProceedToFallbackStageB(stageA.ok, freeze.skipped)) {
            fallbackLog(0, "[Fallback] interface flow evaluation failed");
            break;
        }

        fallbackLog(1, "[B] 固定流量下でフルノード再解フェーズ開始");
        StageBSolveResult stageB = solveStageBFull(
            constants, g, partition, stageA.pressureMap, fallbackLog);

        FallbackOuterAction action = evaluateFallbackOuter(
            constants,
            g,
            edgeGuard,
            partition,
            stageA,
            stageB,
            freeze,
            outer,
            maxOuter,
            minOuter,
            outerTag,
            tols,
            state,
            fallbackLog);

        if (action == FallbackOuterAction::AcceptSolution ||
            action == FallbackOuterAction::StopOuter) {
            break;
        }
    }

    edgeGuard.restore();

    PressureMap seed = state.finalHaveSolution
                           ? state.finalPressureMapFB
                           : state.prevPressureMapFB;
    if (seed.empty()) {
        network_.setLastPressureConverged(false);
        fallbackLog(0, "[Fallback] warm-start 用の圧力シードがありません");
        return std::nullopt;
    }

    fallbackLog(0, state.finalHaveSolution
                       ? "[Fallback] 復元後候補を初期値に通常ソルバーを再実行します"
                       : "[Fallback] 最終圧力を初期値に通常ソルバーを再実行します");
    auto warm = tryPrimaryWarmStart(constants, setup, seed, tols.massBalanceMaxAbs);
    if (warm) {
        fallbackLog(0, "[Fallback] warm-start primary 合格 | 採用");
        return warm;
    }

    network_.setLastPressureConverged(false);
    fallbackLog(0, "[Fallback] warm-start primary 不合格 | 非収束として報告");
    return std::nullopt;
}
