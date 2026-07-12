#include "core/ventilation/pressure_solver_impl.h"
#include "core/ventilation/edge_mutation_guard.h"
#include "core/ventilation/pressure_balance.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "utils/utils.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

PressureSolver::Impl::FallbackOuterAction PressureSolver::Impl::evaluateFallbackOuter(
        const SimulationConstants& constants,
        Graph& g,
        ventilation::EdgeMutationGuard& edgeGuard,
        const SupernodePartition& partition,
        const StageASolveResult& stageA,
        StageBSolveResult& stageB,
        const InterfaceFreezeResult& freeze,
        int outer,
        int maxOuter,
        int minOuter,
        const std::string& outerTag,
        const ventilation::PressureSolverTolerances& tols,
        FallbackOuterState& state,
        const FallbackLogger& fallbackLog) {
    const double massBalanceMaxAbs = tols.massBalanceMaxAbs;
    const double interfaceFlowMaxAbs = tols.interfaceFlowMaxAbs;
    auto& vToParamIdxB = stageB.setup.vertexToParamIndex;
    std::vector<double>& pressuresFBB = stageB.setup.pressures;
    const auto& vertices = partition.vertices;
    const auto& groupOfVertex = partition.groupOfVertex;
    const PressureMap& pressureMapFB_B = stageB.pressureMap;

    if (constants.logFallbackDetails && constants.logVerbosity >= 2) {
        writeLog(logFile_, "\t\tSupernode(B) 内部圧力:");
        int superCountB = *std::max_element(groupOfVertex.begin(), groupOfVertex.end()) + 1;
        if (superCountB <= 0) {
            writeLog(logFile_, "\t\t\t(スーパーノードなし)");
        } else {
            std::vector<bool> printedGroup(superCountB, false);
            for (size_t i = 0; i < groupOfVertex.size(); ++i) {
                int gidv = groupOfVertex[i];
                if (gidv < 0 || printedGroup[gidv]) continue;
                const auto& node = g[vertices[i]];
                double p = node.calc_p ? pressuresFBB[vToParamIdxB[vertices[i]]] : node.current_p;
                std::ostringstream ospp;
                ospp.setf(std::ios::fixed);
                ospp << std::setprecision(6) << p;
                writeLog(logFile_, "\t\t\tG" + std::to_string(gidv) + ": P=" + ospp.str() + " Pa");
                printedGroup[gidv] = true;
            }
        }
    }

    if (constants.logVerbosity >= 2) {
        std::ostringstream osCurr;
        osCurr.setf(std::ios::scientific);
        osCurr << std::setprecision(6) << stageB.summary.final_cost;
        if (state.lastCostOuter == std::numeric_limits<double>::infinity()) {
            fallbackLog(0, outerTag + " 結果: cost=" + osCurr.str() + ", 改善率=N/A");
        } else {
            double improve_pct = (state.lastCostOuter - stageB.summary.final_cost) /
                                 std::max(1e-300, state.lastCostOuter) * 100.0;
            std::ostringstream osPct;
            osPct.setf(std::ios::fixed);
            osPct << std::setprecision(3) << improve_pct;
            std::ostringstream osPrev;
            osPrev.setf(std::ios::scientific);
            osPrev << std::setprecision(6) << state.lastCostOuter;
            fallbackLog(0, outerTag + " 結果: prev=" + osPrev.str() + ", curr=" + osCurr.str() +
                               ", 改善率=" + osPct.str() + "%");
        }
    }

    // temporaryNetworkCost は診断ログ専用。停止判定には使わない。
    double temporaryNetworkCost = std::numeric_limits<double>::infinity();
    double restoredNetworkCost = std::numeric_limits<double>::infinity();

    {
        std::ostringstream osfb2;
        osfb2 << std::scientific << std::setprecision(6) << stageB.summary.final_cost;
        if (stageB.ok) {
            fallbackLog(0, "[圧力] [Fallback] Stage B 仮ネットワーク物理合格 | cost=" + osfb2.str() +
                               " | 外部反復 " + std::to_string(outer) + "/" +
                               std::to_string(maxOuter));
        } else {
            fallbackLog(0, "[圧力] [Fallback] Stage B Ceres/仮物理未達 | cost=" + osfb2.str() +
                               " | 復元後評価へ続行");
        }
    }

    // 仮ネットワーク上の診断コスト（復元前）。改善率比較には使わない。
    if (!edgeGuard.empty()) {
        auto evalTmp = evaluatePressureSolution(pressureMapFB_B, massBalanceMaxAbs);
        if (!evalTmp.flowOk) {
            fallbackLog(1, "[Network] 仮ネットワーク風量評価失敗: " + evalTmp.detail);
        } else {
            temporaryNetworkCost = 0.5 * (evalTmp.solvedNodeMetrics.l2 * evalTmp.solvedNodeMetrics.l2);
            std::ostringstream osl1, osl2, osct, osmax;
            osl1.setf(std::ios::fixed);
            osl1 << std::setprecision(6) << evalTmp.solvedNodeMetrics.l1;
            osl2.setf(std::ios::fixed);
            osl2 << std::setprecision(6) << evalTmp.solvedNodeMetrics.l2;
            osct.setf(std::ios::fixed);
            osct << std::setprecision(6) << temporaryNetworkCost;
            osmax.setf(std::ios::scientific);
            osmax << std::setprecision(6) << evalTmp.solvedNodeMetrics.maxAbs;
            fallbackLog(1, "[Network] temporary L1=" + osl1.str() +
                               " | L2=" + osl2.str() +
                               " | cost=" + osct.str() +
                               " | mass_maxAbs=" + osmax.str() +
                               " (診断専用)");
        }
    }

    PressureMap pressureMapFB;
    auto vr = boost::vertices(g);
    for (auto v : boost::make_iterator_range(vr)) {
        const auto& node = g[v];
        if (node.calc_p) {
            size_t idx = vToParamIdxB[v];
            pressureMapFB[node.key] = pressuresFBB[idx];
        } else {
            pressureMapFB[node.key] = node.current_p;
        }
    }

    if (stageA.hasAnchorTarget) {
        double meanG0 = 0.0;
        int cntG0 = 0;
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (groupOfVertex[i] != 0) continue;
            const auto& node = g[vertices[i]];
            if (!node.calc_p) continue;
            double p = pressureMapFB[node.key];
            meanG0 += p;
            cntG0++;
        }
        if (cntG0 > 0) {
            meanG0 /= static_cast<double>(cntG0);
            double offset = stageA.anchorTargetPressure - meanG0;
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (groupOfVertex[i] != 0) continue;
                const auto& ndG0 = g[vertices[i]];
                if (!ndG0.calc_p) continue;
                auto itp = pressureMapFB.find(ndG0.key);
                if (itp != pressureMapFB.end()) itp->second += offset;
            }
            std::ostringstream osa;
            osa.setf(std::ios::fixed);
            osa << std::setprecision(6) << stageA.anchorTargetPressure;
            fallbackLog(1, "[Gauge] G0平均を " + osa.str() + " Pa に合わせました");
        }
    }

    // Ceres ok に関係なく、必ず元エッジ特性へ復元してから質量収支と interface を評価する。
    edgeGuard.restore();
    auto evalFinal = evaluatePressureSolution(pressureMapFB, massBalanceMaxAbs);
    if (!evalFinal.flowOk) {
        fallbackLog(1, "[Fallback] 復元後質量収支評価スキップ: " + evalFinal.detail);
    } else {
        restoredNetworkCost = 0.5 * (evalFinal.solvedNodeMetrics.l2 * evalFinal.solvedNodeMetrics.l2);
        const auto iface = evaluateInterfaceFlowConsistency(
            pressureMapFB, freeze.frozenFlows);
        const bool massOk = evalFinal.accepted;
        const bool ifaceOk = ventilation::acceptInterfaceFlowConsistency(
            iface, interfaceFlowMaxAbs);
        {
            std::ostringstream osmax, osiface, oscost;
            osmax << std::scientific << std::setprecision(6)
                  << evalFinal.solvedNodeMetrics.maxAbs;
            osiface << std::scientific << std::setprecision(6) << iface.maxAbs;
            oscost << std::scientific << std::setprecision(6) << restoredNetworkCost;
            fallbackLog(1, "[圧力] [Fallback] restored mass_maxAbs=" + osmax.str() +
                               " | mass_tol=" + std::to_string(massBalanceMaxAbs) +
                               " | iface_maxAbs=" + osiface.str() +
                               " | iface_tol=" + std::to_string(interfaceFlowMaxAbs) +
                               " | iface_edges=" + std::to_string(iface.edgeCount) +
                               " | restored_cost=" + oscost.str());
        }
        if (massOk && ifaceOk) {
            state.finalPressureMapFB = pressureMapFB;
            state.finalFlowRatesFB = std::move(evalFinal.flows);
            state.finalBalanceFB = std::move(evalFinal.allNodeBalances);
            state.finalHaveSolution = true;
            fallbackLog(0, "[圧力] [Fallback] 復元後候補物理収支合格 | warm-start へ進む | 外部反復 " +
                               std::to_string(outer) + "/" + std::to_string(maxOuter));
        } else {
            fallbackLog(0, std::string("[圧力] [Fallback] 復元後物理収支未達（継続）") +
                               (!massOk ? " mass" : "") +
                               (!ifaceOk ? " iface" : ""));
        }
    }

    state.prevPressureMapFB.clear();
    auto vrB2 = boost::vertices(g);
    for (auto v : boost::make_iterator_range(vrB2)) {
        const auto& node = g[v];
        if (node.calc_p) {
            size_t idx = vToParamIdxB[v];
            state.prevPressureMapFB[node.key] = pressuresFBB[idx];
        } else {
            state.prevPressureMapFB[node.key] = node.current_p;
        }
    }

    // 停止判定は「前回の復元後コスト」と「今回の復元後コスト」のみ。
    const double currNetworkCostOuter = restoredNetworkCost;

    if (state.finalHaveSolution) {
        return FallbackOuterAction::AcceptSolution;
    }

    const auto progress = ventilation::decideFallbackOuterProgress(
        outer, minOuter,
        stageB.summary.final_cost, state.lastCostOuter,
        currNetworkCostOuter, state.lastNetworkCostOuter);
    if (progress == ventilation::FallbackOuterProgress::StopNoNetImprove) {
        double improve_pct_ceres = (state.lastCostOuter - stageB.summary.final_cost) /
                                   std::max(1e-300, state.lastCostOuter) * 100.0;
        double improve_pct_net   = (state.lastNetworkCostOuter - currNetworkCostOuter) /
                                   std::max(1e-300, state.lastNetworkCostOuter) * 100.0;
        std::ostringstream osC, osN;
        osC.setf(std::ios::fixed);
        osN.setf(std::ios::fixed);
        osC << std::setprecision(3) << improve_pct_ceres;
        osN << std::setprecision(3) << improve_pct_net;
        fallbackLog(0, outerTag + " net改善なし打ち切り (ceres=" + osC.str() +
                          "%, net=" + osN.str() + "%, 閾値=0.5%)");
        return FallbackOuterAction::StopOuter;
    }
    if (progress == ventilation::FallbackOuterProgress::StopNoCeresImprove) {
        double improve_pct_ceres = (state.lastCostOuter - stageB.summary.final_cost) /
                                   std::max(1e-300, state.lastCostOuter) * 100.0;
        double improve_pct_net   = (state.lastNetworkCostOuter - currNetworkCostOuter) /
                                   std::max(1e-300, state.lastNetworkCostOuter) * 100.0;
        std::ostringstream osC, osN;
        osC.setf(std::ios::fixed);
        osN.setf(std::ios::fixed);
        osC << std::setprecision(3) << improve_pct_ceres;
        osN << std::setprecision(3) << improve_pct_net;
        fallbackLog(0, outerTag + " ceres改善なし打ち切り (ceres=" + osC.str() +
                          "%, net=" + osN.str() + "%, 閾値=0.5%)");
        return FallbackOuterAction::StopOuter;
    }

    state.lastCostOuter = stageB.summary.final_cost;
    state.lastNetworkCostOuter = currNetworkCostOuter;
    fallbackLog(0, outerTag + " 継続: 次反復へ引継ぎ");
    return FallbackOuterAction::ContinueOuter;
}
