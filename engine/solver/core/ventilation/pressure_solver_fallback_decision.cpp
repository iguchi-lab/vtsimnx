#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/pressure_balance.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "utils/utils.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

PressureSolver::FallbackOuterAction PressureSolver::evaluateFallbackOuter(
        const SimulationConstants& constants,
        Graph& g,
        const SupernodePartition& partition,
        const StageASolveResult& stageA,
        StageBSolveResult& stageB,
        int outer,
        int maxOuter,
        int minOuter,
        const std::string& outerTag,
        double massBalanceMaxAbs,
        FallbackOuterState& state,
        const FallbackLogger& fallbackLog) {
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

    if (stageB.ok) {
        std::ostringstream osfb2;
        osfb2 << std::scientific << std::setprecision(6) << stageB.summary.final_cost;
        fallbackLog(0, "[Fallback] Ceres trial 完了 | cost=" + osfb2.str() + " | 外部反復 " +
                           std::to_string(outer) + "/" + std::to_string(maxOuter));

        auto evalTmp = evaluatePressureSolution(pressureMapFB_B, massBalanceMaxAbs);
        if (!evalTmp.flowOk) {
            fallbackLog(1, "[Network] 風量評価失敗: " + evalTmp.detail);
        } else {
            const auto& metricsTmp = evalTmp.solvedNodeMetrics;
            double l1 = metricsTmp.l1;
            double l2 = metricsTmp.l2;
            double costNet = 0.5 * (metricsTmp.l2 * metricsTmp.l2);
            {
                std::ostringstream osl1, osl2, osct, osmax, ospv;
                osl1.setf(std::ios::fixed);
                osl1 << std::setprecision(6) << l1;
                osl2.setf(std::ios::fixed);
                osl2 << std::setprecision(6) << l2;
                osct.setf(std::ios::fixed);
                osct << std::setprecision(6) << costNet;
                osmax.setf(std::ios::scientific);
                osmax << std::setprecision(6) << metricsTmp.maxAbs;
                std::string netLine = "[Network] L1=" + osl1.str() +
                                      " | L2=" + osl2.str() +
                                      " | cost=" + osct.str() +
                                      " | mass_maxAbs=" + osmax.str();
                if (state.lastNetworkCostOuter == std::numeric_limits<double>::infinity()) {
                    netLine += " | prev=- | 改善率=N/A";
                } else {
                    double imp_pct = (state.lastNetworkCostOuter - costNet) /
                                     std::max(1e-300, state.lastNetworkCostOuter) * 100.0;
                    ospv.setf(std::ios::fixed);
                    ospv << std::setprecision(3) << imp_pct;
                    std::ostringstream osprev;
                    osprev.setf(std::ios::fixed);
                    osprev << std::setprecision(6) << state.lastNetworkCostOuter;
                    netLine += " | prev=" + osprev.str() + " | 改善率=" + ospv.str() + "%";
                }
                fallbackLog(1, netLine);
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

            auto evalFinal = evaluatePressureSolution(pressureMapFB, massBalanceMaxAbs);
            if (!evalFinal.flowOk) {
                fallbackLog(1, "[Fallback] 質量収支評価スキップ: " + evalFinal.detail);
            } else {
                state.finalPressureMapFB = pressureMapFB;
                state.finalFlowRatesFB = std::move(evalFinal.flows);
                state.finalBalanceFB = std::move(evalFinal.allNodeBalances);
                const auto& metricsFinal = evalFinal.solvedNodeMetrics;
                {
                    std::ostringstream osmax;
                    osmax << std::scientific << std::setprecision(6) << metricsFinal.maxAbs;
                    fallbackLog(1, "[Fallback] mass_maxAbs=" + osmax.str() +
                                       " | mass_tol=" + std::to_string(massBalanceMaxAbs));
                }
                if (evalFinal.accepted) {
                    state.finalHaveSolution = true;
                    fallbackLog(0, "[Fallback] 収束 | mass_maxAbs 合格 | 外部反復 " +
                                       std::to_string(outer) + "/" + std::to_string(maxOuter));
                } else {
                    fallbackLog(0, "[Fallback] 質量収支不合格（継続）");
                }
                state.lastNetworkCostOuter = costNet;
            }
        }
    } else {
        std::ostringstream osfb2;
        osfb2.setf(std::ios::scientific);
        osfb2 << std::setprecision(6) << stageB.summary.final_cost;
        fallbackLog(0, "[Fallback] 未収束 | Ceres cost=" + osfb2.str());
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

    double currNetworkCostOuter = [&]() {
        auto evalNet = evaluatePressureSolution(pressureMapFB_B, massBalanceMaxAbs);
        if (!evalNet.flowOk) {
            return std::numeric_limits<double>::infinity();
        }
        const auto& m = evalNet.solvedNodeMetrics;
        return 0.5 * (m.l2 * m.l2);
    }();

    if (state.finalHaveSolution) {
        return FallbackOuterAction::AcceptSolution;
    }

    bool ceresImproved = stageB.summary.final_cost < state.lastCostOuter * 0.995;
    bool netImproved   = currNetworkCostOuter < state.lastNetworkCostOuter * 0.995;
    // Ceresの方が改善している場合は継続、netのみで打ち切り判定
    if (outer >= minOuter && ceresImproved && !netImproved) {
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
    // Ceresも改善していない場合も打ち切り
    if (outer >= minOuter && !ceresImproved) {
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
