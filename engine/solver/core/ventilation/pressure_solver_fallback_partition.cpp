#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/pressure_balance.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "../archenv/include/archenv.h"

#include <algorithm>
#include <cmath>
#include <vector>

PressureSolver::SupernodePartition PressureSolver::detectSupernodePartition(
        const SimulationConstants& constants,
        const PressureMap& currentPressures) {
    Graph& g = network_.getGraph();
    SupernodePartition partition;

    const size_t vCount = static_cast<size_t>(boost::num_vertices(g));
    partition.incidentEdgesByVertex.assign(vCount, {});
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        Vertex sv = boost::source(e, g);
        Vertex tv = boost::target(e, g);
        partition.incidentEdgesByVertex[static_cast<size_t>(sv)].push_back(e);
        partition.incidentEdgesByVertex[static_cast<size_t>(tv)].push_back(e);
    }

    // 室内同士の高コンダクタンスエッジ抽出（相対閾値: median*ratio）
    std::vector<Edge> candidateEdges;
    std::vector<double> conductances;
    std::vector<double> dpAbsList;
    std::vector<std::string> types;
    candidateEdges.reserve(boost::num_edges(g));
    conductances.reserve(boost::num_edges(g));
    dpAbsList.reserve(boost::num_edges(g));

    auto erange = boost::edges(g);
    for (auto e : boost::make_iterator_range(erange)) {
        const auto& ep = g[e];
        if (!(ep.type == "gap" || ep.type == "simple_opening" || ep.type == "pressure_loss")) continue;

        auto sv = boost::source(e, g);
        auto tv = boost::target(e, g);
        const auto& sn = g[sv];
        const auto& tn = g[tv];

        // スーパーノード候補は室内同士（calc_p）に限定
        if (!sn.calc_p || !tn.calc_p) continue;

        // 現状の圧力差（静水圧補正込み）を推定
        double p_s = currentPressures.count(sn.key) ? currentPressures.at(sn.key) : sn.current_p;
        double p_t = currentPressures.count(tn.key) ? currentPressures.at(tn.key) : tn.current_p;
        double p_s_total = calculateTotalPressure(p_s, sn.current_t, ep.h_from);
        double p_t_total = calculateTotalPressure(p_t, tn.current_t, ep.h_to);
        double p_st = p_s_total - p_t_total;

        // 統一近似導関数 dQ/dp をタイプごとに評価
        double dp_abs = std::max(archenv::TOLERANCE_SMALL, std::abs(p_st));
        double G = 0.0;
        if (ep.type == "simple_opening") {
            double K = ep.alpha * ep.area * std::sqrt(2.0 / archenv::DENSITY_DRY_AIR);
            G = 0.5 * K / std::sqrt(dp_abs);
        } else if (ep.type == "gap") {
            double n = (ep.n != 0.0) ? ep.n : 1.0;
            G = (ep.a / n) * std::pow(dp_abs, (1.0 / n) - 1.0);
        } else if (ep.type == "pressure_loss") {
            double k_total = ep.k_total;
            if (!(k_total > 0.0) && ep.friction_factor > 0.0 && ep.length >= 0.0 && ep.diameter > 0.0) {
                k_total = ep.friction_factor * ep.length / ep.diameter + ep.zeta_total;
            }
            if (ep.area > 0.0 && k_total > 0.0) {
                const double C = ep.area * std::sqrt(2.0 / (archenv::DENSITY_DRY_AIR * k_total));
                G = 0.5 * C / std::sqrt(dp_abs);
            }
        }

        candidateEdges.push_back(e);
        conductances.push_back(G);
        dpAbsList.push_back(dp_abs);
        types.push_back(ep.type);
    }

    auto vrange = boost::vertices(g);
    for (auto v : boost::make_iterator_range(vrange)) partition.vertices.push_back(v);
    for (size_t i = 0; i < partition.vertices.size(); ++i) {
        partition.v2i[partition.vertices[i]] = static_cast<int>(i);
    }
    partition.groupOfVertex.assign(partition.vertices.size(), -1);

    if (!conductances.empty()) {
        const double epsG = 1e-16;
        std::vector<double> logG;
        logG.reserve(conductances.size());
        for (double G : conductances) {
            logG.push_back(std::log10(std::max(G, epsG)));
        }

        double logMin = *std::min_element(logG.begin(), logG.end());
        double logMax = *std::max_element(logG.begin(), logG.end());
        // 候補が1本、または log 空間で明確な幅が無い場合はスーパーノード化しない
        if (logG.size() < 2 || (logMax - logMin) < 1.0) {
            writeLog(logFile_, "\t\tコンダクタンス幅不足のためスーパーノード化をスキップ"
                                 " (n=" + std::to_string(logG.size()) +
                                 ", logSpan=" + std::to_string(logMax - logMin) + ")");
            return partition;
        }

        double cLow = logMin;
        double cHigh = logMax;
        std::vector<int> assign(logG.size(), 0);
        bool clusteringOk = true;
        for (int it = 0; it < 10; ++it) {
            for (size_t i = 0; i < logG.size(); ++i) {
                double dL = std::abs(logG[i] - cLow);
                double dH = std::abs(logG[i] - cHigh);
                assign[i] = (dH < dL) ? 1 : 0;
            }
            double sumL = 0.0, sumH = 0.0;
            size_t cntL = 0, cntH = 0;
            for (size_t i = 0; i < logG.size(); ++i) {
                if (assign[i] == 0) {
                    sumL += logG[i];
                    cntL++;
                } else {
                    sumH += logG[i];
                    cntH++;
                }
            }
            // 空クラスタの中心を人工的に離さない（偽の decade 分離を作らない）
            if (cntL == 0 || cntH == 0) {
                clusteringOk = false;
                break;
            }
            cLow = sumL / static_cast<double>(cntL);
            cHigh = sumH / static_cast<double>(cntH);
        }
        if (!clusteringOk) {
            writeLog(logFile_, "\t\t空クラスタ発生のためスーパーノード化をスキップ");
            return partition;
        }
        if (cLow > cHigh) std::swap(cLow, cHigh);
        double cMid = 0.5 * (cLow + cHigh);
        size_t cntLFinal = 0, cntHFinal = 0;
        for (int a : assign) {
            if (a == 0) cntLFinal++; else cntHFinal++;
        }
        if (cntLFinal == 0 || cntHFinal == 0) {
            writeLog(logFile_, "\t\t最終割当が片側クラスタのためスーパーノード化をスキップ");
            return partition;
        }
        if (constants.logFallbackDetails && constants.logVerbosity >= 2) {
            writeLog(logFile_, "\t\tクラスタ分離(logG): cLow=" + std::to_string(cLow) +
                                 ", cHigh=" + std::to_string(cHigh));
            writeLog(logFile_, "\t\tクラスタサイズ: low=" + std::to_string(cntLFinal) +
                                 ", high=" + std::to_string(cntHFinal));
            writeLog(logFile_, "\t\t選抜閾値(logG中点): " + std::to_string(cMid));
        }

        if (!ventilation::hasClearConductanceSeparation(cLow, cHigh)) {
            writeLog(logFile_, "\t\t明確な強弱分離なしのためスーパーノード化をスキップ"
                                 " (logSep=" + std::to_string(cHigh - cLow) +
                                 ", ratio=" + std::to_string(std::pow(10.0, cHigh - cLow)) + ")");
            return partition;
        }

        std::vector<std::vector<int>> adj(partition.vertices.size());
        std::vector<char> selected(candidateEdges.size(), 0);
        size_t selectedCount = 0;
        for (size_t i = 0; i < candidateEdges.size(); ++i) {
            if (!(assign[i] == 1 || logG[i] >= cMid)) continue;
            int si = partition.v2i[boost::source(candidateEdges[i], g)];
            int ti = partition.v2i[boost::target(candidateEdges[i], g)];
            adj[si].push_back(ti);
            adj[ti].push_back(si);
            partition.highEdgeCount++;
            selected[i] = 1;
            selectedCount++;
        }

        if (selectedCount == 0) {
            std::vector<double> sortedN = logG;
            std::sort(sortedN.begin(), sortedN.end());
            size_t k = std::max<size_t>(1, sortedN.size() / 10);
            double thrP = sortedN[sortedN.size() - k];
            writeLog(logFile_, "\t\t選抜ゼロのためlogGパーセンタイル閾値に切替: p90=" + std::to_string(thrP));
            adj.assign(partition.vertices.size(), {});
            selected.assign(candidateEdges.size(), 0);
            partition.highEdgeCount = 0;
            for (size_t i = 0; i < candidateEdges.size(); ++i) {
                if (logG[i] < thrP) continue;
                int si = partition.v2i[boost::source(candidateEdges[i], g)];
                int ti = partition.v2i[boost::target(candidateEdges[i], g)];
                adj[si].push_back(ti);
                adj[ti].push_back(si);
                partition.highEdgeCount++;
                selected[i] = 1;
            }
        }

        int gid = 0;
        std::vector<char> vis(partition.vertices.size(), 0);
        for (size_t i = 0; i < partition.vertices.size(); ++i) {
            if (vis[i]) continue;
            std::vector<int> st{static_cast<int>(i)};
            std::vector<int> comp;
            vis[i] = 1;
            while (!st.empty()) {
                int u = st.back();
                st.pop_back();
                comp.push_back(u);
                for (int w : adj[u]) {
                    if (!vis[w]) {
                        vis[w] = 1;
                        st.push_back(w);
                    }
                }
            }
            if (comp.size() >= 2) {
                for (int u : comp) partition.groupOfVertex[u] = gid;
                partition.condensedNodeCount += comp.size();
                gid++;
            }
        }

        int superCount = *std::max_element(partition.groupOfVertex.begin(), partition.groupOfVertex.end()) + 1;
        if (constants.logFallbackDetails && constants.logVerbosity >= 2) {
            writeLog(logFile_, "\t\tスーパーノード数: " + std::to_string(superCount));
            writeLog(logFile_, "\t\t高コンダクタンスエッジ本数: " + std::to_string(partition.highEdgeCount));
            writeLog(logFile_, "\t\tスーパーノード化対象ノード数: " + std::to_string(partition.condensedNodeCount));
        }

        if (superCount > 0 && constants.logFallbackDetails && constants.logVerbosity >= 2) {
            std::vector<std::vector<std::string>> groups(superCount);
            for (size_t i = 0; i < partition.vertices.size(); ++i) {
                int gidv = partition.groupOfVertex[i];
                if (gidv >= 0) groups[gidv].push_back(g[partition.vertices[i]].key);
            }
            for (int gidv = 0; gidv < superCount; ++gidv) {
                std::string line = "\t\tスーパーノード #" + std::to_string(gidv) + ": ";
                for (size_t k = 0; k < groups[gidv].size(); ++k) {
                    line += groups[gidv][k];
                    if (k + 1 < groups[gidv].size()) line += ", ";
                }
                writeLog(logFile_, line);
            }
        }
    } else {
        writeLog(logFile_, "\t\tスーパーノード候補エッジがありません");
    }

    return partition;
}
