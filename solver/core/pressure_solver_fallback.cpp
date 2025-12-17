#include "core/pressure_solver.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "../archenv/include/archenv.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

// =============================================================================
// Fallbackループ用ユーティリティ関数
// =============================================================================

void PressureSolver::restoreFixedFlowEdges(
    Graph& graph,
    std::vector<std::string>& changedEdgeIds,
    const std::map<std::string, std::string>& interfaceOriginalTypeById) {
    if (changedEdgeIds.empty()) {
        return;
    }

    std::unordered_set<std::string> changedIdSet(changedEdgeIds.begin(), changedEdgeIds.end());
    auto edgeRange = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edgeRange)) {
        auto& edgeProps = graph[edge];
        if (!changedIdSet.count(edgeProps.unique_id)) {
            continue;
        }
        auto itType = interfaceOriginalTypeById.find(edgeProps.unique_id);
        if (itType != interfaceOriginalTypeById.end()) {
            edgeProps.type = itType->second;
            edgeProps.current_vol = 0.0;
        }
    }
    changedEdgeIds.clear();
}

std::map<std::string, std::string> PressureSolver::captureInterfaceOriginalTypes(
    Graph& graph,
    const std::map<Vertex, int>& vertexToIndex,
    const std::vector<int>& groupOfVertex) {
    std::map<std::string, std::string> result;
    auto edgeRange = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edgeRange)) {
        auto sv = boost::source(edge, graph);
        auto tv = boost::target(edge, graph);
        auto itS = vertexToIndex.find(sv);
        auto itT = vertexToIndex.find(tv);
        if (itS == vertexToIndex.end() || itT == vertexToIndex.end()) {
            continue;
        }
        int gidS = groupOfVertex[itS->second];
        int gidT = groupOfVertex[itT->second];
        bool crossCluster = (gidS != gidT) && (gidS >= 0 || gidT >= 0);
        if (!crossCluster) {
            continue;
        }
        result[graph[edge].unique_id] = graph[edge].type;
    }
    return result;
}

// =============================================================================
// Fallbackメインループ
// =============================================================================

std::optional<PressureSolver::SolverResult> PressureSolver::runFallbackLoop(
        const SimulationConstants& constants,
        SolverSetup& setup,
        ceres::Solver::Summary& summary) {
    auto& nodeNames = setup.nodeNames;
    auto& pressures = setup.pressures;

    writeLog(logFile_, "圧力計算フォールバック");
    auto fallbackLog = [&](int indent, const std::string& message) {
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
    PressureMap prevPressureMapFB = currentPressures; // 外部反復の初期圧力
    double lastCostOuter = std::numeric_limits<double>::infinity();
    double lastNetworkCostOuter = std::numeric_limits<double>::infinity();

    if (constants.logVerbosity >= 1) {
        fallbackLog(0, "[Fallback] スーパーノード化 + 外気ギャップ固定流量化を適用します");
    }

    // 個別ブランチ流量（固定流量化の初期値推定に使用）
    std::map<std::string, double> currentIndividualFlows = calculateIndividualFlowRates(currentPressures);

    Graph& g = network_.getGraph();

    // トポロジはフォールバック中は不変なので、incident edges を一度だけ構築して使い回す
    const size_t vCount = static_cast<size_t>(boost::num_vertices(g));
    std::vector<std::vector<Edge>> incidentEdgesByVertex(vCount);
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        Vertex sv = boost::source(e, g);
        Vertex tv = boost::target(e, g);
        incidentEdgesByVertex[static_cast<size_t>(sv)].push_back(e);
        incidentEdgesByVertex[static_cast<size_t>(tv)].push_back(e);
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
        if (!(ep.type == "gap" || ep.type == "simple_opening")) continue;

        auto sv = boost::source(e, g);
        auto tv = boost::target(e, g);
        const auto& sn = g[sv];
        const auto& tn = g[tv];

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
        }

        candidateEdges.push_back(e);
        conductances.push_back(G);
        dpAbsList.push_back(dp_abs);
        types.push_back(ep.type);
    }

    size_t highEdgeCount = 0;
    size_t condensedNodeCount = 0;
    size_t fixedFlowCount = 0;

    std::vector<Vertex> vertices;
    auto vrange = boost::vertices(g);
    for (auto v : boost::make_iterator_range(vrange)) vertices.push_back(v);
    std::map<Vertex, int> v2i;
    for (size_t i = 0; i < vertices.size(); ++i) v2i[vertices[i]] = static_cast<int>(i);
    std::vector<int> groupOfVertex(vertices.size(), -1);

    if (!conductances.empty()) {
        const double epsG = 1e-16;
        std::vector<double> logG;
        logG.reserve(conductances.size());
        for (double G : conductances) {
            logG.push_back(std::log10(std::max(G, epsG)));
        }

        double logMin = *std::min_element(logG.begin(), logG.end());
        double logMax = *std::max_element(logG.begin(), logG.end());
        double cLow = logMin;
        double cHigh = logMax;
        std::vector<int> assign(logG.size(), 0);
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
            if (cntL > 0) cLow = sumL / cntL; else cLow = cHigh - 1.0;
            if (cntH > 0) cHigh = sumH / cntH; else cHigh = cLow + 1.0;
        }
        if (cLow > cHigh) std::swap(cLow, cHigh);
        double cMid = 0.5 * (cLow + cHigh);
        if (constants.logFallbackDetails && constants.logVerbosity >= 2) {
            writeLog(logFile_, "\t\tクラスタ分離(logG): cLow=" + std::to_string(cLow) +
                                 ", cHigh=" + std::to_string(cHigh));
            size_t cntL = 0, cntH = 0;
            for (int a : assign) {
                if (a == 0) cntL++; else cntH++;
            }
            writeLog(logFile_, "\t\tクラスタサイズ: low=" + std::to_string(cntL) + ", high=" + std::to_string(cntH));
            writeLog(logFile_, "\t\t選抜閾値(logG中点): " + std::to_string(cMid));
        }

        std::vector<std::vector<int>> adj(vertices.size());
        std::vector<char> selected(candidateEdges.size(), 0);
        size_t selectedCount = 0;
        for (size_t i = 0; i < candidateEdges.size(); ++i) {
            if (!(assign[i] == 1 || logG[i] >= cMid)) continue;
            int si = v2i[boost::source(candidateEdges[i], g)];
            int ti = v2i[boost::target(candidateEdges[i], g)];
            adj[si].push_back(ti);
            adj[ti].push_back(si);
            highEdgeCount++;
            selected[i] = 1;
            selectedCount++;
        }

        if (selectedCount == 0) {
            std::vector<double> sortedN = logG;
            std::sort(sortedN.begin(), sortedN.end());
            size_t k = std::max<size_t>(1, sortedN.size() / 10);
            double thrP = sortedN[sortedN.size() - k];
            writeLog(logFile_, "\t\t選抜ゼロのためlogGパーセンタイル閾値に切替: p90=" + std::to_string(thrP));
            adj.assign(vertices.size(), {});
            selected.assign(candidateEdges.size(), 0);
            highEdgeCount = 0;
            for (size_t i = 0; i < candidateEdges.size(); ++i) {
                if (logG[i] < thrP) continue;
                int si = v2i[boost::source(candidateEdges[i], g)];
                int ti = v2i[boost::target(candidateEdges[i], g)];
                adj[si].push_back(ti);
                adj[ti].push_back(si);
                highEdgeCount++;
                selected[i] = 1;
            }
        }

        int gid = 0;
        std::vector<char> vis(vertices.size(), 0);
        for (size_t i = 0; i < vertices.size(); ++i) {
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
                for (int u : comp) groupOfVertex[u] = gid;
                condensedNodeCount += comp.size();
                gid++;
            }
        }

        int superCount = *std::max_element(groupOfVertex.begin(), groupOfVertex.end()) + 1;
        if (constants.logFallbackDetails && constants.logVerbosity >= 2) {
            writeLog(logFile_, "\t\tスーパーノード数: " + std::to_string(superCount));
            writeLog(logFile_, "\t\t高コンダクタンスエッジ本数: " + std::to_string(highEdgeCount));
            writeLog(logFile_, "\t\tスーパーノード化対象ノード数: " + std::to_string(condensedNodeCount));
        }

        if (superCount > 0 && constants.logFallbackDetails && constants.logVerbosity >= 2) {
            std::vector<std::vector<std::string>> groups(superCount);
            for (size_t i = 0; i < vertices.size(); ++i) {
                int gidv = groupOfVertex[i];
                if (gidv >= 0) groups[gidv].push_back(g[vertices[i]].key);
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

    std::map<std::string, std::string> interfaceOriginalTypeById =
        captureInterfaceOriginalTypes(g, v2i, groupOfVertex);

    const int maxOuter = 5;
    const int minOuter = 2;
    std::vector<std::string> changedEdgeIds;
    PressureMap finalPressureMapFB;
    FlowRateMap finalFlowRatesFB;
    FlowBalanceMap finalBalanceFB;
    bool finalHaveSolution = false;

    for (int outer = 1; outer <= maxOuter; ++outer) {
        restoreFixedFlowEdges(g, changedEdgeIds, interfaceOriginalTypeById);

        const std::string outerTag = "[外部反復 " + std::to_string(outer) + "/" + std::to_string(maxOuter) + "]";
        std::string prevCostText = "prev=-";
        if (!std::isinf(lastCostOuter)) {
            prevCostText = "prev=" + formatScientific(lastCostOuter);
        }
        fallbackLog(0, outerTag + " 開始 | " + prevCostText);

        fallbackLog(1, "[A] スーパーノード代表圧フェーズ開始" + std::string(outer >= 2 ? " | source=B(prev)" : " | source=A(current)"));
        StageAMapping stageMapping = buildStageAMapping(g, vertices, groupOfVertex);
        auto& vToParamIdx = stageMapping.vertexToParamIndex;
        std::vector<std::string> nodeNamesFB = stageMapping.nodeNames;
        std::vector<double> pressuresFB = initializeStageAPressures(g, stageMapping, prevPressureMapFB);

        ceres::Problem problemFB;
        int superCountA = *std::max_element(groupOfVertex.begin(), groupOfVertex.end()) + 1;
        double anchorTargetPressureA = 0.0;
        bool hasAnchorTargetA = false;
        setupStageAProblem(
            problemFB,
            stageMapping,
            g,
            vertices,
            groupOfVertex,
            prevPressureMapFB,
            pressuresFB,
            superCountA,
            incidentEdgesByVertex);

        ceres::Solver::Summary fbSummary;
        bool fbOKA = false;
        auto runStageATrial = [&](const std::string& label,
                                  const std::function<void(ceres::Solver::Options&)>& configure,
                                  bool logTolerance = false,
                                  double customTol = 0.0) {
            if (fbOKA) return;
            fallbackLog(2, label);
            ceres::Solver::Options options;
            configure(options);
            double usedTolerance = options.function_tolerance;  // 設定された許容誤差を使用
            if (logTolerance) {
                fallbackLog(3, "[A] 調整済み許容誤差=" + std::to_string(customTol));
            }
            ceres::Solve(options, &problemFB, &fbSummary);
            logCeresTiming(label, fbSummary);
            // フォールバックでは設定した許容誤差で判定（調整済み許容誤差での収束を許容）
            fbOKA = (fbSummary.termination_type == ceres::CONVERGENCE) &&
                    (fbSummary.final_cost <= usedTolerance);
            if (fbOKA) {
                std::ostringstream os;
                os << std::scientific << std::setprecision(6) << fbSummary.final_cost;
                fallbackLog(2, label + " 収束 | residual=" + os.str() + " | tol=" +
                                   std::to_string(usedTolerance));
            }
        };

        runStageATrial("[A-①] 標準設定でソルバーを実行します", [&](ceres::Solver::Options& o) {
            o.linear_solver_type = ceres::DENSE_QR;
            o.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            o.max_num_iterations = constants.maxInnerIteration;
            o.function_tolerance = constants.ventilationTolerance;
            o.parameter_tolerance = constants.ventilationTolerance;
            o.minimizer_progress_to_stdout = false;
        });

        runStageATrial("[A-②] 堅牢設定でソルバーを再実行します", [&](ceres::Solver::Options& o) {
            o.trust_region_strategy_type = ceres::DOGLEG;
            o.linear_solver_type = ceres::DENSE_QR;
            o.max_num_iterations = std::max(500, static_cast<int>(constants.maxInnerIteration * 2));
            o.function_tolerance = constants.ventilationTolerance * 0.01;
            o.parameter_tolerance = constants.ventilationTolerance * 0.01;
            o.gradient_tolerance = constants.ventilationTolerance * 0.1;
            o.jacobi_scaling = true;
            o.use_inner_iterations = true;
            o.max_trust_region_radius = 1e4;
            o.initial_trust_region_radius = 1e2;
            o.minimizer_progress_to_stdout = false;
        });

        runStageATrial("[A-③] DENSE_SCHUR設定でソルバーを再実行します", [&](ceres::Solver::Options& o) {
            o.trust_region_strategy_type = ceres::DOGLEG;
            o.linear_solver_type = ceres::DENSE_SCHUR;
            o.max_num_iterations = 500;
            o.function_tolerance = constants.ventilationTolerance * 0.01;
            o.parameter_tolerance = constants.ventilationTolerance * 0.01;
            o.gradient_tolerance = constants.ventilationTolerance * 0.1;
            o.jacobi_scaling = true;
            o.use_inner_iterations = true;
            o.max_trust_region_radius = 1e4;
            o.initial_trust_region_radius = 1e2;
            o.minimizer_progress_to_stdout = false;
        });

        runStageATrial("[A-④] SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します", [&](ceres::Solver::Options& o) {
            o.trust_region_strategy_type = ceres::DOGLEG;
            o.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            o.max_num_iterations = 1000;
            o.function_tolerance = constants.ventilationTolerance * 0.001;
            o.parameter_tolerance = constants.ventilationTolerance * 0.001;
            o.gradient_tolerance = constants.ventilationTolerance * 0.01;
            o.jacobi_scaling = true;
            o.use_inner_iterations = true;
            o.inner_iteration_tolerance = 1e-8;
            o.max_trust_region_radius = 1e3;
            o.initial_trust_region_radius = 1e1;
            o.minimizer_progress_to_stdout = false;
        });

        if (!fbOKA) {
            fallbackLog(2, "[A-⑤] 段階的緩和法でソルバーを再実行します");
            ceres::Solver::Options o1;
            o1.trust_region_strategy_type = ceres::DOGLEG;
            o1.linear_solver_type = ceres::DENSE_QR;
            o1.max_num_iterations = 200;
            o1.function_tolerance = constants.ventilationTolerance * 10;
            o1.parameter_tolerance = constants.ventilationTolerance * 10;
            o1.gradient_tolerance = constants.ventilationTolerance;
            o1.jacobi_scaling = true;
            o1.minimizer_progress_to_stdout = false;
            ceres::Solve(o1, &problemFB, &fbSummary);
            logCeresTiming("[A-⑤] 段階1", fbSummary);
            {
                std::ostringstream os;
                os << std::scientific << std::setprecision(6) << fbSummary.final_cost;
                fallbackLog(3, "[A-⑤] 段階1完了 | residual=" + os.str());
            }
            ceres::Solver::Options o2;
            o2.trust_region_strategy_type = ceres::DOGLEG;
            o2.linear_solver_type = ceres::DENSE_QR;
            o2.max_num_iterations = 1000;
            o2.function_tolerance = constants.ventilationTolerance * 0.01;
            o2.parameter_tolerance = constants.ventilationTolerance * 0.01;
            o2.gradient_tolerance = constants.ventilationTolerance * 0.1;
            o2.jacobi_scaling = true;
            o2.use_inner_iterations = true;
            o2.minimizer_progress_to_stdout = false;
            ceres::Solve(o2, &problemFB, &fbSummary);
            logCeresTiming("[A-⑤] 段階2", fbSummary);
            // 段階的緩和法の段階2では、設定した許容誤差で判定（調整済み許容誤差での収束を許容）
            double usedTolerance = o2.function_tolerance;
            fbOKA = (fbSummary.termination_type == ceres::CONVERGENCE) &&
                    (fbSummary.final_cost <= usedTolerance);
            if (fbOKA) {
                std::ostringstream os;
                os << std::scientific << std::setprecision(6) << fbSummary.final_cost;
                fallbackLog(2, "[A-⑤] 収束 | residual=" + os.str() + " | tol=" +
                                   std::to_string(usedTolerance));
            }
        }

        runStageATrial("[A-⑥] Line Search方式でソルバーを再実行します", [&](ceres::Solver::Options& o) {
            o.minimizer_type = ceres::LINE_SEARCH;
            o.line_search_direction_type = ceres::LBFGS;
            o.line_search_type = ceres::WOLFE;
            o.max_num_iterations = 1000;
            o.function_tolerance = constants.ventilationTolerance;
            o.parameter_tolerance = constants.ventilationTolerance;
            o.gradient_tolerance = constants.ventilationTolerance * 10;
            o.jacobi_scaling = true;
            o.minimizer_progress_to_stdout = false;
        });

        if (!fbOKA) {
            fallbackLog(2, "[A-⑦] 超精密設定で最終試行します");
            ceres::Solver::Options o;
            o.trust_region_strategy_type = ceres::DOGLEG;
            o.linear_solver_type = ceres::DENSE_QR;
            o.max_num_iterations = 5000;
            double tolerance_factor = std::max(1.0, fbSummary.final_cost / constants.ventilationTolerance * 0.1);
            o.function_tolerance = constants.ventilationTolerance * tolerance_factor;
            o.parameter_tolerance = constants.ventilationTolerance * tolerance_factor;
            o.gradient_tolerance = constants.ventilationTolerance * tolerance_factor * 10;
            o.jacobi_scaling = true;
            o.use_inner_iterations = true;
            o.inner_iteration_tolerance = 1e-12;
            o.max_trust_region_radius = 1e2;
            o.initial_trust_region_radius = 1e0;
            o.min_trust_region_radius = 1e-8;
            fallbackLog(3, "[A-⑦] 調整済み許容誤差=" + std::to_string(o.function_tolerance));
            ceres::Solve(o, &problemFB, &fbSummary);
            logCeresTiming("[A-⑦] 超精密設定", fbSummary);
            fbOKA = (fbSummary.termination_type == ceres::CONVERGENCE) &&
                    (fbSummary.final_cost <= o.function_tolerance);
            if (fbOKA) {
                std::ostringstream os;
                os << std::scientific << std::setprecision(6) << fbSummary.final_cost;
                fallbackLog(2, "[A-⑦] 収束 | residual=" + os.str() + " | tol=" +
                                   std::to_string(o.function_tolerance));
            }
        }

        if (!fbOKA) {
            std::ostringstream os;
            os.setf(std::ios::scientific);
            os << std::setprecision(6) << fbSummary.final_cost;
            fallbackLog(2, "[A] 未収束 | residual=" + os.str() +
                               " | tol=" + std::to_string(constants.ventilationTolerance));
        }

        PressureMap pressureMapFB_A;
        auto vrA = boost::vertices(g);
        for (auto v : boost::make_iterator_range(vrA)) {
            const auto& node = g[v];
            if (node.calc_p) {
                size_t idx = vToParamIdx[v];
                pressureMapFB_A[node.key] = pressuresFB[idx];
            } else {
                pressureMapFB_A[node.key] = node.current_p;
            }
        }

        if (superCountA > 0) {
            double sumG0 = 0.0;
            int cntG0 = 0;
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (groupOfVertex[i] != 0) continue;
                const auto& node = g[vertices[i]];
                if (!node.calc_p) continue;
                auto it = pressureMapFB_A.find(node.key);
                if (it != pressureMapFB_A.end()) {
                    sumG0 += it->second;
                    cntG0++;
                }
            }
            if (cntG0 > 0) {
                anchorTargetPressureA = sumG0 / static_cast<double>(cntG0);
                hasAnchorTargetA = true;
            }
        }

        auto applyFixedFlows = [&]() {
            auto er2 = boost::edges(g);
            size_t alreadyFixed = 0;
            fixedFlowCount = 0;
            PressureMap pressureMapForFixed = (outer >= 2) ? prevPressureMapFB : pressureMapFB_A;
            std::map<std::string, double> indivA = calculateIndividualFlowRates(pressureMapForFixed);

            auto erInt = boost::edges(g);
            for (auto e : boost::make_iterator_range(erInt)) {
                auto sv = boost::source(e, g);
                auto tv = boost::target(e, g);
                auto itS = vToParamIdx.find(sv);
                auto itT = vToParamIdx.find(tv);
                if (itS != vToParamIdx.end() && itT != vToParamIdx.end() && itS->second == itT->second) {
                    const auto& epz = g[e];
                    auto itId = indivA.find(epz.unique_id);
                    if (itId != indivA.end()) itId->second = 0.0;
                }
            }

            for (auto e : boost::make_iterator_range(er2)) {
                auto sv = boost::source(e, g);
                auto tv = boost::target(e, g);
                auto& ep = g[e];

                int si = v2i[sv];
                int ti = v2i[tv];
                int gidS = groupOfVertex[si];
                int gidT = groupOfVertex[ti];
                bool crossCluster = (gidS != gidT) && (gidS >= 0 || gidT >= 0);
                if (!crossCluster) continue;
                if (ep.type == "fixed_flow") {
                    alreadyFixed++;
                    continue;
                }

                auto itf = indivA.find(ep.unique_id);
                if (itf != indivA.end()) {
                    ep.current_vol = itf->second;
                    ep.type = "fixed_flow";
                    fixedFlowCount++;
                    changedEdgeIds.push_back(ep.unique_id);
                }
            }
            const std::string flowSource = (outer >= 2) ? "source=B(prev)" : "source=A(current)";
            fallbackLog(2, "[A] 固定流量化ブランチ=" + std::to_string(fixedFlowCount) +
                              " | already_fixed=" + std::to_string(alreadyFixed) +
                              " | " + flowSource);
        };

        applyFixedFlows();

        fallbackLog(1, "[B] 固定流量下でフルノード再解フェーズ開始");
        StageBSetup stageBSetup = buildStageBSetup(g, pressureMapFB_A);
        ceres::Problem problemFB2;
        auto& vToParamIdxB = stageBSetup.vertexToParamIndex;
        const auto& nodeNamesFBB = stageBSetup.nodeNames;
        std::vector<double>& pressuresFBB = stageBSetup.pressures;

        for (const auto& nodeName : nodeNamesFBB) {
            ceres::CostFunction* costFunction = new FlowBalanceConstraint(
                nodeName,
                g,
                network_.getKeyToVertex(),
                stageBSetup.vertexToParamIndexVec,
                incidentEdgesByVertex,
                pressuresFBB.size(),
                logFile_
            );
            problemFB2.AddResidualBlock(costFunction, nullptr, pressuresFBB.data());
        }
        if (!nodeNamesFBB.empty()) {
            SoftAnchorConstraint* constraint = new SoftAnchorConstraint(0, 0.0, 1e-9, pressuresFBB.size());
            problemFB2.AddResidualBlock(constraint, nullptr, pressuresFBB.data());
        }

        ceres::Solver::Summary fbSummary2;
        bool fbOK2 = runStageBTrials(constants, problemFB2, fbSummary2, fallbackLog);

        PressureMap pressureMapFB_B;
        auto vrB = boost::vertices(g);
        for (auto v : boost::make_iterator_range(vrB)) {
            const auto& node = g[v];
            if (node.calc_p) {
                size_t idx = vToParamIdxB[v];
                pressureMapFB_B[node.key] = pressuresFBB[idx];
            } else {
                pressureMapFB_B[node.key] = node.current_p;
            }
        }

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
            osCurr << std::setprecision(6) << fbSummary2.final_cost;
            if (lastCostOuter == std::numeric_limits<double>::infinity()) {
                fallbackLog(0, outerTag + " 結果: cost=" + osCurr.str() + ", 改善率=N/A");
            } else {
                double improve_pct = (lastCostOuter - fbSummary2.final_cost) / std::max(1e-300, lastCostOuter) * 100.0;
                std::ostringstream osPct;
                osPct.setf(std::ios::fixed);
                osPct << std::setprecision(3) << improve_pct;
                std::ostringstream osPrev;
                osPrev.setf(std::ios::scientific);
                osPrev << std::setprecision(6) << lastCostOuter;
                fallbackLog(0, outerTag + " 結果: prev=" + osPrev.str() + ", curr=" + osCurr.str() +
                                   ", 改善率=" + osPct.str() + "%");
            }
        }

        if (fbOK2) {
            std::ostringstream osfb2;
            osfb2 << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(0, "[Fallback] 収束 | residual=" + osfb2.str() + " | 外部反復 " +
                               std::to_string(outer) + "/" + std::to_string(maxOuter));

            FlowRateMap flowRatesFB_tmp = calculateFlowRates(pressureMapFB_B);
            FlowBalanceMap balanceFB_tmp = verifyBalance(flowRatesFB_tmp);
            double l1 = 0.0, sumsq = 0.0;
            for (const auto& kvb : balanceFB_tmp) {
                l1 += std::abs(kvb.second);
                sumsq += kvb.second * kvb.second;
            }
            double l2 = std::sqrt(sumsq);
            double costNet = 0.5 * sumsq;
            {
                std::ostringstream osl1, osl2, osct, ospv;
                osl1.setf(std::ios::fixed);
                osl1 << std::setprecision(6) << l1;
                osl2.setf(std::ios::fixed);
                osl2 << std::setprecision(6) << l2;
                osct.setf(std::ios::fixed);
                osct << std::setprecision(6) << costNet;
                std::string netLine = "[Network] L1=" + osl1.str() +
                                      " | L2=" + osl2.str() +
                                      " | cost=" + osct.str();
                if (lastNetworkCostOuter == std::numeric_limits<double>::infinity()) {
                    netLine += " | prev=- | 改善率=N/A";
                } else {
                    double imp_pct = (lastNetworkCostOuter - costNet) / std::max(1e-300, lastNetworkCostOuter) * 100.0;
                    ospv.setf(std::ios::fixed);
                    ospv << std::setprecision(3) << imp_pct;
                    std::ostringstream osprev;
                    osprev.setf(std::ios::fixed);
                    osprev << std::setprecision(6) << lastNetworkCostOuter;
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

            if (hasAnchorTargetA) {
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
                    double offset = anchorTargetPressureA - meanG0;
                    for (size_t i = 0; i < vertices.size(); ++i) {
                        if (groupOfVertex[i] != 0) continue;
                        const auto& ndG0 = g[vertices[i]];
                        if (!ndG0.calc_p) continue;
                        auto itp = pressureMapFB.find(ndG0.key);
                        if (itp != pressureMapFB.end()) itp->second += offset;
                    }
                    std::ostringstream osa;
                    osa.setf(std::ios::fixed);
                    osa << std::setprecision(6) << anchorTargetPressureA;
                    fallbackLog(1, "[Gauge] G0平均を " + osa.str() + " Pa に合わせました");
                }
            }

            finalPressureMapFB = pressureMapFB;
            finalFlowRatesFB = calculateFlowRates(finalPressureMapFB);
            finalBalanceFB = verifyBalance(finalFlowRatesFB);
            finalHaveSolution = true;
            lastNetworkCostOuter = costNet;
        } else {
            std::ostringstream osfb2;
            osfb2.setf(std::ios::scientific);
            osfb2 << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(0, "[Fallback] 未収束 | residual=" + osfb2.str() +
                               " | tol=" + std::to_string(constants.ventilationTolerance));
        }

        prevPressureMapFB.clear();
        auto vrB2 = boost::vertices(g);
        for (auto v : boost::make_iterator_range(vrB2)) {
            const auto& node = g[v];
            if (node.calc_p) {
                size_t idx = vToParamIdxB[v];
                prevPressureMapFB[node.key] = pressuresFBB[idx];
            } else {
                prevPressureMapFB[node.key] = node.current_p;
            }
        }

        double currNetworkCostOuter = [&]() {
            FlowRateMap flowRatesFB_tmp = calculateFlowRates(pressureMapFB_B);
            FlowBalanceMap balanceFB_tmp = verifyBalance(flowRatesFB_tmp);
            double sumsqTmp = 0.0;
            for (const auto& kvb : balanceFB_tmp) {
                sumsqTmp += kvb.second * kvb.second;
            }
            return 0.5 * sumsqTmp;
        }();

        bool ceresImproved = fbSummary2.final_cost < lastCostOuter * 0.995;
        bool netImproved   = currNetworkCostOuter < lastNetworkCostOuter * 0.995;
        // Ceresの方が改善している場合は継続、netのみで打ち切り判定
        if (outer >= minOuter && ceresImproved && !netImproved) {
            double improve_pct_ceres = (lastCostOuter - fbSummary2.final_cost) / std::max(1e-300, lastCostOuter) * 100.0;
            double improve_pct_net   = (lastNetworkCostOuter - currNetworkCostOuter) / std::max(1e-300, lastNetworkCostOuter) * 100.0;
            std::ostringstream osC, osN;
            osC.setf(std::ios::fixed);
            osN.setf(std::ios::fixed);
            osC << std::setprecision(3) << improve_pct_ceres;
            osN << std::setprecision(3) << improve_pct_net;
            fallbackLog(0, outerTag + " net改善なし打ち切り (ceres=" + osC.str() +
                              "%, net=" + osN.str() + "%, 閾値=0.5%)");
            break;
        }
        // Ceresも改善していない場合も打ち切り
        if (outer >= minOuter && !ceresImproved) {
            double improve_pct_ceres = (lastCostOuter - fbSummary2.final_cost) / std::max(1e-300, lastCostOuter) * 100.0;
            double improve_pct_net   = (lastNetworkCostOuter - currNetworkCostOuter) / std::max(1e-300, lastNetworkCostOuter) * 100.0;
            std::ostringstream osC, osN;
            osC.setf(std::ios::fixed);
            osN.setf(std::ios::fixed);
            osC << std::setprecision(3) << improve_pct_ceres;
            osN << std::setprecision(3) << improve_pct_net;
            fallbackLog(0, outerTag + " ceres改善なし打ち切り (ceres=" + osC.str() +
                              "%, net=" + osN.str() + "%, 閾値=0.5%)");
            break;
        }

        lastCostOuter = fbSummary2.final_cost;
        lastNetworkCostOuter = currNetworkCostOuter;
        fallbackLog(0, outerTag + " 継続: 次反復へ引継ぎ");
    }

    if (finalHaveSolution) {
        restoreFixedFlowEdges(g, changedEdgeIds, interfaceOriginalTypeById);
        network_.setLastPressureConverged(true);
        return SolverResult{finalPressureMapFB, finalFlowRatesFB, finalBalanceFB};
    }

    restoreFixedFlowEdges(g, changedEdgeIds, interfaceOriginalTypeById);
    network_.setLastPressureConverged(false);
    fallbackLog(0, "[Fallback] 全ての外部反復で収束に至りませんでした");
    return std::nullopt;
}


