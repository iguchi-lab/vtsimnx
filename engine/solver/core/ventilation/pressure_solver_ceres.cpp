#include "core/ventilation/pressure_solver_impl.h"
#include "core/ventilation/pressure_constraints.h"
#include "core/ventilation/pressure_solver_trial_spec.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

std::string sanitizeLogLabel(const std::string& logMessage) {
    if (logMessage.empty()) return "ソルバー試行";
    size_t start = logMessage.find_first_not_of("- \t");
    std::string label = (start == std::string::npos) ? logMessage : logMessage.substr(start);
    size_t dots = label.find("...");
    if (dots != std::string::npos) {
        label = label.substr(0, dots);
    }
    while (!label.empty() && std::isspace(static_cast<unsigned char>(label.back()))) {
        label.pop_back();
    }
    if (label.empty()) {
        return "ソルバー試行";
    }
    return label;
}

std::string formatSeconds(double seconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << seconds;
    return oss.str();
}

} // namespace

void PressureSolver::Impl::logCeresTiming(const std::string& label,
                                    const ceres::Solver::Summary& summary,
                                    std::function<void(const std::string&)> logger) {
    std::string sanitized = sanitizeLogLabel(label);
    std::ostringstream oss;
    oss << "--------" << sanitized << " 所要時間: " << formatSeconds(summary.total_time_in_seconds) << "秒"
        << " (前処理 " << formatSeconds(summary.preprocessor_time_in_seconds) << "秒"
        << ", 残差評価 " << formatSeconds(summary.residual_evaluation_time_in_seconds) << "秒"
        << ", ヤコビアン評価 " << formatSeconds(summary.jacobian_evaluation_time_in_seconds) << "秒"
        << ", 線形ソルバー " << formatSeconds(summary.linear_solver_time_in_seconds) << "秒"
        << ", 最適化 " << formatSeconds(summary.minimizer_time_in_seconds) << "秒)";
    if (logger) {
        logger(oss.str());
    } else {
        writeLog(logFile_, oss.str());
    }
}

// =============================================================================
// Ceresソルバー実行ユーティリティ
// =============================================================================

PressureSolver::Impl::TrialResult PressureSolver::Impl::runSolverTrial(
    const std::string& startLog,
    const std::string& successLog,
    ceres::Problem& problem,
    ceres::Solver::Summary& summary,
    double successTolerance,
    const std::function<void(ceres::Solver::Options&)>& configureOptions,
    std::function<void(const std::string&)> logger) {
    auto log = [&](const std::string& msg) {
        if (msg.empty()) return;
        if (logger) logger(msg);
        else writeLog(logFile_, msg);
    };

    log(startLog);
    ceres::Solver::Options options;
    configureOptions(options);
    // function_tolerance は Ceres の相対停止条件。最終物理合否とは分離する。
    double usedTolerance = (options.function_tolerance > 0.0) ? options.function_tolerance : successTolerance;
    ceres::Solve(options, &problem, &summary);
    logCeresTiming(startLog.empty() ? successLog : startLog, summary, logger);
    // trial「成功」= Ceres が CONVERGENCE を返し cost が有限。final_cost と ventTol は比較しない。
    bool converged = (summary.termination_type == ceres::CONVERGENCE) &&
                     std::isfinite(summary.final_cost);
    if (converged) {
        log(successLog);
    } else if (!converged) {
        // 収束しなかった場合の詳細情報を出力
        std::string terminationType;
        switch(summary.termination_type) {
            case ceres::CONVERGENCE:
                terminationType = "CONVERGENCE";
                break;
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
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "-----未収束: 終了理由=" << terminationType
            << ", Ceres cost=" << summary.final_cost
            << ", function_tol=" << usedTolerance
            << ", 反復回数=" << summary.num_successful_steps;
        log(oss.str());
    }
    TrialResult result;
    result.converged = converged;
    result.usedTolerance = usedTolerance;
    return result;
}

PressureSolver::Impl::TrialResult PressureSolver::Impl::runTwoStageRelaxation(
    const SimulationConstants& constants,
    ceres::Problem& problem,
    ceres::Solver::Summary& summary,
    const std::string& labelStage1,
    const std::string& labelStage2,
    const std::function<void(const ceres::Solver::Summary&)>& afterStage1,
    std::function<void(const std::string&)> logger) {
    // 段階的緩和法:
    // - 段階1: 緩い許容誤差で前進
    // - 段階2: 厳しめ設定で収束判定（この許容誤差で成功判定）

    ceres::Solver::Options options1;
    options1.trust_region_strategy_type = ceres::DOGLEG;
    options1.linear_solver_type = ceres::DENSE_QR;
    options1.max_num_iterations = 200;
    ventilation::applyCeresStopTolerances(options1, constants, /*functionScale=*/100.0,
                                          /*parameterScale=*/100.0, /*gradientScale=*/10.0);
    options1.jacobi_scaling = true;
    options1.minimizer_progress_to_stdout = false;

    ceres::Solve(options1, &problem, &summary);
    logCeresTiming(labelStage1, summary, logger);
    if (afterStage1) {
        afterStage1(summary);
    }

    ceres::Solver::Options options2;
    options2.trust_region_strategy_type = ceres::DOGLEG;
    options2.linear_solver_type = ceres::DENSE_QR;
    options2.max_num_iterations = 1000;
    ventilation::applyCeresStopTolerances(options2, constants);
    options2.jacobi_scaling = true;
    options2.use_inner_iterations = true;
    options2.minimizer_progress_to_stdout = false;

    ceres::Solve(options2, &problem, &summary);
    logCeresTiming(labelStage2, summary, logger);

    TrialResult result;
    result.usedTolerance = options2.function_tolerance;
    // 最終物理合否は solvePressures / fallback 側。ここは Ceres CONVERGENCE のみ。
    result.converged = (summary.termination_type == ceres::CONVERGENCE) &&
                       std::isfinite(summary.final_cost);
    return result;
}

PressureSolver::Impl::TrialResult PressureSolver::Impl::runUltraPreciseTrial(
    const SimulationConstants& constants,
    ceres::Problem& problem,
    ceres::Solver::Summary& summary,
    const std::string& labelTiming,
    double referenceCost,
    const std::function<void(double)>& onTolerance,
    std::function<void(const std::string&)> logger) {
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 5000;

    // Ceres 相対停止のみを緩和。最終物理合否（ventilationTolerance）とは分離する。
    // referenceCost は目的関数値（≈ ½‖r‖²）なので、残差スケールは √(2·cost)。
    const auto tols = ventilation::makePressureSolverTolerances(constants);
    const double residualScale = std::sqrt(std::max(0.0, 2.0 * referenceCost));
    const double tolFactor = std::clamp(
        residualScale / std::max(tols.massBalanceMaxAbs, 1e-30),
        1.0,
        1e3);
    ventilation::applyCeresStopTolerances(options, constants, tolFactor, tolFactor, tolFactor * 10);
    options.jacobi_scaling = true;
    options.use_inner_iterations = true;
    options.inner_iteration_tolerance = 1e-12;
    options.max_trust_region_radius = 1e2;
    options.initial_trust_region_radius = 1e0;
    options.min_trust_region_radius = 1e-8;
    options.minimizer_progress_to_stdout = false;

    if (onTolerance) {
        onTolerance(options.function_tolerance);
    }

    ceres::Solve(options, &problem, &summary);
    logCeresTiming(labelTiming, summary, logger);

    TrialResult result;
    result.usedTolerance = options.function_tolerance;
    result.converged = (summary.termination_type == ceres::CONVERGENCE) &&
                       std::isfinite(summary.final_cost);
    return result;
}

// =============================================================================
// プライマリソルバー（初回圧力計算）
// =============================================================================

void PressureSolver::Impl::runPrimarySolvers(const SimulationConstants& constants,
                                       ceres::Problem& problem,
                                       ceres::Solver::Summary& summary,
                                       SolverSetup& setup,
                                       double massBalanceMaxAbs,
                                       bool& physicalAccepted) {
    physicalAccepted = false;
    auto checkPhysical = [&]() -> bool {
        if (!std::isfinite(summary.final_cost)) return false;
        PressureMap pressureMap = extractPressures(setup.pressures, setup.nodeNames);
        auto eval = evaluatePressureSolution(pressureMap, massBalanceMaxAbs);
        if (eval.flowOk && eval.accepted) {
            physicalAccepted = true;
            writeDomainLog(logFile_, "圧力", "物理収支合格のため試行を終了します (mass_maxAbs=" +
                                   std::to_string(eval.solvedNodeMetrics.maxAbs) + ")");
            return true;
        }
        if (eval.flowOk) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6);
            oss << "物理収支未達: mass_maxAbs=" << eval.solvedNodeMetrics.maxAbs
                << " > tol=" << massBalanceMaxAbs
                << " → 次の手法へ";
            writeDomainLog(logFile_, "圧力", oss.str());
        } else {
            writeDomainLog(logFile_, "圧力", "物理収支評価失敗: " + eval.detail + " → 次の手法へ");
        }
        return false;
    };

    for (const auto& t : ventilation::primaryTrustRegionTrials()) {
        if (physicalAccepted) break;
        (void)runSolverTrial(
                        t.startLog,
                        t.successLog,
                        problem,
                        summary,
                        constants.ventilationTolerance,
                        [&](ceres::Solver::Options& o) { t.configure(o, constants); });
        // Ceres 終了理由は診断のみ。物理合格で試行終了（NO_CONVERGENCE でも可）。
        if (checkPhysical()) break;
    }

    if (!physicalAccepted) {
        writeDomainLog(logFile_, "圧力", "⑤段階的緩和法でソルバーを再実行します...");
        TrialResult r = runTwoStageRelaxation(
            constants,
            problem,
            summary,
            "[圧力] ⑤段階的緩和法(段階1)",
            "[圧力] ⑤段階的緩和法(段階2)",
            [&](const ceres::Solver::Summary& s1) {
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(6);
                oss << "[圧力] 段階1完了: 残差=" << s1.final_cost
                    << ", 終了理由=";
                switch(s1.termination_type) {
                    case ceres::CONVERGENCE: oss << "CONVERGENCE"; break;
                    case ceres::NO_CONVERGENCE: oss << "NO_CONVERGENCE"; break;
                    case ceres::FAILURE: oss << "FAILURE"; break;
                    case ceres::USER_FAILURE: oss << "USER_FAILURE"; break;
                    default: oss << "UNKNOWN(" << static_cast<int>(s1.termination_type) << ")"; break;
                }
                oss << ", 反復回数=" << s1.num_successful_steps;
                writeDomainLog(logFile_, "圧力", oss.str());
            });
        (void)r;
        if (checkPhysical()) {
            writeDomainLog(logFile_, "圧力", "段階的緩和法で物理収支が合格しました");
        } else {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6);
            oss << "段階2後も物理未達: 終了理由=";
            switch(summary.termination_type) {
                case ceres::CONVERGENCE: oss << "CONVERGENCE"; break;
                case ceres::NO_CONVERGENCE: oss << "NO_CONVERGENCE (最大反復回数到達)"; break;
                case ceres::FAILURE: oss << "FAILURE (計算失敗)"; break;
                case ceres::USER_FAILURE: oss << "USER_FAILURE (ユーザー関数エラー)"; break;
                default: oss << "UNKNOWN(" << static_cast<int>(summary.termination_type) << ")"; break;
            }
            oss << ", 最終残差=" << summary.final_cost
                << ", 許容誤差=" << r.usedTolerance
                << ", 反復回数=" << summary.num_successful_steps;
            writeDomainLog(logFile_, "圧力", oss.str());
        }
    }

    if (!physicalAccepted) {
        (void)runSolverTrial(
            "[圧力] ⑥Line Search方式でソルバーを再実行します...",
            "[圧力] Line Search方式: Ceres相対停止 (CONVERGENCE)。物理収支は別判定",
            problem,
            summary,
            constants.ventilationTolerance,
            [&](ceres::Solver::Options& options) {
                ventilation::configureLineSearchLbfgs(options, constants);
            });
        if (checkPhysical()) {
            writeDomainLog(logFile_, "圧力", "Line Search方式で物理収支が合格しました");
        }
    }

    if (!physicalAccepted) {
        writeDomainLog(logFile_, "圧力", "⑦超精密設定で最終試行します...");
        const double refCost = summary.final_cost;
        TrialResult r = runUltraPreciseTrial(
            constants,
            problem,
            summary,
            "[圧力] ⑦超精密設定",
            refCost,
            [&](double usedTol) {
                writeDomainLog(logFile_, "圧力", "調整済み許容誤差: " + std::to_string(usedTol));
            });
        if (checkPhysical()) {
            writeDomainLog(logFile_, "圧力", "超精密設定で物理収支が合格しました");
        } else {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6);
            oss << "超精密設定後も物理未達: 終了理由=";
            switch(summary.termination_type) {
                case ceres::CONVERGENCE: oss << "CONVERGENCE"; break;
                case ceres::NO_CONVERGENCE: oss << "NO_CONVERGENCE (最大反復回数到達)"; break;
                case ceres::FAILURE: oss << "FAILURE (計算失敗)"; break;
                case ceres::USER_FAILURE: oss << "USER_FAILURE (ユーザー関数エラー)"; break;
                default: oss << "UNKNOWN(" << static_cast<int>(summary.termination_type) << ")"; break;
            }
            oss << ", 最終残差=" << summary.final_cost
                << ", 許容誤差=" << r.usedTolerance
                << ", 反復回数=" << summary.num_successful_steps;
            writeDomainLog(logFile_, "圧力", oss.str());
        }
    }

    if (!physicalAccepted) {
        writeDomainLog(logFile_, "圧力", "全てのソルバー手法で物理収支合格に至りませんでした"
                           "（Ceres相対停止と物理合否は別判定）");
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "最終 Ceres cost=" << summary.final_cost
            << " | mass_tol(物理)=" << massBalanceMaxAbs
            << " | 次へ: フォールバック";
        writeDomainLog(logFile_, "圧力", oss.str());
    }
}

// =============================================================================
// Stage A: スーパーノード代表圧フェーズ
// =============================================================================

PressureSolver::Impl::StageAMapping PressureSolver::Impl::buildStageAMapping(
    const Graph& graph,
    const std::vector<Vertex>& vertices,
    const std::vector<int>& groupOfVertex) {
    StageAMapping mapping;
    for (size_t i = 0; i < vertices.size(); ++i) {
        Vertex v = vertices[i];
        const auto& node = graph[v];
        if (!node.calc_p) {
            continue;
        }
        int gid = groupOfVertex[i];
        if (gid >= 0) {
            if (!mapping.groupToParamIndex.count(gid)) {
                size_t idx = mapping.groupToParamIndex.size();
                mapping.groupToParamIndex[gid] = idx;
            }
            mapping.vertexToParamIndex[v] = mapping.groupToParamIndex[gid];
        } else {
            size_t idx = mapping.groupToParamIndex.size() + 1000000 + i;
            mapping.vertexToParamIndex[v] = idx;
        }
        mapping.nodeNames.push_back(node.key);
    }

    std::map<size_t, size_t> remap;
    size_t next = mapping.groupToParamIndex.size();
    for (auto& kv : mapping.vertexToParamIndex) {
        size_t oldIdx = kv.second;
        if (oldIdx >= 1000000) {
            auto it = remap.find(oldIdx);
            if (it == remap.end()) {
                size_t newIdx = next++;
                remap[oldIdx] = newIdx;
                kv.second = newIdx;
            } else {
                kv.second = it->second;
            }
        }
    }
    mapping.parameterCount = next;
    // Vertex -> param index (vecS前提)
    mapping.vertexToParamIndexVec.assign(static_cast<size_t>(boost::num_vertices(graph)), -1);
    for (const auto& kv : mapping.vertexToParamIndex) {
        mapping.vertexToParamIndexVec[static_cast<size_t>(kv.first)] = static_cast<int>(kv.second);
    }
    return mapping;
}

std::vector<double> PressureSolver::Impl::initializeStageAPressures(
    const Graph& graph,
    const StageAMapping& mapping,
    const PressureMap& prevPressureMapFB) {
    std::vector<double> pressures(mapping.parameterCount, 0.0);
    for (const auto& kv : mapping.vertexToParamIndex) {
        Vertex v = kv.first;
        size_t idx = kv.second;
        const auto& node = graph[v];
        double p0 = prevPressureMapFB.count(node.key)
                        ? prevPressureMapFB.at(node.key)
                        : node.current_p;
        if (idx < pressures.size()) {
            pressures[idx] = p0;
        }
    }
    return pressures;
}

void PressureSolver::Impl::setupStageAProblem(
    ceres::Problem& problemFB,
    const StageAMapping& mapping,
    Graph& graph,
    const std::vector<Vertex>& vertices,
    const std::vector<int>& groupOfVertex,
    const PressureMap& prevPressureMapFB,
    std::vector<double>& pressuresFB,
    int superCountA,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex) {
    size_t parameterCount = pressuresFB.size();
    double* parameterData = pressuresFB.data();
    const auto* density = network_.densityCache();

    auto addNodeResidual = [&](Vertex nodeVertex) {
        ceres::CostFunction* costFunction = PressureConstraints::createFlowBalanceConstraint(
            nodeVertex,
            graph,
            mapping.vertexToParamIndexVec,
            incidentEdgesByVertex,
            density,
            parameterCount,
            logFile_
        );
        problemFB.AddResidualBlock(costFunction, nullptr, parameterData);
    };

    if (superCountA > 0) {
        std::vector<std::vector<Vertex>> groupVertices(superCountA);
        std::vector<Vertex> nonGroupVertices;
        for (size_t i = 0; i < vertices.size(); ++i) {
            const auto& node = graph[vertices[i]];
            if (!node.calc_p) continue;
            int gid = groupOfVertex[i];
            if (gid >= 0) {
                groupVertices[gid].push_back(vertices[i]);
            } else {
                nonGroupVertices.push_back(vertices[i]);
            }
        }

        for (const auto& gv : groupVertices) {
            if (gv.empty()) continue;
            ceres::CostFunction* costG = PressureConstraints::createGroupFlowBalanceConstraint(
                gv,
                graph,
                mapping.vertexToParamIndexVec,
                incidentEdgesByVertex,
                density,
                parameterCount,
                logFile_
            );
            problemFB.AddResidualBlock(costG, nullptr, parameterData);
        }

        for (auto v : nonGroupVertices) {
            const auto& node = graph[v];
            if (!node.calc_p) continue;
            addNodeResidual(v);
        }

        // 前回平均圧への弱い正則化（ゲージ固定とは別。流量収支を支配しない重み）。
        std::vector<double> groupMean(superCountA, 0.0);
        std::vector<int> groupCount(superCountA, 0);
        for (int gid = 0; gid < superCountA; ++gid) {
            for (auto v : groupVertices[gid]) {
                const auto& node = graph[v];
                auto it = prevPressureMapFB.find(node.key);
                if (it != prevPressureMapFB.end()) {
                    groupMean[gid] += it->second;
                    groupCount[gid]++;
                }
            }
        }
        for (const auto& kv : mapping.groupToParamIndex) {
            int gid = kv.first;
            size_t idx = kv.second;
            if (!(gid >= 0 && gid < superCountA && groupCount[gid] > 0)) {
                continue;
            }
            const double target = groupMean[gid] / static_cast<double>(groupCount[gid]);
            problemFB.AddResidualBlock(
                PressureConstraints::createSoftAnchorConstraint(idx, target, /*weight=*/1e-9, parameterCount),
                nullptr,
                parameterData);
        }
    } else {
        const auto& keyToVertex = network_.getKeyToVertex();
        for (const auto& nodeName : mapping.nodeNames) {
            auto it = keyToVertex.find(nodeName);
            if (it == keyToVertex.end()) {
                continue;
            }
            addNodeResidual(it->second);
        }
    }

    // 固定圧境界のない連結成分ごとにゲージを1つだけ固定（全グループへの強固定はしない）。
    addPressureGaugeAnchors(graph,
                            mapping.vertexToParamIndexVec,
                            pressuresFB,
                            &incidentEdgesByVertex,
                            problemFB,
                            /*anchorWeight=*/1.0);
}

// =============================================================================
// Stage B: フルノード再解フェーズ
// =============================================================================

PressureSolver::Impl::StageBSetup PressureSolver::Impl::buildStageBSetup(
    const Graph& graph,
    const PressureMap& stageAPressureMap) {
    StageBSetup setup;
    size_t nextIndex = 0;
    auto vr = boost::vertices(graph);
    for (auto v : boost::make_iterator_range(vr)) {
        const auto& node = graph[v];
        if (!node.calc_p) {
            continue;
        }
        setup.vertexToParamIndex[v] = nextIndex++;
        setup.nodeNames.push_back(node.key);
    }
    setup.pressures.resize(nextIndex, 0.0);
    for (const auto& kv : setup.vertexToParamIndex) {
        Vertex v = kv.first;
        size_t idx = kv.second;
        const auto& node = graph[v];
        double p0 = stageAPressureMap.count(node.key)
                        ? stageAPressureMap.at(node.key)
                        : node.current_p;
        setup.pressures[idx] = p0;
    }
    setup.vertexToParamIndexVec.assign(static_cast<size_t>(boost::num_vertices(graph)), -1);
    for (const auto& kv : setup.vertexToParamIndex) {
        setup.vertexToParamIndexVec[static_cast<size_t>(kv.first)] = static_cast<int>(kv.second);
    }
    return setup;
}

bool PressureSolver::Impl::runStageBTrials(const SimulationConstants& constants,
                                     ceres::Problem& problemFB2,
                                     ceres::Solver::Summary& fbSummary2,
                                     StageBSetup& setup,
                                     double massBalanceMaxAbs,
                                     const std::function<void(int, const std::string&)>& fallbackLog) {
    bool physicalAccepted = false;
    auto log2 = [&](const std::string& msg) { fallbackLog(2, msg); };

    auto checkPhysical = [&]() -> bool {
        if (!std::isfinite(fbSummary2.final_cost)) return false;
        PressureMap pressureMap = extractPressures(setup.pressures, setup.nodeNames);
        auto eval = evaluatePressureSolution(pressureMap, massBalanceMaxAbs);
        if (eval.flowOk && eval.accepted) {
            physicalAccepted = true;
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << eval.solvedNodeMetrics.maxAbs;
            fallbackLog(2, "[B] 仮ネットワーク物理収支合格 | mass_maxAbs=" + os.str());
            return true;
        }
        return false;
    };

    auto tryTrial = [&](const std::string& startMsg,
                        const std::string& successPrefix,
                        const std::function<void(ceres::Solver::Options&)>& configure) {
        if (physicalAccepted) return;
        TrialResult r = runSolverTrial(startMsg,
                                       /*successLog=*/"",
                                       problemFB2,
                                       fbSummary2,
                                       constants.ventilationTolerance,
                                       configure,
                                       log2);
        // Ceres CONVERGENCE は診断。物理合格で打ち切る。
        if (r.converged) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(2, successPrefix + os.str() + " | tol=" + std::to_string(r.usedTolerance));
        }
        (void)checkPhysical();
    };

    for (const auto& t : ventilation::stageBTrustRegionTrials()) {
        if (physicalAccepted) break;
        tryTrial(
            t.startLog,
            t.successLog,
            [&](ceres::Solver::Options& o) { t.configure(o, constants); });
    }

    if (!physicalAccepted) {
        fallbackLog(2, "[B-⑤] 段階的緩和法でソルバーを再実行します");
        TrialResult r = runTwoStageRelaxation(
            constants,
            problemFB2,
            fbSummary2,
            "[B-⑤] 段階1",
            "[B-⑤] 段階2",
            [&](const ceres::Solver::Summary& s1) {
                std::ostringstream os;
                os << std::scientific << std::setprecision(6) << s1.final_cost;
                fallbackLog(3, "[B-⑤] 段階1完了 | residual=" + os.str());
            },
            log2);
        (void)r;
        if (checkPhysical()) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(2, "[B-⑤] 物理合格 | residual=" + os.str() + " | tol=" +
                               std::to_string(r.usedTolerance));
        }
    }

    tryTrial("[B-⑥] Line Search方式でソルバーを再実行します",
             "[B-⑥] Ceres CONVERGENCE | residual=",
             [&](ceres::Solver::Options& o) {
                 ventilation::configureLineSearchLbfgs(o, constants);
             });

    if (!physicalAccepted) {
        fallbackLog(2, "[B-⑦] 超精密設定で最終試行します");
        const double refCost = fbSummary2.final_cost;
        TrialResult r = runUltraPreciseTrial(
            constants,
            problemFB2,
            fbSummary2,
            "[B-⑦] 超精密設定",
            refCost,
            [&](double usedTol) {
                fallbackLog(3, "[B-⑦] 調整済み許容誤差=" + std::to_string(usedTol));
            },
            log2);
        (void)r;
        if (checkPhysical()) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(6) << fbSummary2.final_cost;
            fallbackLog(2, "[B-⑦] 物理合格 | residual=" + os.str() + " | tol=" +
                               std::to_string(r.usedTolerance));
        }
    }

    // ok: 仮ネットワーク上で質量収支合格（復元後の iface は decision 側）
    return physicalAccepted;
}

