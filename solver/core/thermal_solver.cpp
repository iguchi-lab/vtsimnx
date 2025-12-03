#include "core/thermal_solver.h"
#include "network/thermal_network.h"
#include "utils/utils.h"
#include "core/physical_constants.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <unordered_map>

// 物理定数は core/physical_constants.h から使用

// =============================================================================
// HeatCalculation名前空間 - 熱計算の共通テンプレート関数群
// =============================================================================

namespace HeatCalculation {

template <typename T>
T calcAdvectionHeat(const T& sourceTemp, const T& targetTemp, const EdgeProperties& edgeData) {
    double flowRate = edgeData.flow_rate; // m3/s
    if (std::abs(flowRate) < PhysicalConstants::FLOW_RATE_MIN) return T(0.0);
    
    double mDotCp = PhysicalConstants::RHO_AIR * PhysicalConstants::C_AIR * std::abs(flowRate);
    // 流出側の温度を使用：正の流れなら source→target なので sourceTemp
    T outTemp = (flowRate > 0.0) ? sourceTemp : targetTemp;
    T direction = T(flowRate > 0.0 ? 1.0 : -1.0);
    
    return direction * T(mDotCp) * outTemp;
}

template <typename T>
T calcConductionHeat(const T& sourceTemp, const T& targetTemp, const EdgeProperties& edgeData) {
    return T(edgeData.conductance) * (sourceTemp - targetTemp);
}

template <typename T>
T calcGenerationHeat(const EdgeProperties& edgeData) {
    return T(edgeData.current_heat_generation);
}

template <typename T>
T calculateUnifiedHeat(const T& sourceTemp, const T& targetTemp, const EdgeProperties& edgeData) {
    if (edgeData.type == "advection") {
        return calcAdvectionHeat(sourceTemp, targetTemp, edgeData);
    } else if (edgeData.type == "conductance") {
        return calcConductionHeat(sourceTemp, targetTemp, edgeData);
    } else if (edgeData.type == "heat_generation") {
        return calcGenerationHeat<T>(edgeData);
    } else {
        return T(0.0);
    }
}

} // namespace HeatCalculation

// =============================================================================
// HeatBalanceConstraintクラス - Ceres自動微分用の制約条件
// =============================================================================

template <typename T>
T HeatBalanceConstraint::getNodeTemperature(Vertex v, T const* const* parameters) const {
    auto it = vertexToParameterIndex_.find(v);
    if (it != vertexToParameterIndex_.end()) {
        return parameters[0][it->second];
    } else {
        return T(graph_[v].current_t);
    }
}

template <typename T>
bool HeatBalanceConstraint::operator()(T const* const* parameters, T* residual) const {
    auto nodeIt = nodeKeyToVertex_.find(nodeName_);
    if (nodeIt == nodeKeyToVertex_.end()) {
        residual[0] = T(0.0);
        return true;
    }
    Vertex nodeVertex = nodeIt->second;
    [[maybe_unused]] T nodeTemp = getNodeTemperature(nodeVertex, parameters);

    T heatIn = T(0.0);
    T heatOut = T(0.0);

    // 統一された関数を使用してエッジごとの熱流量を計算
    auto edge_range = boost::edges(graph_);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        Vertex sourceVertex = boost::source(edge, graph_);
        Vertex targetVertex = boost::target(edge, graph_);
        const auto& edgeData = graph_[edge];

        T sourceTemp = getNodeTemperature(sourceVertex, parameters);
        T targetTemp = getNodeTemperature(targetVertex, parameters);
        
        // 統一された熱計算関数を使用
        T heatRate = HeatCalculation::calculateUnifiedHeat(sourceTemp, targetTemp, edgeData);
        
        // このノードに関連するエッジの場合、流入/流出を計算
        if (sourceVertex == nodeVertex) {
            // このノードからの流出
            if (heatRate > T(0.0))
                heatOut += heatRate;
            else
                heatIn += -heatRate;
        } else if (targetVertex == nodeVertex) {
            // このノードへの流入
            if (heatRate > T(0.0))
                heatIn += heatRate;
            else
                heatOut += -heatRate;
        }
    }
    // エアコンによるバランス調整
    const auto& currentNodeData = graph_[nodeVertex];
    
    // このノードがエアコンノードの場合、set_nodeのバランス不足分を引き受ける
    if (currentNodeData.type == "aircon" && currentNodeData.on && !currentNodeData.set_node.empty()) {
        auto setNodeIt = nodeKeyToVertex_.find(currentNodeData.set_node);
        if (setNodeIt != nodeKeyToVertex_.end()) {
            Vertex setNodeVertex = setNodeIt->second;
            
            // set_nodeの熱バランスを統一された関数で計算
            T setNodeHeatIn = T(0.0);
            T setNodeHeatOut = T(0.0);
            
            auto edge_range_local = boost::edges(graph_);
            for (auto edge_local : boost::make_iterator_range(edge_range_local)) {
                Vertex sv = boost::source(edge_local, graph_);
                Vertex dv = boost::target(edge_local, graph_);
                const auto& eprop = graph_[edge_local];

                T srcTemp = getNodeTemperature(sv, parameters);
                T dstTemp = getNodeTemperature(dv, parameters);
                T heatRate = HeatCalculation::calculateUnifiedHeat(srcTemp, dstTemp, eprop);
                
                if (sv == setNodeVertex) {
                    // set_nodeからの流出
                    if (heatRate > T(0.0))
                        setNodeHeatOut += heatRate;
                    else
                        setNodeHeatIn += -heatRate;
                } else if (dv == setNodeVertex) {
                    // set_nodeへの流入
                    if (heatRate > T(0.0))
                        setNodeHeatIn += heatRate;
                    else
                        setNodeHeatOut += -heatRate;
                }
            }
            
            // エアコンがset_nodeのバランス不足分を引き受ける
            T setNodeBalance = setNodeHeatIn - setNodeHeatOut;
            heatIn += -setNodeBalance; // 不足分を補う
        }
    }
    
    // このノードがset_nodeでエアコンがオンの場合、バランスをゼロにする
    auto vertex_range_ac = boost::vertices(graph_);
    for (auto vertex_ac : boost::make_iterator_range(vertex_range_ac)) {
        const auto& nodeData_ac = graph_[vertex_ac];
        if (nodeData_ac.type == "aircon" && nodeData_ac.on && 
            nodeData_ac.set_node == nodeName_) {
            // このノードはエアコンによって温度が制御されるため、バランスは0
            residual[0] = T(0.0);
            return true;
        }
    }
    residual[0] = heatIn - heatOut;
    return true;
}

// =============================================================================
// ThermalSolverクラス - 温度・熱流量計算のメインソルバー  
// =============================================================================

ThermalSolver::ThermalSolver(ThermalNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

void ThermalSolver::setInitialTemperatures(std::vector<double>& temperatures, const std::vector<std::string>& nodeNames) {
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        temperatures[i] = network_.getNode(nodeNames[i]).current_t;
    }
    
    // サイズチェック
    if (temperatures.size() != nodeNames.size()) {
        writeLog(logFile_, "  警告: 温度配列とノード名配列のサイズが一致しません");
    }
}

TemperatureMap ThermalSolver::extractTemperatures(const std::vector<double>& temperatures,
                                                const std::vector<std::string>& nodeNames) {
    TemperatureMap tempMap;
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        tempMap[nodeNames[i]] = temperatures[i];
    }
    const auto& graph = network_.getGraph();
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& properties = graph[vertex];
        if (!properties.calc_t) {
            tempMap[properties.key] = properties.current_t;
        }
    }
    return tempMap;
}

HeatRateMap ThermalSolver::calculateHeatRates(const TemperatureMap& tempMap) {
    HeatRateMap heatRates;
    
    // 各エッジ（熱ブランチ）の熱流量を計算
    auto edge_range = boost::edges(network_.getGraph());
    for (auto edge : boost::make_iterator_range(edge_range)) {
        Vertex sourceVertex = boost::source(edge, network_.getGraph());
        Vertex targetVertex = boost::target(edge, network_.getGraph());
        const auto& edgeData = network_.getGraph()[edge];
        const auto& sourceNode = network_.getGraph()[sourceVertex];
        const auto& targetNode = network_.getGraph()[targetVertex];
        
        std::string sourceName = sourceNode.key;
        std::string targetName = targetNode.key;
        
        // 温度値の存在チェック
        auto sourceTempIt = tempMap.find(sourceName);
        auto targetTempIt = tempMap.find(targetName);
        
        if (sourceTempIt == tempMap.end() || targetTempIt == tempMap.end()) {
            writeLog(logFile_, "  警告: ノード温度が見つかりません - " + sourceName + " → " + targetName);
            continue;
        }
        
        double sourceTempValue = sourceTempIt->second;
        double targetTempValue = targetTempIt->second;
        
        std::pair<std::string, std::string> edgeKey = {sourceNode.key, targetNode.key};
        
        // 統一された熱計算関数を使用
        double heatRate = HeatCalculation::calculateUnifiedHeat(sourceTempValue, targetTempValue, edgeData);
        
        // 異常値チェック
        if (!std::isfinite(heatRate)) {
            writeLog(logFile_, "  警告: 無限大または非数の熱流量値が検出されました - " + sourceName + " → " + targetName);
            heatRate = 0.0;
        }
        
        heatRates[edgeKey] = heatRate;
    }
    
    return heatRates;
}

HeatBalanceMap ThermalSolver::verifyBalance(const HeatRateMap& heatRates) {
    HeatBalanceMap balance;

    const auto& graph = network_.getGraph();

    // ノード初期化（0.0）
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        balance[graph[vertex].key] = 0.0;
    }

    // すでに計算済みの heatRates から入出熱を構築
    for (const auto& kv : heatRates) {
        const auto& edgeKey = kv.first; // {source, target}
        const std::string& srcName = edgeKey.first;
        const std::string& dstName = edgeKey.second;
        double q = kv.second; // src -> dst を正

        // 符号に関係なく一貫した更新
        balance[srcName] -= q; // 出た分だけ減算
        balance[dstName] += q; // 入った分だけ加算
    }
    // --- エアコンによるバランス付け替え ---
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& nodeData = graph[vertex];
        const std::string& nodeName = nodeData.key;
        if (nodeData.type == "aircon" && nodeData.on) {
            const std::string& setNode = nodeData.set_node;
            auto itSet = balance.find(setNode);
            if (itSet != balance.end()) {
                //writeLog(logFile_, "エアコンによるバランス付け替え: " + nodeName + " -> " + setNode + " " + std::to_string(-itSet->second));
                balance[nodeName] = -itSet->second;
                itSet->second = 0.0;
            }
        }
    }
    return balance;
}

std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> ThermalSolver::solveTemperatures(
    const SimulationConstants& constants) {
    
    const auto& graph = network_.getGraph();
    std::vector<std::string> nodeNames;
    std::map<Vertex, size_t> vertexToParameterIndex;
    size_t parameterIndex = 0;
    
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& properties = graph[vertex];
        if (properties.calc_t) {
            nodeNames.push_back(properties.key);
            vertexToParameterIndex[vertex] = parameterIndex++;
        }
    }

    if (nodeNames.empty()) {
        writeLog(logFile_, "  警告: 温度計算対象のノードがありません");
        return {TemperatureMap{}, HeatRateMap{}, HeatBalanceMap{}};
    }

    std::vector<double> temperatures(nodeNames.size());
    setInitialTemperatures(temperatures, nodeNames);

    ceres::Problem problem;
    for (const std::string& nodeName : nodeNames) {
        auto constraint = new HeatBalanceConstraint(
            nodeName,
            network_.getGraph(),
            network_.getKeyToVertex(),
            vertexToParameterIndex,
            logFile_
        );
        auto costFunction = new ceres::DynamicAutoDiffCostFunction<HeatBalanceConstraint>(constraint);
        costFunction->AddParameterBlock(temperatures.size());
        costFunction->SetNumResiduals(1);
        problem.AddResidualBlock(costFunction, nullptr, temperatures.data());
    }

    // 複数のソルバー設定を順次試行
    ceres::Solver::Summary summary;
    bool converged = false;

    // 1st try: 標準設定（従来）
    {
        writeLog(logFile_, "--------①標準設定でソルバーを実行します...");
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = constants.maxInnerIteration;
        options.function_tolerance = constants.thermalTolerance;
        options.parameter_tolerance = constants.thermalTolerance;
        options.minimizer_progress_to_stdout = false;

        ceres::Solve(options, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "--------標準設定で収束しました");
        }
    }

    // 2nd try: 堅牢設定（収束しなかった場合）
    if (!converged) {
        writeLog(logFile_, "---堅牢設定でソルバーを再実行します...");
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = std::max(500, static_cast<int>(constants.maxInnerIteration * 2));

        // より厳しい収束判定
        options.function_tolerance = constants.thermalTolerance * 0.01;
        options.parameter_tolerance = constants.thermalTolerance * 0.01;
        options.gradient_tolerance = constants.thermalTolerance * 0.1;

        // 数値安定性向上
        options.jacobi_scaling = true;
        options.use_inner_iterations = true;
        options.max_trust_region_radius = 1e4;
        options.initial_trust_region_radius = 1e2;
        options.minimizer_progress_to_stdout = false;

        ceres::Solve(options, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "---堅牢設定で収束しました");
        }
    }

    // 3rd try: DENSE_SCHUR ソルバー（条件数改善）
    if (!converged) {
        writeLog(logFile_, "---DENSE_SCHUR設定でソルバーを再実行します...");
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.linear_solver_type = ceres::DENSE_SCHUR;  // より堅牢な線形ソルバー
        options.max_num_iterations = 500;

        options.function_tolerance = constants.thermalTolerance * 0.01;
        options.parameter_tolerance = constants.thermalTolerance * 0.01;
        options.gradient_tolerance = constants.thermalTolerance * 0.1;

        // 数値安定性向上
        options.jacobi_scaling = true;
        options.use_inner_iterations = true;
        options.max_trust_region_radius = 1e4;
        options.initial_trust_region_radius = 1e2;
        options.minimizer_progress_to_stdout = false;

        ceres::Solve(options, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "---DENSE_SCHUR設定で収束しました");
        }
    }

    // 4th try: SPARSE_NORMAL_CHOLESKY ソルバー（大規模問題用）
    if (!converged) {
        writeLog(logFile_, "---SPARSE_NORMAL_CHOLESKY設定でソルバーを再実行します...");
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // スパース行列用
        options.max_num_iterations = 1000;

        options.function_tolerance = constants.thermalTolerance * 0.001;
        options.parameter_tolerance = constants.thermalTolerance * 0.001;
        options.gradient_tolerance = constants.thermalTolerance * 0.01;

        // 最大限の安定性設定
        options.jacobi_scaling = true;
        options.use_inner_iterations = true;
        options.inner_iteration_tolerance = 1e-8;
        options.max_trust_region_radius = 1e3;
        options.initial_trust_region_radius = 1e1;
        options.minimizer_progress_to_stdout = false;

        ceres::Solve(options, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "---SPARSE_NORMAL_CHOLESKY設定で収束しました");
        }
    }

    // 5th try: 段階的緩和法（初期値改善）
    if (!converged) {
        writeLog(logFile_, "---段階的緩和法でソルバーを再実行します...");

        // 段階1: 緩い許容誤差で初期解を求める
        ceres::Solver::Options options1;
        options1.trust_region_strategy_type = ceres::DOGLEG;
        options1.linear_solver_type = ceres::DENSE_QR;
        options1.max_num_iterations = 200;
        options1.function_tolerance = constants.thermalTolerance * 10;   // 10倍緩い
        options1.parameter_tolerance = constants.thermalTolerance * 10;
        options1.gradient_tolerance = constants.thermalTolerance;
        options1.jacobi_scaling = true;
        options1.minimizer_progress_to_stdout = false;

        ceres::Solve(options1, &problem, &summary);
        writeLog(logFile_, "----段階1完了: 残差 " + std::to_string(summary.final_cost));

        // 段階2: より厳しい許容誤差で最終解を求める
        ceres::Solver::Options options2;
        options2.trust_region_strategy_type = ceres::DOGLEG;
        options2.linear_solver_type = ceres::DENSE_QR;
        options2.max_num_iterations = 1000;
        options2.function_tolerance = constants.thermalTolerance * 0.01;
        options2.parameter_tolerance = constants.thermalTolerance * 0.01;
        options2.gradient_tolerance = constants.thermalTolerance * 0.1;
        options2.jacobi_scaling = true;
        options2.use_inner_iterations = true;
        options2.minimizer_progress_to_stdout = false;

        ceres::Solve(options2, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "---段階的緩和法で収束しました");
        }
    }

    // 6th try: Line Search ベース（Trust Region以外のアプローチ）
    if (!converged) {
        writeLog(logFile_, "---Line Search方式でソルバーを再実行します...");
        ceres::Solver::Options options;
        options.minimizer_type = ceres::LINE_SEARCH;  // Trust Regionではなく Line Search
        options.line_search_direction_type = ceres::LBFGS;  // L-BFGS
        options.line_search_type = ceres::WOLFE;
        options.max_num_iterations = 1000;

        // より緩い許容誤差から開始
        options.function_tolerance = constants.thermalTolerance;
        options.parameter_tolerance = constants.thermalTolerance;
        options.gradient_tolerance = constants.thermalTolerance * 10;

        options.jacobi_scaling = true;
        options.minimizer_progress_to_stdout = false;

        ceres::Solve(options, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "---Line Search方式で収束しました");
        }
    }

    // 7th try: 超精密設定（最後の手段）
    if (!converged) {
        writeLog(logFile_, "---超精密設定で最終試行します...");
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 5000;  // 大幅増加

        // 段階的な許容誤差緩和
        double tolerance_factor = std::max(1.0, summary.final_cost / constants.thermalTolerance * 0.1);
        options.function_tolerance = constants.thermalTolerance * tolerance_factor;
        options.parameter_tolerance = constants.thermalTolerance * tolerance_factor;
        options.gradient_tolerance = constants.thermalTolerance * tolerance_factor * 10;

        // 最大限の数値安定性
        options.jacobi_scaling = true;
        options.use_inner_iterations = true;
        options.inner_iteration_tolerance = 1e-12;
        options.max_trust_region_radius = 1e2;
        options.initial_trust_region_radius = 1e0;
        options.min_trust_region_radius = 1e-8;
        options.minimizer_progress_to_stdout = false;

        writeLog(logFile_, "----調整済み許容誤差: " + std::to_string(options.function_tolerance));

        ceres::Solve(options, &problem, &summary);
        converged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.thermalTolerance);

        if (converged) {
            writeLog(logFile_, "---超精密設定で収束しました");
        } else {
            writeLog(logFile_, "---全ての先進的ソルバー手法で収束に失敗しました");
            writeLog(logFile_, "---最終残差: " + std::to_string(summary.final_cost) + 
                             " (目標: " + std::to_string(constants.thermalTolerance) + ")");
        }
    }
    
    // 実際の残差をチェックして真の収束判定を行う
    bool reallyConverged = (summary.termination_type == ceres::CONVERGENCE) && 
                          (summary.final_cost <= constants.thermalTolerance);
    
    if (reallyConverged) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << summary.final_cost;
        std::string line = "-----熱計算収束 | iter=" + std::to_string(summary.iterations.size()) +
                          " | residual=" + oss.str() +
                          " | tol=" + std::to_string(constants.thermalTolerance);
        writeLog(logFile_, line);
    } 
    else {
        // 詳細な失敗診断ログ
        writeLog(logFile_, "--[ERROR] 熱計算が収束しませんでした");
        
        // 終了理由の詳細
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
        writeLog(logFile_, "    終了理由: " + terminationType);
        
        // 数値的問題の診断
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << summary.final_cost;
        writeLog(logFile_, "    最終残差: " + oss.str());
        writeLog(logFile_, "    実行反復数: " + std::to_string(summary.iterations.size()));
        writeLog(logFile_, "    設定最大反復数: " + std::to_string(constants.maxInnerIteration));
        writeLog(logFile_, "    許容誤差: " + std::to_string(constants.thermalTolerance));
        writeLog(logFile_, "    残差/許容誤差比: " + std::to_string(summary.final_cost / constants.thermalTolerance));
        
        // 利用可能な追加情報
        if (summary.iterations.size() > 0) {
            writeLog(logFile_, "    ソルバー情報: 線形ソルバー=" + std::to_string(static_cast<int>(summary.linear_solver_type_used)));
            writeLog(logFile_, "    最終コスト値: " + std::to_string(summary.final_cost));
        }
        
        // 計算途中の温度状態出力（最初の5個のみ）
        writeLog(logFile_, "    現在の温度値 (最初の5個):");
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), temperatures.size()); ++i) {
            writeLog(logFile_, "      " + nodeNames[i] + ": " + std::to_string(temperatures[i]) + " ℃");
        }
        if (temperatures.size() > 5) {
            writeLog(logFile_, "      ... (残り " + std::to_string(temperatures.size() - 5) + " 個のノード)");
        }
        
        // 収束性改善の提案
        writeLog(logFile_, "    収束改善の提案:");
        if (summary.final_cost > constants.thermalTolerance * 1000) {
            writeLog(logFile_, "      - 熱計算許容誤差を緩める (1e-4 〜 1e-3 程度に設定)");
        }
        if (summary.iterations.size() >= constants.maxInnerIteration) {
            writeLog(logFile_, "      - 最大反復回数を増やす (200〜500回程度に設定)");
        }
        writeLog(logFile_, "    詳細レポート: " + summary.BriefReport());
        
        // 自動的にシステム診断情報を出力
        outputThermalSystemDiagnostics();
    }

    // 計算結果の出力
    //writeLog(logFile_, "    温度値: " + pure_to_string(temperatures));
    TemperatureMap tempMap = extractTemperatures(temperatures, nodeNames);
    //writeLog(logFile_, "    温度マップ: " + map_to_string(tempMap));
    HeatRateMap heatRates = calculateHeatRates(tempMap);
    //writeLog(logFile_, "    熱流量マップ: " + map_to_string(heatRates));
    HeatBalanceMap balance = verifyBalance(heatRates);
    //writeLog(logFile_, "    バランス検証: " + map_to_string(balance));

    return {tempMap, heatRates, balance};
}

void ThermalSolver::outputThermalSystemDiagnostics() {
    writeLog(logFile_, "  熱システム診断情報:");
    
    const auto& graph = network_.getGraph();
    
    // ノード数統計
    size_t totalNodes = boost::num_vertices(graph);
    size_t thermalCalcNodes = 0;
    size_t fixedTempNodes = 0;
    
    // 温度設定の統計
    std::vector<double> fixedTemps;
    std::vector<double> calcTemps;
    
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& properties = graph[vertex];
        if (properties.calc_t) {
            thermalCalcNodes++;
            calcTemps.push_back(properties.current_t);
        } else {
            fixedTempNodes++;
            fixedTemps.push_back(properties.current_t);
        }
    }
    
    writeLog(logFile_, "    ノード数: " + std::to_string(totalNodes));
    
    // ブランチ数統計
    size_t totalBranches = boost::num_edges(graph);
    writeLog(logFile_, "    ブランチ数: " + std::to_string(totalBranches));
    
    // 熱特性値の統計（熱伝導係数や熱容量など）
    std::vector<double> thermalConductivities;
    std::vector<double> heatCapacities;
    
    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        const auto& properties = graph[edge];
        if (properties.type == "thermal") {
            // 熱伝導係数の統計（仮想的な値として`a`を使用）
            if (properties.a > 0) {
                thermalConductivities.push_back(properties.a);
            }
        }
    }
    
    if (!thermalConductivities.empty()) {
        auto minMax = std::minmax_element(thermalConductivities.begin(), thermalConductivities.end());
        writeLog(logFile_, "    熱特性値の統計:");
        writeLog(logFile_, "      最小: " + std::to_string(*minMax.first) + " W/K");
        writeLog(logFile_, "      最大: " + std::to_string(*minMax.second) + " W/K");
        if (*minMax.first > 0) {
            writeLog(logFile_, "      比率: " + std::to_string(*minMax.second / *minMax.first));
        }
    }
    
    writeLog(logFile_, "    固定温度ノード数: " + std::to_string(fixedTempNodes));
    writeLog(logFile_, "    計算ノード数: " + std::to_string(thermalCalcNodes));
    
    // 固定温度の範囲
    if (!fixedTemps.empty()) {
        auto minMax = std::minmax_element(fixedTemps.begin(), fixedTemps.end());
        writeLog(logFile_, "    固定温度の範囲:");
        writeLog(logFile_, "      最小: " + std::to_string(*minMax.first) + " ℃");
        writeLog(logFile_, "      最大: " + std::to_string(*minMax.second) + " ℃");
        writeLog(logFile_, "      範囲: " + std::to_string(*minMax.second - *minMax.first) + " ℃");
    }
    
    // 計算温度の範囲（現在値）
    if (!calcTemps.empty()) {
        auto minMax = std::minmax_element(calcTemps.begin(), calcTemps.end());
        writeLog(logFile_, "    計算温度の現在範囲:");
        writeLog(logFile_, "      最小: " + std::to_string(*minMax.first) + " ℃");
        writeLog(logFile_, "      最大: " + std::to_string(*minMax.second) + " ℃");
        writeLog(logFile_, "      範囲: " + std::to_string(*minMax.second - *minMax.first) + " ℃");
    }
}