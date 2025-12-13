#include "core/thermal_solver.h"
#include "core/heat_calculation.h"
#include "core/thermal_solver_ceres.h"
#include "core/thermal_solver_newton_gs.h"
#include "core/thermal_constraints.h"
#include "network/thermal_network.h"
#include "utils/utils.h"
#include "../archenv/include/archenv.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <unordered_map>


// =============================================================================
// ThermalSolverクラス - 温度・熱流量計算のメインソルバー  
// =============================================================================

ThermalSolver::ThermalSolver(ThermalNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

// =============================================================================
// 初期化関数
// =============================================================================

void ThermalSolver::setInitialTemperatures(std::vector<double>& temperatures, const std::vector<std::string>& nodeNames) {
    // サイズチェック（初期化前に実行）
    if (temperatures.size() != nodeNames.size()) {
        writeLog(logFile_, "--警告: 温度配列とノード名配列のサイズが一致しません (" + 
                 std::to_string(temperatures.size()) + " vs " + std::to_string(nodeNames.size()) + ")");
        return;
    }
    
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        temperatures[i] = network_.getNode(nodeNames[i]).current_t;
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

// =============================================================================
// 熱流量・バランス計算
// =============================================================================

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
            writeLog(logFile_, "--警告: ノード温度が見つかりません - " + sourceName + " → " + targetName);
            continue;
        }
        
        double sourceTempValue = sourceTempIt->second;
        double targetTempValue = targetTempIt->second;
        
        std::pair<std::string, std::string> edgeKey = {sourceNode.key, targetNode.key};
        
        // 統一された熱計算関数を使用
        double heatRate = HeatCalculation::calculateUnifiedHeat(sourceTempValue, targetTempValue, edgeData);
        
        // 異常値チェック
        if (!std::isfinite(heatRate)) {
            writeLog(logFile_, "--警告: 無限大または非数の熱流量値が検出されました - " + sourceName + " → " + targetName);
            heatRate = 0.0;
        }
        
        heatRates[edgeKey] += heatRate;  // 同一ノードペア間の流量を合計
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

// =============================================================================
// メイン温度計算
// =============================================================================

std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> ThermalSolver::solveTemperatures(
    const SimulationConstants& constants) {
    
    const auto& graph = network_.getGraph();
    std::unordered_map<Vertex, std::vector<Edge>> incidentEdges;
    incidentEdges.reserve(boost::num_vertices(graph));
    auto incident_edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(incident_edge_range)) {
        Vertex sv = boost::source(edge, graph);
        Vertex tv = boost::target(edge, graph);
        incidentEdges[sv].push_back(edge);
        incidentEdges[tv].push_back(edge);
    }

    std::unordered_map<std::string, std::vector<Vertex>> airconBySetNode;
    auto vertex_range_aircon = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range_aircon)) {
        const auto& properties = graph[vertex];
        if (properties.type == "aircon" && !properties.set_node.empty()) {
            airconBySetNode[properties.set_node].push_back(vertex);
        }
    }

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
        writeLog(logFile_, "--警告: 温度計算対象のノードがありません");
        return {TemperatureMap{}, HeatRateMap{}, HeatBalanceMap{}};
    }

    // まず Newton-GS+SOR（線形サブ問題をGS+SORで解く）を試行
    writeLog(logFile_, "--------Newton-GS+SORソルバーを試行します...");
    try {
        // 過緩和係数（経験的に安定〜やや加速）: 1.2 をデフォルトで使用
        constexpr double kSorOmega = 1.2;
        auto gsResult = ThermalSolverNewtonGS::solveTemperaturesNewtonGS(
            network_, constants, kSorOmega, logFile_);
        
        // 結果を検証
        double maxBalance = 0.0;
        for (const auto& [nodeName, balance] : std::get<2>(gsResult)) {
            maxBalance = std::max(maxBalance, std::abs(balance));
        }

        if (maxBalance <= constants.thermalTolerance) {
            return gsResult;
        } else {
            writeLog(logFile_, "--------Newton-GS+SORは線形残差で収束したものの、熱バランス最大偏差="
                               + std::to_string(maxBalance) + " が許容値 " + std::to_string(constants.thermalTolerance) +
                               " を超過。Ceresにフォールバックします...");
        }
    } catch (const std::exception& e) {
        writeLog(logFile_, "--------Newton-GS+SORでエラー: " + std::string(e.what()) + " Ceresにフォールバックします...");
    }

    // Ceresソルバーで温度計算を実行
    writeLog(logFile_, "--------Ceresソルバーで温度計算を実行します...");
    std::vector<double> temperatures(nodeNames.size());
    setInitialTemperatures(temperatures, nodeNames);

    ceres::Problem problem;
    for (const std::string& nodeName : nodeNames) {
        ceres::CostFunction* costFunction = new HeatBalanceConstraint(
            nodeName,
            network_.getGraph(),
            network_.getKeyToVertex(),
            vertexToParameterIndex,
            incidentEdges,
            airconBySetNode,
            temperatures.size(),
            logFile_
        );
        problem.AddResidualBlock(costFunction, nullptr, temperatures.data());
    }

    ceres::Solver::Summary summary;
    ThermalSolverCeres::runThermalSolvers(constants, problem, summary, logFile_);

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
        writeLog(logFile_, "----終了理由: " + terminationType);
        
        // 数値的問題の診断
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << summary.final_cost;
        writeLog(logFile_, "----最終残差: " + oss.str());
        writeLog(logFile_, "----実行反復数: " + std::to_string(summary.iterations.size()));
        writeLog(logFile_, "----設定最大反復数: " + std::to_string(constants.maxInnerIteration));
        writeLog(logFile_, "----許容誤差: " + std::to_string(constants.thermalTolerance));
        writeLog(logFile_, "----残差/許容誤差比: " + std::to_string(summary.final_cost / constants.thermalTolerance));
        
        // 利用可能な追加情報
        if (summary.iterations.size() > 0) {
            writeLog(logFile_, "----ソルバー情報: 線形ソルバー=" + std::to_string(static_cast<int>(summary.linear_solver_type_used)));
            writeLog(logFile_, "----最終コスト値: " + std::to_string(summary.final_cost));
        }
        
        // 計算途中の温度状態出力（最初の5個のみ）
        writeLog(logFile_, "----現在の温度値 (最初の5個):");
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), temperatures.size()); ++i) {
            writeLog(logFile_, "------" + nodeNames[i] + ": " + std::to_string(temperatures[i]) + " ℃");
        }
        if (temperatures.size() > 5) {
            writeLog(logFile_, "------... (残り " + std::to_string(temperatures.size() - 5) + " 個のノード)");
        }
        
        // 収束性改善の提案
        writeLog(logFile_, "----収束改善の提案:");
        if (summary.final_cost > constants.thermalTolerance * 1000) {
            writeLog(logFile_, "------熱計算許容誤差を緩める (1e-4 〜 1e-3 程度に設定)");
        }
        if (summary.iterations.size() >= constants.maxInnerIteration) {
            writeLog(logFile_, "------最大反復回数を増やす (200〜500回程度に設定)");
        }
        writeLog(logFile_, "----詳細レポート: " + summary.BriefReport());
        
        // 自動的にシステム診断情報を出力
        outputThermalSystemDiagnostics();
    }

    // 計算結果の出力
    //writeLog(logFile_, "--温度値: " + pure_to_string(temperatures));
    TemperatureMap tempMap = extractTemperatures(temperatures, nodeNames);
    //writeLog(logFile_, "--温度マップ: " + map_to_string(tempMap));
    HeatRateMap heatRates = calculateHeatRates(tempMap);
    //writeLog(logFile_, "--熱流量マップ: " + map_to_string(heatRates));
    HeatBalanceMap balance = verifyBalance(heatRates);
    //writeLog(logFile_, "--バランス検証: " + map_to_string(balance));

    return {tempMap, heatRates, balance};
}

// =============================================================================
// 診断情報出力
// =============================================================================

void ThermalSolver::outputThermalSystemDiagnostics() {
    writeLog(logFile_, "--熱システム診断情報:");
    
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
    
    writeLog(logFile_, "----ノード数: " + std::to_string(totalNodes));
    
    // ブランチ数統計
    size_t totalBranches = boost::num_edges(graph);
    writeLog(logFile_, "----ブランチ数: " + std::to_string(totalBranches));
    
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
        writeLog(logFile_, "----熱特性値の統計:");
        writeLog(logFile_, "------最小: " + std::to_string(*minMax.first) + " W/K");
        writeLog(logFile_, "------最大: " + std::to_string(*minMax.second) + " W/K");
        if (*minMax.first > 0) {
            writeLog(logFile_, "------比率: " + std::to_string(*minMax.second / *minMax.first));
        }
    }
    
    writeLog(logFile_, "----固定温度ノード数: " + std::to_string(fixedTempNodes));
    writeLog(logFile_, "----計算ノード数: " + std::to_string(thermalCalcNodes));
    
    // 固定温度の範囲
    if (!fixedTemps.empty()) {
        auto minMax = std::minmax_element(fixedTemps.begin(), fixedTemps.end());
        writeLog(logFile_, "----固定温度の範囲:");
        writeLog(logFile_, "------最小: " + std::to_string(*minMax.first) + " ℃");
        writeLog(logFile_, "------最大: " + std::to_string(*minMax.second) + " ℃");
        writeLog(logFile_, "------範囲: " + std::to_string(*minMax.second - *minMax.first) + " ℃");
    }
    
    // 計算温度の範囲（現在値）
    if (!calcTemps.empty()) {
        auto minMax = std::minmax_element(calcTemps.begin(), calcTemps.end());
        writeLog(logFile_, "----計算温度の現在範囲:");
        writeLog(logFile_, "------最小: " + std::to_string(*minMax.first) + " ℃");
        writeLog(logFile_, "------最大: " + std::to_string(*minMax.second) + " ℃");
        writeLog(logFile_, "------範囲: " + std::to_string(*minMax.second - *minMax.first) + " ℃");
    }
}