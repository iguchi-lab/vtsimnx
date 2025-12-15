#include "core/pressure_solver.h"
#include "core/flow_calculation.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "../archenv/include/archenv.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <limits>

double calculateDensity(double temperature) {
    return archenv::STANDARD_ATMOSPHERIC_PRESSURE / 
           (archenv::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

// =============================================================================
// PressureSolverクラス - 圧力・風量計算のメインソルバー
// =============================================================================

PressureSolver::PressureSolver(VentilationNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

// =============================================================================
// ヘルパー関数
// =============================================================================

double PressureSolver::calculateTotalPressure(double pressure, double temperature, double height) const {
    double rho = calculateDensity(temperature);
    return pressure - rho * archenv::GRAVITY * height;
}

double PressureSolver::calculatePressureDifference(
    const VertexProperties& sourceNode,
    const VertexProperties& targetNode,
    const EdgeProperties& edgeData,
    const PressureMap& pressureMap) const {
    
    auto sourcePressureIt = pressureMap.find(sourceNode.key);
    auto targetPressureIt = pressureMap.find(targetNode.key);
    
    if (sourcePressureIt == pressureMap.end() || targetPressureIt == pressureMap.end()) {
        return 0.0;
    }
    
    double source_total = calculateTotalPressure(
        sourcePressureIt->second, sourceNode.current_t, edgeData.h_from);
    double target_total = calculateTotalPressure(
        targetPressureIt->second, targetNode.current_t, edgeData.h_to);
    
    return source_total - target_total;
}

// =============================================================================
// 初期化関数
// =============================================================================

void PressureSolver::setInitialPressures(std::vector<double>& pressures, 
                                        const std::vector<std::string>& nodeNames) {
    // サイズチェック（初期化前に実行）
    if (pressures.size() != nodeNames.size()) {
        writeLog(logFile_, "--警告: 圧力配列とノード名配列のサイズが一致しません (" + 
                 std::to_string(pressures.size()) + " vs " + std::to_string(nodeNames.size()) + ")");
        return;
    }
    
    // 初期圧力を設定（トポロジ上の current_p をそのまま使用）
    // ※人工的な初期勾配（i*10Pa等）は入れない
    const auto& graph = network_.getGraph();
    const auto& keyToVertex = network_.getKeyToVertex();
    for (size_t i = 0; i < pressures.size(); ++i) {
        auto it = keyToVertex.find(nodeNames[i]);
        if (it != keyToVertex.end()) {
            pressures[i] = graph[it->second].current_p;
        } else {
            pressures[i] = 0.0;
        }
    }
}

bool PressureSolver::initializeSolverSetup(SolverSetup& setup) {
    const auto& graph = network_.getGraph();
    auto vertex_range = boost::vertices(graph);
    size_t parameterIndex = 0;
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& nodeData = graph[vertex];
        if (nodeData.calc_p) {
            setup.nodeNames.push_back(nodeData.key);
            setup.vertexToParameterIndex[vertex] = parameterIndex++;
        }
    }

    if (setup.nodeNames.empty()) {
        writeLog(logFile_, "--警告: 圧力計算対象のノードがありません");
        return false;
    }

    setup.pressures.resize(setup.nodeNames.size());
    setInitialPressures(setup.pressures, setup.nodeNames);
    return true;
}

// =============================================================================
// Ceres問題の構築
// =============================================================================

void PressureSolver::addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem) {
    for (const std::string& nodeName : setup.nodeNames) {
        ceres::CostFunction* costFunction = new FlowBalanceConstraint(
            nodeName,
            network_.getGraph(),
            network_.getKeyToVertex(),
            setup.vertexToParameterIndex,
            setup.pressures.size(),
            logFile_
        );

        problem.AddResidualBlock(costFunction, nullptr, const_cast<double*>(setup.pressures.data()));
    }
}

// =============================================================================
// 圧力・風量の抽出と検証
// =============================================================================

PressureMap PressureSolver::extractPressures(const std::vector<double>& pressures,
                                            const std::vector<std::string>& nodeNames) {
    PressureMap pressureMap;
    // 計算対象ノードの圧力をマップに追加
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        pressureMap[nodeNames[i]] = pressures[i];
    }
    
    // 固定圧力ノードの圧力をマップに追加
    const auto& graph = network_.getGraph();
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& properties = graph[vertex];
        if (!properties.calc_p) {
            pressureMap[properties.key] = properties.current_p;
        }
    }
    return pressureMap;
}

FlowBalanceMap PressureSolver::verifyBalance(const FlowRateMap& flowRates) {
    FlowBalanceMap balance;

    const auto& graph = network_.getGraph();

    // ノードごとの初期化
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        balance[graph[vertex].key] = 0.0;
    }

    // 計算済みの flowRates から入出流を構築
    for (const auto& kv : flowRates) {
        const auto& edgeKey = kv.first; // {source, target}
        const std::string& srcName = edgeKey.first;
        const std::string& dstName = edgeKey.second;
        double q = kv.second; // src -> dst を正

        if (q == 0.0) continue;

        // 符号に関係なく一貫した更新
        balance[srcName] -= q; // 出た分だけ減算
        balance[dstName] += q; // 入った分だけ加算
    }

    return balance;
}

// =============================================================================
// 風量計算
// =============================================================================

FlowRateMap PressureSolver::calculateFlowRates(const PressureMap& pressureMap) {
    FlowRateMap flowRates;
    
    const auto& graph = network_.getGraph();
    auto edge_range = boost::edges(graph);
    
    for (auto edge : boost::make_iterator_range(edge_range)) {
        auto sourceVertex = boost::source(edge, graph);
        auto targetVertex = boost::target(edge, graph);
        
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];
        const auto& edgeData = graph[edge];
        
        // 圧力差を計算
        double dp = calculatePressureDifference(sourceNode, targetNode, edgeData, pressureMap);
        if (dp == 0.0 && (pressureMap.find(sourceNode.key) == pressureMap.end() || 
                          pressureMap.find(targetNode.key) == pressureMap.end())) {
            writeLog(logFile_, "--警告: ノード圧力が見つかりません - " + 
                     sourceNode.key + " → " + targetNode.key);
            continue;
        }
        
        double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
        
        // 異常値チェック
        if (!std::isfinite(flow)) {
            writeLog(logFile_, "--警告: 無限大または非数の風量値が検出されました - " + 
                     sourceNode.key + " → " + targetNode.key);
            flow = 0.0;
        }
        
        std::pair<std::string, std::string> edgeKey = {sourceNode.key, targetNode.key};
        flowRates[edgeKey] += flow;  // 同一ノードペア間の流量を合計
    }
    
    return flowRates;
}

std::map<std::string, double> PressureSolver::calculateIndividualFlowRates(const PressureMap& pressureMap) {
    std::map<std::string, double> individualFlowRates;
    
    const auto& graph = network_.getGraph();
    auto edge_range = boost::edges(graph);
    
    for (auto edge : boost::make_iterator_range(edge_range)) {
        auto sourceVertex = boost::source(edge, graph);
        auto targetVertex = boost::target(edge, graph);
        
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];
        const auto& edgeData = graph[edge];
        
        // 圧力差を計算
        double dp = calculatePressureDifference(sourceNode, targetNode, edgeData, pressureMap);
        if (dp == 0.0 && (pressureMap.find(sourceNode.key) == pressureMap.end() || 
                          pressureMap.find(targetNode.key) == pressureMap.end())) {
            writeLog(logFile_, "--警告: ノード圧力が見つかりません - " + 
                     sourceNode.key + " → " + targetNode.key);
            continue;
        }
        
        double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
        
        // 異常値チェック
        if (!std::isfinite(flow)) {
            writeLog(logFile_, "--警告: 無限大または非数の風量値が検出されました - " + edgeData.unique_id);
            flow = 0.0;
        }
        
        // ユニークIDをキーとして個別ブランチの流量を保存
        individualFlowRates[edgeData.unique_id] = flow;
    }
    
    return individualFlowRates;
}

// =============================================================================
// メイン圧力計算
// =============================================================================

PressureSolver::SolverResult PressureSolver::solvePressures(
    const SimulationConstants& constants) {
    SolverSetup setup;
    if (!initializeSolverSetup(setup)) {
        return SolverResult{PressureMap{}, FlowRateMap{}, FlowBalanceMap{}};
    }
    auto& nodeNames = setup.nodeNames;
    auto& pressures = setup.pressures;

    // まず Newton-GS+SOR を試行（圧力版）
    writeLog(logFile_, "--------Newton-GS+SOR(圧力)ソルバーを試行します...");
    try {
        constexpr double kSorOmega = 1.2;
        auto ngsResult = PressureSolverNewtonGS::solvePressuresNewtonGS(
            network_, constants, kSorOmega, logFile_);

        // 残差チェック（maxBalanceで判定）
        double maxBalance = 0.0;
        for (const auto& [node, bal] : std::get<2>(ngsResult)) {
            maxBalance = std::max(maxBalance, std::abs(bal));
        }
        if (maxBalance <= constants.ventilationTolerance) {
            network_.setLastPressureConverged(true);
            return ngsResult;
        } else {
            writeLog(logFile_, "--------Newton-GS+SOR(圧力)はバランス超過 (maxBalance="
                               + std::to_string(maxBalance) + ", tol=" + std::to_string(constants.ventilationTolerance)
                               + ")。Ceresにフォールバックします...");
        }
    } catch (const std::exception& e) {
        writeLog(logFile_, "--------Newton-GS+SOR(圧力)でエラー: " + std::string(e.what()) + " Ceresにフォールバックします...");
    }

    ceres::Problem problem;
    addFlowBalanceConstraints(setup, problem);

    ceres::Solver::Summary summary;
    runPrimarySolvers(constants, problem, summary);

    // 結果のログ出力
    // 実際の残差をチェックして真の収束判定を行う
    bool reallyConverged = (summary.termination_type == ceres::CONVERGENCE) && 
                   (summary.final_cost <= constants.ventilationTolerance);
        
    if (reallyConverged) {
        network_.setLastPressureConverged(true);
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << summary.final_cost;
        std::string line = "---圧力計算収束 | iter=" + std::to_string(summary.iterations.size()) +
                          " | residual=" + oss.str() +
                          " | tol=" + std::to_string(constants.ventilationTolerance);
        writeLog(logFile_, line);
    } else {
        auto fallbackResult = runFallbackLoop(constants, setup, summary);
        if (fallbackResult) {
            return *fallbackResult;
        }
    }
    
    // 計算結果の出力
    //writeLog(logFile_, "--圧力値: " + pure_to_string(pressures));
    PressureMap pressureMap = extractPressures(pressures, nodeNames);
    //writeLog(logFile_, "--圧力マップ: " + map_to_string(pressureMap));
    FlowRateMap flowRates = calculateFlowRates(pressureMap);
    //writeLog(logFile_, "--風量マップ: " + map_to_string(flowRates));
    FlowBalanceMap balance = verifyBalance(flowRates);
    //writeLog(logFile_, "--バランス検証: " + map_to_string(balance));
    
    return SolverResult{pressureMap, flowRates, balance};
}


