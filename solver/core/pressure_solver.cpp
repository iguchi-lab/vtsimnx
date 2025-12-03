#include "core/pressure_solver.h"
#include "core/flow_calculation.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "core/physical_constants.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <limits>

// 物理定数は core/physical_constants.h から使用

double calculateDensity(double temperature) {
    return PhysicalConstants::STANDARD_ATMOSPHERIC_PRESSURE / 
           (PhysicalConstants::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

// =============================================================================
// PressureSolverクラス - 圧力・風量計算のメインソルバー
// =============================================================================

PressureSolver::PressureSolver(VentilationNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

void PressureSolver::setInitialPressures(std::vector<double>& pressures, 
                                        const std::vector<std::string>& nodeNames) {
    // 初期圧力を設定（各ノードに少しずつ異なる初期値）
    const double basePressure = 0.0; // Pa
    for (size_t i = 0; i < pressures.size(); ++i) {
        pressures[i] = basePressure + (i * 10.0);
    }
    
    // サイズチェック
    if (pressures.size() != nodeNames.size()) {
        writeLog(logFile_, "  警告: 圧力配列とノード名配列のサイズが一致しません");
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
        writeLog(logFile_, "  警告: 圧力計算対象のノードがありません");
        return false;
    }

    setup.pressures.resize(setup.nodeNames.size());
    setInitialPressures(setup.pressures, setup.nodeNames);
        return true;
    }
    
void PressureSolver::addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem) {
    for (const std::string& nodeName : setup.nodeNames) {
        auto constraint = new FlowBalanceConstraint(
            nodeName,
            network_.getGraph(),
            network_.getKeyToVertex(),
            setup.vertexToParameterIndex,
            logFile_
        );

        auto costFunction = new ceres::DynamicAutoDiffCostFunction<FlowBalanceConstraint>(constraint);
        costFunction->AddParameterBlock(setup.pressures.size());
        costFunction->SetNumResiduals(1);

        problem.AddResidualBlock(costFunction, nullptr, const_cast<double*>(setup.pressures.data()));
    }
}

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
        
        std::string sourceName = sourceNode.key;
        std::string targetName = targetNode.key;
        
        // 圧力値の存在チェック
        auto sourcePressureIt = pressureMap.find(sourceName);
        auto targetPressureIt = pressureMap.find(targetName);
        
        if (sourcePressureIt == pressureMap.end() || targetPressureIt == pressureMap.end()) {
            writeLog(logFile_, "　警告: ノード圧力が見つかりません - " + sourceName + " → " + targetName);
            continue;
        }
        
        double sourcePressure = sourcePressureIt->second;
        double targetPressure = targetPressureIt->second;
        
        // 高さ補正を含む圧力差（温度差による浮力を考慮）
        double sourceTemp = sourceNode.current_t;
        double targetTemp = targetNode.current_t;
        double rho_source = calculateDensity(sourceTemp);
        double rho_target = calculateDensity(targetTemp);
        
        // エッジ両端での静水圧補正を考慮した圧力差（参照高さは地面=0m）
        double source_total_pressure = sourcePressure - rho_source * PhysicalConstants::GRAVITY * edgeData.h_from;
        double target_total_pressure = targetPressure - rho_target * PhysicalConstants::GRAVITY * edgeData.h_to;
        double dp = source_total_pressure - target_total_pressure;
        
        double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
        
        // 異常値チェック
        if (!std::isfinite(flow)) {
            writeLog(logFile_, "　警告: 無限大または非数の風量値が検出されました - " + sourceName + " → " + targetName);
            flow = 0.0;
        }
        
        std::pair<std::string, std::string> edgeKey = {sourceName, targetName};
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
        
        std::string sourceName = sourceNode.key;
        std::string targetName = targetNode.key;
        
        // 圧力値の存在チェック
        auto sourcePressureIt = pressureMap.find(sourceName);
        auto targetPressureIt = pressureMap.find(targetName);
        
        if (sourcePressureIt == pressureMap.end() || targetPressureIt == pressureMap.end()) {
            writeLog(logFile_, "　警告: ノード圧力が見つかりません - " + sourceName + " → " + targetName);
            continue;
        }
        
        double sourcePressure = sourcePressureIt->second;
        double targetPressure = targetPressureIt->second;
        
        // 高さ補正を含む圧力差（温度差による浮力を考慮）
        double sourceTemp = sourceNode.current_t;
        double targetTemp = targetNode.current_t;
        double rho_source = calculateDensity(sourceTemp);
        double rho_target = calculateDensity(targetTemp);
        
        // エッジ両端での静水圧補正を考慮した圧力差（参照高さは地面=0m）
        double source_total_pressure = sourcePressure - rho_source * PhysicalConstants::GRAVITY * edgeData.h_from;
        double target_total_pressure = targetPressure - rho_target * PhysicalConstants::GRAVITY * edgeData.h_to;
        double dp = source_total_pressure - target_total_pressure;
        
        double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
        
        // 異常値チェック
        if (!std::isfinite(flow)) {
            writeLog(logFile_, "　警告: 無限大または非数の風量値が検出されました - " + edgeData.unique_id);
            flow = 0.0;
        }
        
        // ユニークIDをキーとして個別ブランチの流量を保存
        individualFlowRates[edgeData.unique_id] = flow;
    }
    
    return individualFlowRates;
}

PressureSolver::SolverResult PressureSolver::solvePressures(
    const SimulationConstants& constants) {
    SolverSetup setup;
    if (!initializeSolverSetup(setup)) {
        return SolverResult{PressureMap{}, FlowRateMap{}, FlowBalanceMap{}};
    }
    auto& nodeNames = setup.nodeNames;
    auto& pressures = setup.pressures;

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
        std::string line = "　　　圧力計算収束 | iter=" + std::to_string(summary.iterations.size()) +
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
    //writeLog(logFile_, "　　圧力値: " + pure_to_string(pressures));
    PressureMap pressureMap = extractPressures(pressures, nodeNames);
    //writeLog(logFile_, "　　圧力マップ: " + map_to_string(pressureMap));
    FlowRateMap flowRates = calculateFlowRates(pressureMap);
    //writeLog(logFile_, "　　風量マップ: " + map_to_string(flowRates));
    FlowBalanceMap balance = verifyBalance(flowRates);
    //writeLog(logFile_, "　　バランス検証: " + map_to_string(balance));
    
    return SolverResult{pressureMap, flowRates, balance};
}


