#include "network/ventilation_network.h"
#include "core/pressure_solver.h"
#include "core/flow_calculation.h"
#include "utils/utils.h"

#include <iostream>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <fstream>

#include <boost/graph/adjacency_list.hpp>

// ノードを追加
Vertex VentilationNetwork::addNode(const VertexProperties& node) {
    VertexProperties props = node;
    props.name = node.key;  // nameフィールドにkeyの値を設定

    Vertex v = boost::add_vertex(props, graph);
    keyToVertex[node.key] = v;
    return v;
}

// エッジを追加
void VentilationNetwork::addEdge(const EdgeProperties& edge) {
    auto sourceIt = keyToVertex.find(edge.source);
    auto targetIt = keyToVertex.find(edge.target);
    if (sourceIt != keyToVertex.end() && targetIt != keyToVertex.end()) {
        boost::add_edge(sourceIt->second, targetIt->second, edge, graph);
    }
}

// ノード数を取得
int VentilationNetwork::getNodeCount() const {
    return static_cast<int>(boost::num_vertices(graph));
}

// エッジ数を取得
int VentilationNetwork::getEdgeCount() const {
    return static_cast<int>(boost::num_edges(graph));
}

// キーによるノードアクセス
VertexProperties& VentilationNetwork::getNode(const std::string& key) {
    return graph[keyToVertex.at(key)];
}

// データから換気回路網を構築
void VentilationNetwork::buildFromData(const std::vector<VertexProperties>& allNodes,
                                       const std::vector<EdgeProperties>& ventilationBranches,
                                       const SimulationConstants& simConstants,
                                       std::ostream& logs) {

    if (simConstants.pressureCalc) {
        logs << "--換気回路網を作成中...\n";
        const int verbosity = (simConstants.logVerbosity > 0) ? simConstants.logVerbosity : 2;
        // 換気ブランチ両端のノードを収集
        std::set<std::string> allNodeKeys;
        for (const auto& edge : ventilationBranches) {
            allNodeKeys.insert(edge.source);
            allNodeKeys.insert(edge.target);
        }

        // 必要なノードのみ追加
        for (const auto& node : allNodes) {
            if (allNodeKeys.find(node.key) != allNodeKeys.end()) {
                addNode(node);
                if (verbosity >= 2) {
                    logs << "---換気回路網にノードを追加: " << node.key << " (" << node.type << ") "
                         << "calc_p:" << (node.calc_p ? "true" : "false") << " "
                         << "calc_t:" << (node.calc_t ? "true" : "false") << " "
                         << "calc_x:" << (node.calc_x ? "true" : "false") << " "
                         << "calc_c:" << (node.calc_c ? "true" : "false") << "\n";
                }
            }
        }

        // 換気ブランチを追加
        for (const auto& edge : ventilationBranches) {
            addEdge(edge);
        }
    }

    logs << "--換気回路網を作成しました: "
         << getNodeCount() << "ノード, "
         << getEdgeCount() << "ブランチ\n";
}

// ノード圧力更新
void VentilationNetwork::updateNodePressures(const PressureMap& pressureMap) {
    for (auto const& [name, pressure] : pressureMap) {
        auto it = keyToVertex.find(name);
        if (it != keyToVertex.end()) {
            graph[it->second].current_p = pressure;
        }
    }
}

// ノード温度更新
void VentilationNetwork::updateNodeTemperatures(const TemperatureMap& tempMap) {
    for (auto const& [name, temp] : tempMap) {
        auto it = keyToVertex.find(name);
        if (it != keyToVertex.end()) {
            graph[it->second].current_t = temp;
        }
    }
}

// 圧力計算
std::tuple<PressureMap, std::map<std::pair<std::string, std::string>, double>, FlowBalanceMap>
VentilationNetwork::calculatePressure(const SimulationConstants& constants, std::ostream& logs) {
    PressureSolver solver(*this, logs);
    auto [pressureMap, flowRates, balanceMap] = solver.solvePressures(constants);
    return {pressureMap, flowRates, balanceMap};
}

// 流量マップからグラフを一括更新
void VentilationNetwork::updateFlowRatesInGraph(const FlowRateMap& /*flowRates*/) {
    // 個別ブランチごとに (P - ρgh) を用いた流量を再計算して格納
    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        Vertex sourceVertex = boost::source(edge, graph);
        Vertex targetVertex = boost::target(edge, graph);
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];
        const auto& edgeData   = graph[edge];

        double rho_source = calculateDensity(sourceNode.current_t);
        double rho_target = calculateDensity(targetNode.current_t);

        double source_total_pressure = sourceNode.current_p - rho_source * 9.81 * edgeData.h_from;
        double target_total_pressure = targetNode.current_p - rho_target * 9.81 * edgeData.h_to;
        double dp = source_total_pressure - target_total_pressure;

        double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
        graph[edge].flow_rate = std::isfinite(flow) ? flow : 0.0;
    }
}

// 風量データを収集（個別ブランチの風量データを返す）
std::map<std::string, double> VentilationNetwork::collectFlowRates() const {
    std::map<std::string, double> flowRates;

    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        const auto& edgeData = graph[edge];

        // 出力用キーから末尾の"_000"を除去（存在する場合のみ）
        std::string key = edgeData.unique_id;
        const std::string suffix = "_000";
        if (key.size() > suffix.size() &&
            key.rfind(suffix) == key.size() - suffix.size()) {
            key.erase(key.size() - suffix.size());
        }

        // ソルバー確定後に保存された枝流量をそのまま使用
        flowRates[key] = edgeData.flow_rate;
    }

    return flowRates;
}

// 圧力と風量を同時に更新
void VentilationNetwork::updateCalculationResults(const PressureMap& pressureMap, const FlowRateMap& flowRates) {
    // 圧力を更新
    updateNodePressures(pressureMap);
    
    // 風量を更新
    updateFlowRatesInGraph(flowRates);
}

// タイムステップに応じてノードとエッジの時変プロパティを更新
void VentilationNetwork::updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                                     const std::vector<EdgeProperties>& ventilationBranches,
                                                     long timestep) {
    // ノードのプロパティを更新（keyでマッチング）
    for (const auto& node : allNodes) {
        auto it = keyToVertex.find(node.key);
        if (it != keyToVertex.end()) {
            VertexProperties& graphNode = graph[it->second];
            // 時系列データから現在のタイムステップの値を更新
            graphNode.updateForTimestep(static_cast<int>(timestep));
        }
    }
    
    // エッジのプロパティを更新（unique_idでマッチング）
    // 事前に unique_id -> ブランチ へのマップを作っておき、探索コストを削減する
    std::unordered_map<std::string, const EdgeProperties*> branchById;
    branchById.reserve(ventilationBranches.size());
    for (const auto& branch : ventilationBranches) {
        branchById[branch.unique_id] = &branch;
    }

    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        EdgeProperties& graphEdge = graph[edge];

        auto it = branchById.find(graphEdge.unique_id);
        if (it == branchById.end()) {
            continue;
        }

        const EdgeProperties* srcBranch = it->second;
        // 時系列データから現在のタイムステップの値を更新
        graphEdge.updateForTimestep(static_cast<int>(timestep));
        // 時系列データ自体も更新（次回の更新に備える）
        graphEdge.vol = srcBranch->vol;
        graphEdge.enabled = srcBranch->enabled;
    }
}