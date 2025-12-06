#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "core/thermal_solver.h"

#include <iostream>
#include <set>
#include <stdexcept>
#include <fstream>
#include <unordered_map>

#include <boost/graph/adjacency_list.hpp>

// ノードを追加
Vertex ThermalNetwork::addNode(const VertexProperties& node) {
    VertexProperties props = node;
    props.name = node.key;  // nameフィールドにkeyの値を設定

    Vertex v = boost::add_vertex(props, graph);
    keyToVertex[node.key] = v;
    return v;
}

// エッジを追加
void ThermalNetwork::addEdge(const EdgeProperties& edge) {
    auto sourceIt = keyToVertex.find(edge.source);
    auto targetIt = keyToVertex.find(edge.target);
    if (sourceIt != keyToVertex.end() && targetIt != keyToVertex.end()) {
        boost::add_edge(sourceIt->second, targetIt->second, edge, graph);
    }
}

// ノード数を取得
int ThermalNetwork::getNodeCount() const {
    return static_cast<int>(boost::num_vertices(graph));
}

// エッジ数を取得
int ThermalNetwork::getEdgeCount() const {
    return static_cast<int>(boost::num_edges(graph));
}

// キーによるノードアクセス
VertexProperties& ThermalNetwork::getNode(const std::string& key) {
    return graph[keyToVertex.at(key)];
}

const VertexProperties& ThermalNetwork::getNode(const std::string& key) const {
    return graph[keyToVertex.at(key)];
}

// データからネットワークを構築（熱ブランチのみ）
void ThermalNetwork::buildFromData(const std::vector<VertexProperties>& allNodes,
                                   const std::vector<EdgeProperties>& thermalBranches,
                                   const std::vector<EdgeProperties>& ventilationBranches,
                                   const SimulationConstants& simConstants,
                                   std::ostream& logs) {

    if (simConstants.temperatureCalc) {
        logs << "--熱回路網を作成中...\n";
        const int verbosity = (simConstants.logVerbosity > 0) ? simConstants.logVerbosity : 2;
        // 熱ブランチの両端ノードを収集
        std::set<std::string> allNodeKeys;
        for (const auto& edge : thermalBranches) {
            allNodeKeys.insert(edge.source);
            allNodeKeys.insert(edge.target);
        }
        // 換気ブランチの両端ノードも対象に含める
        for (const auto& edge : ventilationBranches) {
            allNodeKeys.insert(edge.source);
            allNodeKeys.insert(edge.target);
        }

        // 対象ノードを追加
        for (const auto& node : allNodes) {
            if (allNodeKeys.find(node.key) != allNodeKeys.end()) {
                addNode(node);
                if (verbosity >= 2) {
                    logs << "---熱回路網にノードを追加: " << node.key << " (" << node.type << ") "
                         << "calc_p:" << (node.calc_p ? "true" : "false") << " "
                         << "calc_t:" << (node.calc_t ? "true" : "false") << " "
                         << "calc_x:" << (node.calc_x ? "true" : "false") << " "
                         << "calc_c:" << (node.calc_c ? "true" : "false") << "\n";
                }
            }
        }

        // 熱ブランチを追加
        for (const auto& edge : thermalBranches) {
            addEdge(edge);
        }

        // 換気ブランチに対応する移流熱ブランチを作成・追加
        for (const auto& ventEdge : ventilationBranches) {
            EdgeProperties advectionEdge;
            advectionEdge.key = "advection_" + ventEdge.key;
            advectionEdge.unique_id = "advection_" + ventEdge.unique_id;
            advectionEdge.type = "advection";
            advectionEdge.source = ventEdge.source;
            advectionEdge.target = ventEdge.target;
            advectionEdge.flow_rate = 0.0; // 初期値
            advectionEdge.comment = "換気ブランチ " + ventEdge.key + " に対応する移流熱ブランチ";
            addEdge(advectionEdge);
        }
        if (!ventilationBranches.empty() && verbosity >= 2) {
            logs << "---換気ブランチに対応する移流熱ブランチを " << ventilationBranches.size() << " 個作成しました\n";
        }
    }

    logs << "--熱回路網を作成しました: "
         << getNodeCount() << "ノード, "
         << getEdgeCount() << "ブランチ\n";
}

// 換気回路網から風量を同期
void ThermalNetwork::syncFlowRatesFromVentilationNetwork(const VentilationNetwork& ventNetwork) {
    const auto& ventGraph = ventNetwork.getGraph();

    // 熱回路網側で移流エッジのマップを構築
    std::unordered_map<std::string, Edge> advectionEdgeByKey;
    advectionEdgeByKey.reserve(boost::num_edges(graph));
    auto thermal_edge_range = boost::edges(graph);
    for (auto thermal_edge : boost::make_iterator_range(thermal_edge_range)) {
        const auto& thermalEdgeProps = graph[thermal_edge];
        if (thermalEdgeProps.type == "advection" &&
            thermalEdgeProps.key.rfind("advection_", 0) == 0) {
            advectionEdgeByKey.emplace(thermalEdgeProps.key, thermal_edge);
        }
    }

    // 換気エッジを走査し、対応する移流エッジに風量をコピー
    auto vent_edge_range = boost::edges(ventGraph);
    for (auto vent_edge : boost::make_iterator_range(vent_edge_range)) {
        const auto& ventEdge = ventGraph[vent_edge];
        std::string advectionKey = "advection_" + ventEdge.key;
        auto it = advectionEdgeByKey.find(advectionKey);
        if (it == advectionEdgeByKey.end()) continue;
        graph[it->second].flow_rate = ventEdge.flow_rate;
    }
}

// ノード温度更新
void ThermalNetwork::updateNodeTemperatures(const TemperatureMap& tempMap) {
    for (auto const& [name, temp] : tempMap) {
        auto it = keyToVertex.find(name);
        if (it != keyToVertex.end()) {
            graph[it->second].current_t = temp;
        }
    }
}

// 熱流量をグラフに更新
void ThermalNetwork::updateHeatRatesInGraph(const HeatRateMap& heatRates) {
    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        Vertex sourceVertex = boost::source(edge, graph);
        Vertex targetVertex = boost::target(edge, graph);
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];

        std::pair<std::string, std::string> edgeKey = {sourceNode.key, targetNode.key};
        auto heatIt = heatRates.find(edgeKey);
        if (heatIt != heatRates.end()) {
            graph[edge].heat_rate = heatIt->second;
        }
    }
}

// 温度と熱流量を同時に更新
void ThermalNetwork::updateCalculationResults(const TemperatureMap& tempMap, const HeatRateMap& heatRates) {
    updateNodeTemperatures(tempMap);
    updateHeatRatesInGraph(heatRates);
}


// 温度計算
std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap>
ThermalNetwork::calculateTemperature(const SimulationConstants& constants, std::ostream& logs) {
    ThermalSolver solver(*this, logs);
    return solver.solveTemperatures(constants);
}

// 熱流量データを収集（個別ブランチの熱流量データを返す）
std::map<std::string, double> ThermalNetwork::collectHeatRates() const {
    std::map<std::string, double> heatRates;

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

        heatRates[key] = edgeData.heat_rate;
    }

    return heatRates;
}

// タイムステップに応じてノードとエッジの時変プロパティを更新
void ThermalNetwork::updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                                  const std::vector<EdgeProperties>& thermalBranches,
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
    
    // 熱ブランチ/換気ブランチを高速参照するマップを構築
    std::unordered_map<std::string, const EdgeProperties*> thermalBranchById;
    thermalBranchById.reserve(thermalBranches.size());
    for (const auto& branch : thermalBranches) {
        thermalBranchById.emplace(branch.unique_id, &branch);
    }
    (void)ventilationBranches; // 現状、移流エッジでは換気ブランチの時系列更新は不要

    // エッジのプロパティを更新（unique_idでマッチング）
    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        EdgeProperties& graphEdge = graph[edge];

        if (graphEdge.type == "advection" && graphEdge.key.rfind("advection_", 0) == 0) {
            // 移流エッジは換気ブランチに紐付く。現時点で更新する時変プロパティはないため処理なし。
            continue;
        }

        auto thermalIt = thermalBranchById.find(graphEdge.unique_id);
        if (thermalIt == thermalBranchById.end()) {
            continue;
        }

        const EdgeProperties* srcBranch = thermalIt->second;
        // 時系列データから現在のタイムステップの値を更新
        graphEdge.updateForTimestep(static_cast<int>(timestep));
        // 時系列データも更新（次回の更新に備える）
        graphEdge.heat_generation = srcBranch->heat_generation;
        graphEdge.enabled = srcBranch->enabled;
    }
} 