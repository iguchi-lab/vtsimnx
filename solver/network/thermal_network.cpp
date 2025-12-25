#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "core/thermal_solver.h"
#include "utils/utils.h"

#include <iostream>
#include <set>
#include <stdexcept>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cstdint>

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

    // 再構築に備えてキャッシュを無効化
    temperatureCacheInitialized = false;
    heatRateCacheInitialized = false;
    advectionEdgeCacheInitialized = false;
    advectionEdgeByVertexPair.clear();

    if (simConstants.temperatureCalc) {
        writeLog(logs, "  熱回路網を作成中...");
        int verbosity = simConstants.logVerbosity;
        if (verbosity < 0) verbosity = 1;
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
                    std::ostringstream oss;
                    oss << "    熱回路網にノードを追加: " << node.key << " (" << node.type << ") "
                        << "calc_p:" << (node.calc_p ? "true" : "false") << " "
                        << "calc_t:" << (node.calc_t ? "true" : "false") << " "
                        << "calc_x:" << (node.calc_x ? "true" : "false") << " "
                        << "calc_c:" << (node.calc_c ? "true" : "false");
                    writeLog(logs, oss.str());
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
            std::ostringstream oss;
            oss << "    換気ブランチに対応する移流熱ブランチを " << ventilationBranches.size() << " 個作成しました";
            writeLog(logs, oss.str());
        }
    }

    {
        std::ostringstream oss;
        oss << "  熱回路網を作成しました: "
            << getNodeCount() << "ノード, "
            << getEdgeCount() << "ブランチ";
        writeLog(logs, oss.str());
    }
}

// 換気回路網から風量を同期
void ThermalNetwork::syncFlowRatesFromVentilationNetwork(const VentilationNetwork& ventNetwork) {
    const auto& ventGraph = ventNetwork.getGraph();

    // 熱回路網側の移流エッジ（source/target vertex pair）キャッシュを構築（初回のみ）
    if (!advectionEdgeCacheInitialized) {
        advectionEdgeByVertexPair.clear();
        advectionEdgeByVertexPair.reserve(boost::num_edges(graph));
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const auto& ep = graph[e];
            if (ep.getTypeCode() != EdgeProperties::TypeCode::Advection) continue;
            const Vertex sv = boost::source(e, graph);
            const Vertex tv = boost::target(e, graph);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sv)) << 32) |
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(tv));
            advectionEdgeByVertexPair.emplace(key, e);
        }
        advectionEdgeCacheInitialized = true;
    }

    // 換気エッジを走査し、対応する移流エッジに風量をコピー
    auto vent_edge_range = boost::edges(ventGraph);
    for (auto vent_edge : boost::make_iterator_range(vent_edge_range)) {
        const auto& ventEp = ventGraph[vent_edge];
        // 換気側の source/target key を、熱側の vertex に変換して同期する
        auto itS = keyToVertex.find(ventEp.source);
        auto itT = keyToVertex.find(ventEp.target);
        if (itS == keyToVertex.end() || itT == keyToVertex.end()) continue;
        const Vertex sv = itS->second;
        const Vertex tv = itT->second;
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sv)) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(tv));
        auto it = advectionEdgeByVertexPair.find(key);
        if (it == advectionEdgeByVertexPair.end()) continue;
        graph[it->second].flow_rate = ventEp.flow_rate;
    }
}

// 温度計算
void ThermalNetwork::calculateTemperature(const SimulationConstants& constants, std::ostream& logs) {
    ThermalSolver solver(*this, logs);
    solver.solveTemperatures(constants);
}

const std::vector<std::string>& ThermalNetwork::getTemperatureKeys() const {
    if (!temperatureCacheInitialized) {
        // TemperatureMap は std::map でキーが昇順になるため、同じ順序（key昇順）で固定する
        std::vector<std::pair<std::string, Vertex>> items;
        items.reserve(boost::num_vertices(graph));
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            items.emplace_back(graph[v].key, v);
        }
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        temperatureVerticesOrdered.clear();
        temperatureKeysOrdered.clear();
        temperatureVerticesOrdered.reserve(items.size());
        temperatureKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            temperatureKeysOrdered.push_back(kv.first);
            temperatureVerticesOrdered.push_back(kv.second);
        }
        temperatureCacheInitialized = true;
    }
    return temperatureKeysOrdered;
}

std::vector<double> ThermalNetwork::collectTemperatureValues() const {
    const auto& keys = getTemperatureKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(temperatureVerticesOrdered.size());
    for (size_t i = 0; i < temperatureVerticesOrdered.size(); ++i) {
        values[i] = graph[temperatureVerticesOrdered[i]].current_t;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeys() const {
    if (!heatRateCacheInitialized) {
        // (key, edge) を集めてキーでソートし、順序を固定する
        std::vector<std::pair<std::string, Edge>> items;
        items.reserve(boost::num_edges(graph));
        for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
            const auto& edgeData = graph[edge];

            // 出力用キーから末尾の"_000"を除去（存在する場合のみ）
            std::string key = edgeData.unique_id;
            const std::string suffix = "_000";
            if (key.size() > suffix.size() &&
                key.rfind(suffix) == key.size() - suffix.size()) {
                key.erase(key.size() - suffix.size());
            }
            items.emplace_back(std::move(key), edge);
        }
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        heatRateEdgesOrdered.clear();
        heatRateKeysOrdered.clear();
        heatRateEdgesOrdered.reserve(items.size());
        heatRateKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            heatRateKeysOrdered.push_back(kv.first);
            heatRateEdgesOrdered.push_back(kv.second);
        }
        heatRateCacheInitialized = true;
    }
    return heatRateKeysOrdered;
}

std::vector<double> ThermalNetwork::collectHeatRateValues() const {
    const auto& keys = getHeatRateKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(heatRateEdgesOrdered.size());
    for (size_t i = 0; i < heatRateEdgesOrdered.size(); ++i) {
        values[i] = graph[heatRateEdgesOrdered[i]].heat_rate;
    }
    return values;
}

// タイムステップに応じてノードとエッジの時変プロパティを更新
void ThermalNetwork::updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                                  const std::vector<EdgeProperties>& thermalBranches,
                                                 const std::vector<EdgeProperties>& ventilationBranches,
                                                  long timestep) {
    // 時系列を保持している graph 内のプロパティを更新（JSON再パースは不要）
    (void)allNodes;
    (void)thermalBranches;
    (void)ventilationBranches;

    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        graph[v].updateForTimestep(static_cast<int>(timestep));
    }
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        auto& ep = graph[e];
        if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) {
            continue; // 換気同期でflow_rateが入るため時系列更新は不要
        }
        ep.updateForTimestep(static_cast<int>(timestep));
    }
} 