#include "network/ventilation_network.h"
#include "core/pressure_solver.h"
#include "core/flow_calculation.h"
#include "utils/utils.h"
#include "network/thermal_network.h"

#include <iostream>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>

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
    if (sourceIt == keyToVertex.end() || targetIt == keyToVertex.end()) {
        std::vector<std::string> missingNodes;
        if (sourceIt == keyToVertex.end()) {
            missingNodes.push_back(edge.source);
        }
        if (targetIt == keyToVertex.end()) {
            missingNodes.push_back(edge.target);
        }
        std::string message = "換気ブランチ '" + edge.key + "' に必要なノードが見つかりません: ";
        for (size_t i = 0; i < missingNodes.size(); ++i) {
            message += missingNodes[i];
            if (i + 1 < missingNodes.size()) {
                message += ", ";
            }
        }
        throw std::runtime_error(message);
    }
    boost::add_edge(sourceIt->second, targetIt->second, edge, graph);
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
        writeLog(logs, "  換気回路網を作成中...");
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
                    std::ostringstream oss;
                    oss << "    換気回路網にノードを追加: " << node.key << " (" << node.type << ") "
                        << "calc_p:" << (node.calc_p ? "true" : "false") << " "
                        << "calc_t:" << (node.calc_t ? "true" : "false") << " "
                        << "calc_x:" << (node.calc_x ? "true" : "false") << " "
                        << "calc_c:" << (node.calc_c ? "true" : "false");
                    writeLog(logs, oss.str());
                }
            }
        }

        // 換気ブランチを追加
        for (const auto& edge : ventilationBranches) {
            addEdge(edge);
        }
    }

    {
        std::ostringstream oss;
        oss << "  換気回路網を作成しました: "
            << getNodeCount() << "ノード, "
            << getEdgeCount() << "ブランチ";
        writeLog(logs, oss.str());
    }
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

void VentilationNetwork::updateNodeTemperaturesFromThermalNetwork(const ThermalNetwork& thermalNetwork) {
    const auto& tKeyToV = thermalNetwork.getKeyToVertex();
    const auto& tGraph = thermalNetwork.getGraph();
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const std::string& key = graph[v].key;
        auto it = tKeyToV.find(key);
        if (it != tKeyToV.end()) {
            graph[v].current_t = tGraph[it->second].current_t;
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
void VentilationNetwork::updateFlowRatesInGraph(const FlowRateMap& flowRates) {
    // 外部I/Fは (string,string)->double の map だが、内部更新は vertex index に落として O(E) 化する
    std::unordered_map<std::uint64_t, double> flowByVertexPair;
    flowByVertexPair.reserve(flowRates.size() * 2 + 1);
    for (const auto& kv : flowRates) {
        const auto& srcKey = kv.first.first;
        const auto& dstKey = kv.first.second;
        auto itS = keyToVertex.find(srcKey);
        auto itT = keyToVertex.find(dstKey);
        if (itS == keyToVertex.end() || itT == keyToVertex.end()) continue;
        const Vertex sv = itS->second;
        const Vertex tv = itT->second;
        const std::uint64_t packed =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sv)) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(tv));
        flowByVertexPair[packed] = kv.second;
    }

    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        Vertex sourceVertex = boost::source(edge, graph);
        Vertex targetVertex = boost::target(edge, graph);
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];
        const auto& edgeData   = graph[edge];

        double flow = 0.0;
        bool flowResolved = false;

        const std::uint64_t fwd =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sourceVertex)) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(targetVertex));
        auto itF = flowByVertexPair.find(fwd);
        if (itF != flowByVertexPair.end()) {
            flow = itF->second;
            flowResolved = true;
        } else {
            const std::uint64_t rev =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(targetVertex)) << 32) |
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(sourceVertex));
            auto itR = flowByVertexPair.find(rev);
            if (itR != flowByVertexPair.end()) {
                flow = -itR->second;
                flowResolved = true;
            }
        }

        if (!flowResolved) {
            double rho_source = calculateDensity(sourceNode.current_t);
            double rho_target = calculateDensity(targetNode.current_t);

            double source_total_pressure = sourceNode.current_p - rho_source * 9.81 * edgeData.h_from;
            double target_total_pressure = targetNode.current_p - rho_target * 9.81 * edgeData.h_to;
            double dp = source_total_pressure - target_total_pressure;

            double recomputed = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
            flow = std::isfinite(recomputed) ? recomputed : 0.0;
        }

        graph[edge].flow_rate = flow;
    }
}

const std::vector<std::string>& VentilationNetwork::getPressureKeys() const {
    if (!pressureCacheInitialized) {
        // PressureMap は std::map でキーが昇順になるため、同じ順序（key昇順）で固定する。
        // ただし出力対象は calc_p=true のノードに限定する（従来の圧力mapと合わせる）。
        std::vector<std::pair<std::string, Vertex>> items;
        items.reserve(boost::num_vertices(graph));
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            if (nd.calc_p) {
                items.emplace_back(nd.key, v);
            }
        }
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        pressureVerticesOrdered.clear();
        pressureKeysOrdered.clear();
        pressureVerticesOrdered.reserve(items.size());
        pressureKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            pressureKeysOrdered.push_back(kv.first);
            pressureVerticesOrdered.push_back(kv.second);
        }
        pressureCacheInitialized = true;
    }
    return pressureKeysOrdered;
}

std::vector<double> VentilationNetwork::collectPressureValues() const {
    const auto& keys = getPressureKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(pressureVerticesOrdered.size());
    for (size_t i = 0; i < pressureVerticesOrdered.size(); ++i) {
        values[i] = graph[pressureVerticesOrdered[i]].current_p;
    }
    return values;
}

const std::vector<std::string>& VentilationNetwork::getFlowRateKeys() const {
    if (!flowRateCacheInitialized) {
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

        flowRateEdgesOrdered.clear();
        flowRateKeysOrdered.clear();
        flowRateEdgesOrdered.reserve(items.size());
        flowRateKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            flowRateKeysOrdered.push_back(kv.first);
            flowRateEdgesOrdered.push_back(kv.second);
        }
        flowRateCacheInitialized = true;
    }
    return flowRateKeysOrdered;
}

std::vector<double> VentilationNetwork::collectFlowRateValues() const {
    const auto& keys = getFlowRateKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(flowRateEdgesOrdered.size());
    for (size_t i = 0; i < flowRateEdgesOrdered.size(); ++i) {
        values[i] = graph[flowRateEdgesOrdered[i]].flow_rate;
    }
    return values;
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
    // 時系列を保持している graph 内のプロパティを更新（JSON再パースは不要）
    (void)allNodes;
    (void)ventilationBranches;

    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        graph[v].updateForTimestep(static_cast<int>(timestep));
    }
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        graph[e].updateForTimestep(static_cast<int>(timestep));
    }
}