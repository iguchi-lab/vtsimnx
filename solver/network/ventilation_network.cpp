#include "network/ventilation_network.h"
#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/flow_calculation.h"
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
    // 再構築に備えて内部状態をリセット（積み増し防止）
    graph = Graph{};
    keyToVertex.clear();
    pressureCacheInitialized = false;
    pressureVerticesOrdered.clear();
    pressureKeysOrdered.clear();
    flowRateCacheInitialized = false;
    flowRateEdgesOrdered.clear();
    flowRateKeysOrdered.clear();
    invalidateSupernodeCache();
    lastPressureConverged = false;

    // pressureCalc=false でも、熱計算（移流）で換気ブランチの fixed_flow 等が必要になるため
    // 換気回路網自体は構築しておく（圧力は解かない）。
    if (simConstants.pressureCalc || simConstants.temperatureCalc || simConstants.humidityCalc || simConstants.concentrationCalc) {
        writeLog(logs, "  換気回路網を作成中...");
        int verbosity = simConstants.logVerbosity;
        if (verbosity < 0) verbosity = 1;
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

        // pressureCalc=false の場合でも、熱計算（移流）で流量が必要になる。
        // - fixed_flow はもちろん確定値
        // - type 未指定でも vol が与えられている枝（例: builder が出す「換気量固定」や aircon の送風）も
        //   「固定流量」とみなし、flow_rate を vol から設定する
        if (!simConstants.pressureCalc) {
            for (auto e : boost::make_iterator_range(boost::edges(graph))) {
                auto& ep = graph[e];
                if (ep.type == "fixed_flow" || !ep.vol.empty()) {
                    ep.flow_rate = ep.current_enabled ? ep.current_vol : 0.0;
                }
            }
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
    // NOTE:
    // `flowRates` はノードペア（source,target）ごとの合計流量（並列枝は合算）を持つ。
    // これをそのまま各ブランチへ配ると「並列枝が同じ流量になってしまう」ため、
    // ブランチ（エッジ）ごとに差圧から流量を再計算して graph[edge].flow_rate を更新する。
    (void)flowRates;

    auto edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(edge_range)) {
        Vertex sourceVertex = boost::source(edge, graph);
        Vertex targetVertex = boost::target(edge, graph);
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];
        auto& edgeData = graph[edge];

        double flow = 0.0;
        if (!edgeData.current_enabled) {
            flow = 0.0;
        } else if (edgeData.type == "fixed_flow" || !edgeData.vol.empty()) {
            // 固定流量（type=fixed_flow）や vol 指定はそのまま
            flow = edgeData.current_vol;
        } else {
            const double rho_source = calculateDensity(sourceNode.current_t);
            const double rho_target = calculateDensity(targetNode.current_t);
            const double source_total_pressure = sourceNode.current_p - rho_source * archenv::GRAVITY * edgeData.h_from;
            const double target_total_pressure = targetNode.current_p - rho_target * archenv::GRAVITY * edgeData.h_to;
            const double dp = source_total_pressure - target_total_pressure;

            const double recomputed = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
            flow = std::isfinite(recomputed) ? recomputed : 0.0;
        }

        edgeData.flow_rate = flow;
    }
}

const std::vector<std::string>& VentilationNetwork::getPressureKeys() const {
    if (!pressureCacheInitialized) {
        // directedS のため in_degree が使えない（in-edge リストを持たない）ので、
        // 全エッジを 1 回走査して「接続のある頂点」をマーキングする。
        std::vector<uint8_t> connected;
        connected.assign(boost::num_vertices(graph), 0);
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const Vertex sv = boost::source(e, graph);
            const Vertex tv = boost::target(e, graph);
            connected[static_cast<size_t>(sv)] = 1;
            connected[static_cast<size_t>(tv)] = 1;
        }

        // PressureMap は std::map でキーが昇順になるため、同じ順序（key昇順）で固定する。
        // 出力対象は「換気回路網で接続されているノード」のみに限定する（孤立ノードは除外）。
        std::vector<std::pair<std::string, Vertex>> items;
        items.reserve(boost::num_vertices(graph));
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            // 出力対象は normal + aircon + unknown のみ（capacity/layer は出力しない）
            const auto tc = nd.getTypeCode();
            if (tc == VertexProperties::TypeCode::Capacity) continue;
            if (tc == VertexProperties::TypeCode::Layer) continue;
            // 換気回路網で孤立しているノード（接続ブランチ無し）は出力しない
            // （圧力は換気回路網の結果として意味があるノードだけを出す）
            if (static_cast<size_t>(v) < connected.size() &&
                connected[static_cast<size_t>(v)] != 0) {
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

FlowRateMap VentilationNetwork::collectFlowRateMap() const {
    FlowRateMap out;
    // edge direction（source->target）に対して flow_rate を合算する。
    // 同じ (source, target) ペアに複数のエッジが存在する場合（例: 24h換気と局所換気スケジュールが
    // 同一ノード間に重複するケース）は流量を加算する。`=` による上書きだと、後から処理された
    // エッジの値だけが残り、スケジュール=0 の深夜帯に常時換気流量が消えるバグが生じる。
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        const Vertex sv = boost::source(e, graph);
        const Vertex tv = boost::target(e, graph);
        const auto& ep = graph[e];
        out[{graph[sv].key, graph[tv].key}] += ep.flow_rate;
    }
    return out;
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
        auto& ep = graph[e];
        ep.updateForTimestep(static_cast<int>(timestep));
        // pressure を解かない場合でも、fixed_flow および vol 指定枝は確定値なので flow_rate を追従させる
        // （type 未指定の「換気量固定」/ aircon 送風なども含む）
        if (ep.type == "fixed_flow" || !ep.vol.empty()) {
            ep.flow_rate = ep.current_enabled ? ep.current_vol : 0.0;
        }
    }
}