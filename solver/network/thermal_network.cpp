#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "core/thermal/thermal_solver.h"
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
#include <boost/range/iterator_range.hpp>

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
    if (sourceIt == keyToVertex.end() || targetIt == keyToVertex.end()) {
        std::vector<std::string> missingNodes;
        if (sourceIt == keyToVertex.end()) missingNodes.push_back(edge.source);
        if (targetIt == keyToVertex.end()) missingNodes.push_back(edge.target);
        std::string message = "熱ブランチ '" + edge.key + "' に必要なノードが見つかりません: ";
        for (size_t i = 0; i < missingNodes.size(); ++i) {
            message += missingNodes[i];
            if (i + 1 < missingNodes.size()) message += ", ";
        }
        throw std::runtime_error(message);
    }
    boost::add_edge(sourceIt->second, targetIt->second, edge, graph);
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

    // 再構築に備えて内部状態をリセット（積み増し防止）
    graph = Graph{};
    keyToVertex.clear();
    lastThermalConverged = true;
    lastThermalRmseBalance = 0.0;
    lastThermalMaxBalance = 0.0;
    lastThermalMethod.clear();

    // 出力/同期キャッシュを無効化
    temperatureCacheInitialized = false;
    temperatureVerticesOrdered.clear();
    temperatureKeysOrdered.clear();
    temperatureVerticesOrderedCapacity.clear();
    temperatureKeysOrderedCapacity.clear();
    temperatureVerticesOrderedLayer.clear();
    temperatureKeysOrderedLayer.clear();
    heatRateCacheInitialized = false;
    heatRateEdgesOrderedAdvection.clear();
    heatRateKeysOrderedAdvection.clear();
    heatRateEdgesOrderedHeatGeneration.clear();
    heatRateKeysOrderedHeatGeneration.clear();
    heatRateEdgesOrderedSolarGain.clear();
    heatRateKeysOrderedSolarGain.clear();
    heatRateEdgesOrderedNocturnalLoss.clear();
    heatRateKeysOrderedNocturnalLoss.clear();
    heatRateEdgesOrderedConvection.clear();
    heatRateKeysOrderedConvection.clear();
    heatRateEdgesOrderedConduction.clear();
    heatRateKeysOrderedConduction.clear();
    heatRateEdgesOrderedRadiation.clear();
    heatRateKeysOrderedRadiation.clear();
    heatRateEdgesOrderedCapacity.clear();
    heatRateKeysOrderedCapacity.clear();
    advectionEdgeCacheInitialized = false;
    advectionEdgesByVertexPair.clear();
    advectionEdgeByVentUniqueId.clear();

    if (simConstants.temperatureCalc || simConstants.humidityCalc || simConstants.concentrationCalc) {
        writeLog(logs, "  熱回路網を作成中...");
        int verbosity = simConstants.logVerbosity;
        if (verbosity < 0) verbosity = 1;
        // 熱ブランチの両端ノードを収集
        std::set<std::string> allNodeKeys;
        if (simConstants.temperatureCalc || simConstants.humidityCalc) {
            for (const auto& edge : thermalBranches) {
                allNodeKeys.insert(edge.source);
                allNodeKeys.insert(edge.target);
            }
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
        // - 温度計算: 従来どおり全て利用
        // - 湿度計算: moisture_conductance を持つ枝を humidity_solver が参照するため読み込む
        if (simConstants.temperatureCalc || simConstants.humidityCalc) {
            for (const auto& edge : thermalBranches) {
                addEdge(edge);
            }
        }

        if (simConstants.temperatureCalc) {
            // response_conduction の履歴初期化（両端の初期温度で埋める）
            // - parseNodes(..., timestep=0) 済みなので current_t が入っている前提
            for (auto e : boost::make_iterator_range(boost::edges(graph))) {
                auto& ep = graph[e];
                if (ep.getTypeCode() != EdgeProperties::TypeCode::ResponseConduction) continue;
                const Vertex sv = boost::source(e, graph);
                const Vertex tv = boost::target(e, graph);
                const double Ts0 = graph[sv].current_t;
                const double Tt0 = graph[tv].current_t;

                const size_t tSrcLag = (ep.resp_a_src.size() > 0 ? ep.resp_a_src.size() - 1 : 0);
                const size_t tTgtLag = (ep.resp_a_tgt.size() > 0 ? ep.resp_a_tgt.size() - 1 : 0);
                // b係数の遅れも同じ長さを要求するのが自然だが、入力の自由度を保つため max を採用
                const size_t tLagMax = std::max({tSrcLag, tTgtLag,
                                                 (ep.resp_b_src.size() > 0 ? ep.resp_b_src.size() - 1 : 0),
                                                 (ep.resp_b_tgt.size() > 0 ? ep.resp_b_tgt.size() - 1 : 0)});

                ep.hist_t_src.assign(tLagMax, Ts0);
                ep.hist_t_tgt.assign(tLagMax, Tt0);
                ep.hist_q_src.assign(ep.resp_c_src.size(), 0.0);
                ep.hist_q_tgt.assign(ep.resp_c_tgt.size(), 0.0);
                ep.current_q_src = 0.0;
                ep.current_q_tgt = 0.0;
                ep.response_initialized = true;
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

                // エアコンノードへの流入（還気）判定を事前に行う
                auto itT = keyToVertex.find(ventEdge.target);
                if (itT != keyToVertex.end()) {
                    if (graph[itT->second].type == "aircon") {
                        advectionEdge.is_aircon_inflow = true;
                    }
                }

                addEdge(advectionEdge);
            }
            if (!ventilationBranches.empty() && verbosity >= 2) {
                std::ostringstream oss;
                oss << "    換気ブランチに対応する移流熱ブランチを " << ventilationBranches.size() << " 個作成しました";
                writeLog(logs, oss.str());
            }
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

    // 熱回路網側の移流エッジキャッシュを構築（初回のみ）
    // - 移流エッジの unique_id は "advection_" + 換気枝 unique_id で一意対応するため、順序に依存せず同期可能
    if (!advectionEdgeCacheInitialized) {
        advectionEdgesByVertexPair.clear();
        advectionEdgeByVentUniqueId.clear();
        advectionEdgesByVertexPair.reserve(boost::num_edges(graph));
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const auto& ep = graph[e];
            if (ep.getTypeCode() != EdgeProperties::TypeCode::Advection) continue;
            const Vertex sv = boost::source(e, graph);
            const Vertex tv = boost::target(e, graph);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sv)) << 32) |
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(tv));
            advectionEdgesByVertexPair[key].push_back(e);
            advectionEdgeByVentUniqueId[ep.unique_id] = e;
        }
        advectionEdgeCacheInitialized = true;
    }

    // 換気エッジを走査し、unique_id で対応する移流エッジに風量をコピー（boost::edges の順序に依存しない）
    for (auto vent_edge : boost::make_iterator_range(boost::edges(ventGraph))) {
        const auto& ventEp = ventGraph[vent_edge];
        const std::string thermalUniqueId = "advection_" + ventEp.unique_id;
        auto itAd = advectionEdgeByVentUniqueId.find(thermalUniqueId);
        if (itAd == advectionEdgeByVentUniqueId.end()) continue;
        graph[itAd->second].flow_rate = ventEp.flow_rate;
    }
}

// 温度計算
void ThermalNetwork::calculateTemperature(const SimulationConstants& constants, std::ostream& logs) {
    ThermalSolver solver(*this, logs);
    solver.solveTemperatures(constants);
}

void ThermalNetwork::setLastThermalConvergence(bool ok, double rmseBalance, double maxBalance, const std::string& method) {
    lastThermalConverged = ok;
    lastThermalRmseBalance = rmseBalance;
    lastThermalMaxBalance = maxBalance;
    lastThermalMethod = method;
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