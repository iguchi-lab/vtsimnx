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

void ThermalNetwork::setLastThermalConvergence(bool ok, double rmseBalance, double maxBalance, const std::string& method) {
    lastThermalConverged = ok;
    lastThermalRmseBalance = rmseBalance;
    lastThermalMaxBalance = maxBalance;
    lastThermalMethod = method;
}

const std::vector<std::string>& ThermalNetwork::getTemperatureKeys() const {
    if (!temperatureCacheInitialized) {
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

        // TemperatureMap は std::map でキーが昇順になるため、同じ順序（key昇順）で固定する
        std::vector<std::pair<std::string, Vertex>> itemsMain;
        std::vector<std::pair<std::string, Vertex>> itemsCap;
        std::vector<std::pair<std::string, Vertex>> itemsLayer;
        itemsMain.reserve(boost::num_vertices(graph));
        itemsCap.reserve(boost::num_vertices(graph) / 8 + 1);
        itemsLayer.reserve(boost::num_vertices(graph) / 2 + 1);

        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            if (static_cast<size_t>(v) >= connected.size() || connected[static_cast<size_t>(v)] == 0) continue;

            const auto tc = nd.getTypeCode();
            if (tc == VertexProperties::TypeCode::Capacity) {
                itemsCap.emplace_back(nd.key, v);
            } else if (tc == VertexProperties::TypeCode::Layer) {
                itemsLayer.emplace_back(nd.key, v);
            } else {
                // main: normal + aircon + unknown
                itemsMain.emplace_back(nd.key, v);
            }
        }
        std::sort(itemsMain.begin(), itemsMain.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::sort(itemsCap.begin(), itemsCap.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::sort(itemsLayer.begin(), itemsLayer.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        temperatureVerticesOrdered.clear();
        temperatureKeysOrdered.clear();
        temperatureVerticesOrderedCapacity.clear();
        temperatureKeysOrderedCapacity.clear();
        temperatureVerticesOrderedLayer.clear();
        temperatureKeysOrderedLayer.clear();

        temperatureVerticesOrdered.reserve(itemsMain.size());
        temperatureKeysOrdered.reserve(itemsMain.size());
        temperatureVerticesOrderedCapacity.reserve(itemsCap.size());
        temperatureKeysOrderedCapacity.reserve(itemsCap.size());
        temperatureVerticesOrderedLayer.reserve(itemsLayer.size());
        temperatureKeysOrderedLayer.reserve(itemsLayer.size());

        for (const auto& kv : itemsMain) {
            temperatureKeysOrdered.push_back(kv.first);
            temperatureVerticesOrdered.push_back(kv.second);
        }
        for (const auto& kv : itemsCap) {
            temperatureKeysOrderedCapacity.push_back(kv.first);
            temperatureVerticesOrderedCapacity.push_back(kv.second);
        }
        for (const auto& kv : itemsLayer) {
            temperatureKeysOrderedLayer.push_back(kv.first);
            temperatureVerticesOrderedLayer.push_back(kv.second);
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

const std::vector<std::string>& ThermalNetwork::getTemperatureKeysCapacity() const {
    // main を呼べばキャッシュが構築される
    (void)getTemperatureKeys();
    return temperatureKeysOrderedCapacity;
}

std::vector<double> ThermalNetwork::collectTemperatureValuesCapacity() const {
    const auto& keys = getTemperatureKeysCapacity();
    (void)keys;
    std::vector<double> values;
    values.resize(temperatureVerticesOrderedCapacity.size());
    for (size_t i = 0; i < temperatureVerticesOrderedCapacity.size(); ++i) {
        values[i] = graph[temperatureVerticesOrderedCapacity[i]].current_t;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getTemperatureKeysLayer() const {
    // main を呼べばキャッシュが構築される
    (void)getTemperatureKeys();
    return temperatureKeysOrderedLayer;
}

std::vector<double> ThermalNetwork::collectTemperatureValuesLayer() const {
    const auto& keys = getTemperatureKeysLayer();
    (void)keys;
    std::vector<double> values;
    values.resize(temperatureVerticesOrderedLayer.size());
    for (size_t i = 0; i < temperatureVerticesOrderedLayer.size(); ++i) {
        values[i] = graph[temperatureVerticesOrderedLayer[i]].current_t;
    }
    return values;
}

namespace {
static inline std::string heatRateOutputKeyFromUniqueId(const std::string& uniqueId) {
    // 出力用キーから末尾の"_000"を除去（存在する場合のみ）
    std::string key = uniqueId;
    const std::string suffix = "_000";
    if (key.size() > suffix.size() &&
        key.rfind(suffix) == key.size() - suffix.size()) {
        key.erase(key.size() - suffix.size());
    }
    return key;
}
} // namespace

static void buildHeatRateCachesIfNeeded(const Graph& graph,
                                       bool& cacheInitialized,
                                       std::vector<Edge>& edgesAdvection,
                                       std::vector<std::string>& keysAdvection,
                                       std::vector<Edge>& edgesHeatGen,
                                       std::vector<std::string>& keysHeatGen,
                                       std::vector<Edge>& edgesSolar,
                                       std::vector<std::string>& keysSolar,
                                       std::vector<Edge>& edgesNoct,
                                       std::vector<std::string>& keysNoct,
                                       std::vector<Edge>& edgesConv,
                                       std::vector<std::string>& keysConv,
                                       std::vector<Edge>& edgesCond,
                                       std::vector<std::string>& keysCond,
                                       std::vector<Edge>& edgesRad,
                                       std::vector<std::string>& keysRad,
                                       std::vector<Edge>& edgesCap,
                                       std::vector<std::string>& keysCap) {
    if (cacheInitialized) return;

    std::vector<std::pair<std::string, Edge>> itemsAdvection;
    std::vector<std::pair<std::string, Edge>> itemsHeatGen;
    std::vector<std::pair<std::string, Edge>> itemsSolar;
    std::vector<std::pair<std::string, Edge>> itemsNoct;
    std::vector<std::pair<std::string, Edge>> itemsConv;
    std::vector<std::pair<std::string, Edge>> itemsCond;
    std::vector<std::pair<std::string, Edge>> itemsRad;
    std::vector<std::pair<std::string, Edge>> itemsCap;

    const size_t eCount = static_cast<size_t>(boost::num_edges(graph));
    itemsAdvection.reserve(eCount / 4 + 1);
    itemsHeatGen.reserve(eCount / 16 + 1);
    itemsSolar.reserve(eCount / 16 + 1);
    itemsNoct.reserve(eCount / 16 + 1);
    itemsConv.reserve(eCount / 8 + 1);
    itemsCond.reserve(eCount / 2 + 1);
    itemsRad.reserve(eCount / 8 + 1);
    itemsCap.reserve(eCount / 16 + 1);

    for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
        const auto& ep = graph[edge];
        const auto typeCode = ep.getTypeCode();
        const std::string key = heatRateOutputKeyFromUniqueId(ep.unique_id);

        if (typeCode == EdgeProperties::TypeCode::Advection) {
            itemsAdvection.emplace_back(key, edge);
            continue;
        }

        if (typeCode == EdgeProperties::TypeCode::HeatGeneration) {
            if (ep.subtype == "solar_gain") {
                itemsSolar.emplace_back(key, edge);
            } else if (ep.subtype == "nocturnal_loss") {
                itemsNoct.emplace_back(key, edge);
            } else {
                itemsHeatGen.emplace_back(key, edge);
            }
            continue;
        }

        // conductance（および未知タイプ）は subtype で分類する
        if (ep.subtype == "convection") {
            itemsConv.emplace_back(key, edge);
        } else if (ep.subtype == "radiation") {
            itemsRad.emplace_back(key, edge);
        } else if (ep.subtype == "capacity") {
            itemsCap.emplace_back(key, edge);
        } else {
            // "conduction" または未指定/未知は conduction 扱いに寄せる
            itemsCond.emplace_back(key, edge);
        }
    }

    auto sortItems = [](auto& items) {
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    };
    sortItems(itemsAdvection);
    sortItems(itemsHeatGen);
    sortItems(itemsSolar);
    sortItems(itemsNoct);
    sortItems(itemsConv);
    sortItems(itemsCond);
    sortItems(itemsRad);
    sortItems(itemsCap);

    auto fillOut = [](const auto& items, std::vector<Edge>& edgesOut, std::vector<std::string>& keysOut) {
        edgesOut.clear();
        keysOut.clear();
        edgesOut.reserve(items.size());
        keysOut.reserve(items.size());
        for (const auto& kv : items) {
            keysOut.push_back(kv.first);
            edgesOut.push_back(kv.second);
        }
    };

    fillOut(itemsAdvection, edgesAdvection, keysAdvection);
    fillOut(itemsHeatGen, edgesHeatGen, keysHeatGen);
    fillOut(itemsSolar, edgesSolar, keysSolar);
    fillOut(itemsNoct, edgesNoct, keysNoct);
    fillOut(itemsConv, edgesConv, keysConv);
    fillOut(itemsCond, edgesCond, keysCond);
    fillOut(itemsRad, edgesRad, keysRad);
    fillOut(itemsCap, edgesCap, keysCap);

    cacheInitialized = true;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysAdvection() const {
    buildHeatRateCachesIfNeeded(graph,
                                heatRateCacheInitialized,
                                heatRateEdgesOrderedAdvection, heatRateKeysOrderedAdvection,
                                heatRateEdgesOrderedHeatGeneration, heatRateKeysOrderedHeatGeneration,
                                heatRateEdgesOrderedSolarGain, heatRateKeysOrderedSolarGain,
                                heatRateEdgesOrderedNocturnalLoss, heatRateKeysOrderedNocturnalLoss,
                                heatRateEdgesOrderedConvection, heatRateKeysOrderedConvection,
                                heatRateEdgesOrderedConduction, heatRateKeysOrderedConduction,
                                heatRateEdgesOrderedRadiation, heatRateKeysOrderedRadiation,
                                heatRateEdgesOrderedCapacity, heatRateKeysOrderedCapacity);
    return heatRateKeysOrderedAdvection;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesAdvection() const {
    (void)getHeatRateKeysAdvection();
    std::vector<double> values(heatRateEdgesOrderedAdvection.size());
    for (size_t i = 0; i < heatRateEdgesOrderedAdvection.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedAdvection[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysHeatGeneration() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedHeatGeneration;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesHeatGeneration() const {
    (void)getHeatRateKeysHeatGeneration();
    std::vector<double> values(heatRateEdgesOrderedHeatGeneration.size());
    for (size_t i = 0; i < heatRateEdgesOrderedHeatGeneration.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedHeatGeneration[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysSolarGain() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedSolarGain;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesSolarGain() const {
    (void)getHeatRateKeysSolarGain();
    std::vector<double> values(heatRateEdgesOrderedSolarGain.size());
    for (size_t i = 0; i < heatRateEdgesOrderedSolarGain.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedSolarGain[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysNocturnalLoss() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedNocturnalLoss;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesNocturnalLoss() const {
    (void)getHeatRateKeysNocturnalLoss();
    std::vector<double> values(heatRateEdgesOrderedNocturnalLoss.size());
    for (size_t i = 0; i < heatRateEdgesOrderedNocturnalLoss.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedNocturnalLoss[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysConvection() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedConvection;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesConvection() const {
    (void)getHeatRateKeysConvection();
    std::vector<double> values(heatRateEdgesOrderedConvection.size());
    for (size_t i = 0; i < heatRateEdgesOrderedConvection.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedConvection[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysConduction() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedConduction;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesConduction() const {
    (void)getHeatRateKeysConduction();
    std::vector<double> values(heatRateEdgesOrderedConduction.size());
    for (size_t i = 0; i < heatRateEdgesOrderedConduction.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedConduction[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysRadiation() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedRadiation;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesRadiation() const {
    (void)getHeatRateKeysRadiation();
    std::vector<double> values(heatRateEdgesOrderedRadiation.size());
    for (size_t i = 0; i < heatRateEdgesOrderedRadiation.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedRadiation[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysCapacity() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedCapacity;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesCapacity() const {
    (void)getHeatRateKeysCapacity();
    std::vector<double> values(heatRateEdgesOrderedCapacity.size());
    for (size_t i = 0; i < heatRateEdgesOrderedCapacity.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedCapacity[i]].heat_rate;
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