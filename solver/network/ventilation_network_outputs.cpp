#include "network/ventilation_network.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/iterator_range.hpp>

const std::vector<std::string>& VentilationNetwork::getOutputKeys() const {
    return getPressureKeys();
}

std::vector<double> VentilationNetwork::collectOutputValues() const {
    return collectPressureValues();
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

