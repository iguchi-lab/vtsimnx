#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "types/common_types.h"

#include <algorithm>

#include <boost/range/iterator_range.hpp>

void HumidityNetwork::buildTerms(ConstNodeStateView nodeState,
                                 const VentilationNetwork& ventNetwork,
                                 HumidityNetworkTerms& terms) const {
    ensureNodeIndex(nodeState);
    const auto& tGraph = nodeState.graph;
    const auto& vGraph = ventNetwork.getGraph();
    const auto& tKeyToV = nodeKeyToVertex;
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));

    terms.genByVertex.clear();
    terms.genByVertex.reserve(boost::num_vertices(tGraph) / 4 + 1);
    terms.outSum.assign(nV, 0.0);
    terms.inflow.assign(nV, {});
    terms.moistureLinks.assign(nV, {});
    terms.updateVertices.clear();
    terms.updateVertices.reserve(nV / 4 + 1);

    auto idxOf = [](Vertex v) { return static_cast<size_t>(v); };

    // 生成項（発湿）: 換気ブランチの humidity_generation を target 側へ集計
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        const double g = ep.current_humidity_generation;
        if (g == 0.0) continue;
        auto itT = tKeyToV.find(ep.target);
        if (itT == tKeyToV.end()) continue;
        terms.genByVertex[itT->second] += g;
    }

    // 換気枝から inflow/outflow を構築
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        const double f = ep.flow_rate; // [m3/s]
        if (f == 0.0) continue;

        const Vertex vSv = boost::source(e, vGraph);
        const Vertex vTv = boost::target(e, vGraph);
        const std::string& kS = vGraph[vSv].key;
        const std::string& kT = vGraph[vTv].key;

        auto itTS = tKeyToV.find(kS);
        auto itTT = tKeyToV.find(kT);
        if (itTS == tKeyToV.end() || itTT == tKeyToV.end()) continue;

        Vertex src = itTS->second;
        Vertex dst = itTT->second;
        double mDot = f * PhysicalConstants::DENSITY_DRY_AIR; // [kg/s]
        if (mDot < 0.0) {
            mDot = -mDot;
            std::swap(src, dst);
        }

        terms.outSum[idxOf(src)] += mDot;
        terms.inflow[idxOf(dst)].push_back({src, mDot});
    }

    // 湿気回路網（双方向）
    for (auto e : boost::make_iterator_range(boost::edges(tGraph))) {
        const auto& ep = tGraph[e];
        const double k = ep.moisture_conductance;
        if (!(k > 0.0)) continue;
        const Vertex sv = boost::source(e, tGraph);
        const Vertex tv = boost::target(e, tGraph);
        terms.moistureLinks[idxOf(sv)].push_back({tv, k});
        terms.moistureLinks[idxOf(tv)].push_back({sv, k});
    }

    // 更新対象を決定
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        if (tGraph[v].calc_x) terms.updateVertices.push_back(v);
    }
    std::sort(terms.updateVertices.begin(), terms.updateVertices.end(), [&](Vertex a, Vertex b) {
        return tGraph[a].key < tGraph[b].key;
    });
}

void HumidityNetwork::ensureNodeIndex(ConstNodeStateView nodeState) const {
    if (nodeIndexInitialized) return;
    nodeKeyToVertex.clear();
    const auto& graph = nodeState.graph;
    nodeKeyToVertex.reserve(boost::num_vertices(graph));
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        nodeKeyToVertex.emplace(graph[v].key, v);
    }
    nodeIndexInitialized = true;
}

