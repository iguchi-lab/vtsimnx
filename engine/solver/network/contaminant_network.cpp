#include "network/contaminant_network.h"

#include "network/ventilation_network.h"

#include <algorithm>

#include <boost/range/iterator_range.hpp>

void ContaminantNetwork::buildTerms(ConstNodeStateView nodeState,
                                    const VentilationNetwork& ventNetwork,
                                    ContaminantNetworkTerms& terms) const {
    ensureNodeIndex(nodeState);
    const auto& tGraph = nodeState.graph;
    const auto& vGraph = ventNetwork.getGraph();
    const auto& tKeyToV = nodeKeyToVertex;

    const auto idxOf = [](Vertex v) -> size_t { return static_cast<size_t>(v); };
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));

    // 生成項（発塵）: 換気ブランチの dust_generation を target 側へ集計（単位: [個/s] を想定）
    terms.genByVertex.clear();
    terms.genByVertex.reserve(boost::num_vertices(tGraph) / 4 + 1);
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        const double g = ep.current_dust_generation;
        if (g == 0.0) continue;
        auto itT = tKeyToV.find(ep.target);
        if (itT == tKeyToV.end()) continue;
        terms.genByVertex[itT->second] += g;
    }

    // 流量（m3/s）から inflow/outflow を構築（eta は inflow のみに適用）
    terms.outSum.assign(nV, 0.0); // Σ(outflow) [m3/s]
    terms.inflow.assign(nV, {});
    terms.inflow.reserve(nV);

    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const Vertex vSv = boost::source(e, vGraph);
        const Vertex vTv = boost::target(e, vGraph);
        const auto& ep = vGraph[e];

        const double qSigned = ep.flow_rate; // edge direction に沿って正
        if (qSigned == 0.0) continue;
        const double eta = ep.eta; // 未設定は0

        // nodeGraph 側の頂点へマッピング
        const std::string& kS = vGraph[vSv].key;
        const std::string& kT = vGraph[vTv].key;
        auto itTS = tKeyToV.find(kS);
        auto itTT = tKeyToV.find(kT);
        if (itTS == tKeyToV.end() || itTT == tKeyToV.end()) continue;
        Vertex tS = itTS->second;
        Vertex tT = itTT->second;

        Vertex src = tS;
        Vertex dst = tT;
        double q = qSigned;
        if (q < 0.0) {
            q = -q;
            src = tT;
            dst = tS;
        }

        terms.outSum[idxOf(src)] += q;
        terms.inflow[idxOf(dst)].push_back({src, q * (1.0 - eta)});
    }

    // 更新対象（calc_c=true）の頂点を key でソートして決定性を確保
    terms.updateVertices.clear();
    terms.updateVertices.reserve(nV / 4 + 1);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        if (tGraph[v].calc_c) terms.updateVertices.push_back(v);
    }
    std::sort(terms.updateVertices.begin(), terms.updateVertices.end(), [&](Vertex a, Vertex b) {
        return tGraph[a].key < tGraph[b].key;
    });
}

void ContaminantNetwork::ensureNodeIndex(ConstNodeStateView nodeState) const {
    if (nodeIndexInitialized) return;
    nodeKeyToVertex.clear();
    const auto& graph = nodeState.graph;
    nodeKeyToVertex.reserve(boost::num_vertices(graph));
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        nodeKeyToVertex.emplace(graph[v].key, v);
    }
    nodeIndexInitialized = true;
}

