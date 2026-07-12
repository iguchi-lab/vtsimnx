#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "core/humidity/humidity_coupling.h"
#include "types/common_types.h"

#include <algorithm>

#include <boost/range/iterator_range.hpp>

HumidityNetwork::HumidityNetwork()
    : humiditySolverContext_(std::make_unique<core::humidity::HumiditySolverContext>()) {}

HumidityNetwork::~HumidityNetwork() = default;
HumidityNetwork::HumidityNetwork(HumidityNetwork&&) noexcept = default;
HumidityNetwork& HumidityNetwork::operator=(HumidityNetwork&&) noexcept = default;

core::humidity::HumiditySolverContext& HumidityNetwork::humiditySolverContext() {
    return *humiditySolverContext_;
}

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
    terms.ventNeighbors.assign(nV, {});
    terms.updateVertices.clear();
    terms.updateVertices.reserve(nV / 4 + 1);

    auto idxOf = [](Vertex v) { return static_cast<size_t>(v); };

    auto addVentNeighbor = [&](Vertex a, Vertex b) {
        if (a == b) return;
        auto& na = terms.ventNeighbors[idxOf(a)];
        if (std::find(na.begin(), na.end(), b) == na.end()) {
            na.push_back(b);
        }
        auto& nb = terms.ventNeighbors[idxOf(b)];
        if (std::find(nb.begin(), nb.end(), a) == nb.end()) {
            nb.push_back(a);
        }
    };

    // 生成項（発湿）: 換気ブランチの humidity_generation を target 側へ集計
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        if (!ep.current_enabled) continue;
        const double g = ep.current_humidity_generation;
        if (g == 0.0) continue;
        auto itT = tKeyToV.find(ep.target);
        if (itT == tKeyToV.end()) continue;
        terms.genByVertex[itT->second] += g;
    }

    // 換気枝から inflow/outflow を構築し、有効枝は流量ゼロでも無向隣接を登録
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        if (!ep.current_enabled) continue;

        const Vertex vSv = boost::source(e, vGraph);
        const Vertex vTv = boost::target(e, vGraph);
        const std::string& kS = vGraph[vSv].key;
        const std::string& kT = vGraph[vTv].key;

        auto itTS = tKeyToV.find(kS);
        auto itTT = tKeyToV.find(kT);
        if (itTS == tKeyToV.end() || itTT == tKeyToV.end()) continue;

        addVentNeighbor(itTS->second, itTT->second);

        const double f = ep.flow_rate; // [m3/s]
        if (f == 0.0) continue;

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
        if (!ep.current_enabled) continue;
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
    const Graph* ptr = &nodeState.graph;
    const size_t nV = static_cast<size_t>(boost::num_vertices(nodeState.graph));
    const size_t nE = static_cast<size_t>(boost::num_edges(nodeState.graph));
    if (nodeIndexInitialized &&
        cachedGraphPtr_ == ptr &&
        cachedNumVertices_ == nV &&
        cachedNumEdges_ == nE &&
        cachedTopologyRevision_ == nodeState.topologyRevision) {
        return;
    }

    nodeKeyToVertex.clear();
    const auto& graph = nodeState.graph;
    nodeKeyToVertex.reserve(nV);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        nodeKeyToVertex.emplace(graph[v].key, v);
    }
    cachedGraphPtr_ = ptr;
    cachedNumVertices_ = nV;
    cachedNumEdges_ = nE;
    cachedTopologyRevision_ = nodeState.topologyRevision;
    nodeIndexInitialized = true;
    // グラフが変わった場合は出力キャッシュも無効化
    outputCacheInitialized = false;
    outputVerticesOrdered.clear();
    outputKeysOrdered.clear();
}
