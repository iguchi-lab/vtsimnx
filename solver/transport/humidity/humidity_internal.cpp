#include "transport/humidity/humidity_internal.h"

#include <algorithm>
#include <limits>

#include <boost/range/iterator_range.hpp>

namespace transport::humidity_internal {

namespace {
inline size_t idxOf(Vertex v) { return static_cast<size_t>(v); }
} // namespace

void buildHumidityNetworkTerms(const Graph& vGraph,
                               const Graph& tGraph,
                               const std::unordered_map<std::string, Vertex>& tKeyToV,
                               NetworkTerms& terms) {
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    terms.genByVertex.clear();
    terms.genByVertex.reserve(boost::num_vertices(tGraph) / 4 + 1);
    terms.outSum.assign(nV, 0.0);
    terms.inflow.assign(nV, {});
    terms.moistureLinks.assign(nV, {});
    terms.updateVertices.clear();
    terms.updateVertices.reserve(nV / 4 + 1);

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

void initializeHumidityState(const Graph& tGraph,
                             std::vector<double>& xOld,
                             std::vector<double>& xNew) {
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    xOld.assign(nV, 0.0);
    xNew.assign(nV, 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        const size_t i = idxOf(v);
        xOld[i] = tGraph[v].current_x;
        xNew[i] = tGraph[v].current_x;
    }
}

void solveHumidityImplicitStep(const Graph& tGraph,
                               const NetworkTerms& terms,
                               double dt,
                               std::vector<double>& xNew,
                               const std::vector<double>& xOld) {
    constexpr double rho = PhysicalConstants::DENSITY_DRY_AIR; // [kg/m3]
    const int maxIter = 80;
    const double tol = 1e-9;

    for (int it = 0; it < maxIter; ++it) {
        double maxDiff = 0.0;
        for (Vertex v : terms.updateVertices) {
            const size_t i = idxOf(v);
            const double V = tGraph[v].v; // [m3]
            const double cap = (tGraph[v].moisture_capacity > 0.0)
                                   ? tGraph[v].moisture_capacity
                                   : (rho * V);
            const auto itG = terms.genByVertex.find(v);
            const double g = (itG == terms.genByVertex.end()) ? 0.0 : itG->second;

            // 容量なしノードは流入混合のみ
            if (!(cap > 0.0)) {
                double sumIn = 0.0;
                double sumInX = 0.0;
                for (const auto& in : terms.inflow[i]) {
                    const Vertex sv = in.first;
                    const double md = in.second;
                    sumIn += md;
                    sumInX += md * xNew[idxOf(sv)];
                }
                double x = xNew[i];
                if (sumIn > 0.0) x = sumInX / sumIn;
                maxDiff = std::max(maxDiff, std::abs(x - xNew[i]));
                xNew[i] = x;
                continue;
            }

            const double out = terms.outSum[i]; // [kg/s]
            double denom = 1.0 + dt * out / cap;

            double rhs = xOld[i];
            rhs += dt * (g / cap);
            for (const auto& in : terms.inflow[i]) {
                const Vertex sv = in.first;
                const double md = in.second;
                rhs += dt * (md / cap) * xNew[idxOf(sv)];
            }
            for (const auto& lk : terms.moistureLinks[i]) {
                const Vertex ov = lk.first;
                const double k = lk.second;
                denom += dt * (k / cap);
                rhs += dt * (k / cap) * xNew[idxOf(ov)];
            }

            const double x = rhs / denom;
            maxDiff = std::max(maxDiff, std::abs(x - xNew[i]));
            xNew[i] = x;
        }
        if (maxDiff < tol) break;
    }
}

void applyHumidityStateToGraphs(Graph& tGraph,
                                Graph& vGraph,
                                const std::unordered_map<std::string, Vertex>& vKeyToV,
                                const std::vector<Vertex>& updateVertices,
                                const std::vector<double>& xNew) {
    for (Vertex v : updateVertices) {
        const size_t i = idxOf(v);
        tGraph[v].current_x = xNew[i];
        tGraph[v].current_w = xNew[i];
        auto itV = vKeyToV.find(tGraph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_x = xNew[i];
        }
    }
}

} // namespace transport::humidity_internal

