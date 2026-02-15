#include "transport/concentration_solver.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include <boost/range/iterator_range.hpp>

namespace transport {

void updateConcentrationIfEnabled(const SimulationConstants& constants,
                                  VentilationNetwork& ventNetwork,
                                  ThermalNetwork& thermalNetwork,
                                  std::ostream& logs,
                                  TimingList& timings,
                                  const std::string& meta) {
    (void)logs;
    if (!constants.concentrationCalc) return;

    ScopedTimer timer(timings, "concentration_update", meta);

    auto& tGraph = thermalNetwork.getGraph();
    auto& vGraph = ventNetwork.getGraph();
    const auto& tKeyToV = thermalNetwork.getKeyToVertex();
    const auto& vKeyToV = ventNetwork.getKeyToVertex();

    const double dt = static_cast<double>(constants.timestep);
    if (!(dt > 0.0)) return;

    auto idxOf = [](Vertex v) -> size_t { return static_cast<size_t>(v); };
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));

    // c(t) の前値（全ノード）
    std::vector<double> cOld(nV, 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        cOld[idxOf(v)] = tGraph[v].current_c;
    }

    // 生成項（発塵）: 換気ブランチの dust_generation を target 側へ集計（単位: [個/s] を想定）
    std::unordered_map<Vertex, double> genByVertex;
    genByVertex.reserve(boost::num_vertices(tGraph) / 4 + 1);
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        const double g = ep.current_dust_generation;
        if (g == 0.0) continue;
        auto itT = tKeyToV.find(ep.target);
        if (itT == tKeyToV.end()) continue;
        genByVertex[itT->second] += g;
    }

    // 流量（m3/s）から inflow/outflow を構築（eta は inflow のみに適用）
    std::vector<double> outSum(nV, 0.0); // Σ(outflow) [m3/s]
    std::vector<std::vector<std::pair<Vertex, double>>> inflowCoeff(nV); // {srcVertex, q*(1-eta)}
    inflowCoeff.reserve(nV);

    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const Vertex vSv = boost::source(e, vGraph);
        const Vertex vTv = boost::target(e, vGraph);
        const auto& ep = vGraph[e];

        const double qSigned = ep.flow_rate; // edge direction に沿って正
        if (qSigned == 0.0) continue;
        const double eta = ep.eta; // 未設定は0

        // thermal 側の頂点へマッピング
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

        outSum[idxOf(src)] += q;
        inflowCoeff[idxOf(dst)].push_back({src, q * (1.0 - eta)});
    }

    // 更新対象（calc_c=true）の頂点を key でソートして決定性を確保
    std::vector<Vertex> updateVertices;
    updateVertices.reserve(nV / 4 + 1);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        if (tGraph[v].calc_c) updateVertices.push_back(v);
    }
    std::sort(updateVertices.begin(), updateVertices.end(), [&](Vertex a, Vertex b) {
        return tGraph[a].key < tGraph[b].key;
    });

    std::vector<double> cNew = cOld;
    for (Vertex v : updateVertices) {
        const size_t i = idxOf(v);
        const double V = tGraph[v].v; // [m3]
        if (!(V > 0.0)) {
            // v<=0 は安全側: 流入混合のみ（沈着・発生は無視）
            double sumIn = 0.0;
            double sumInC = 0.0;
            for (const auto& in : inflowCoeff[i]) {
                const Vertex sv = in.first;
                const double qEff = in.second; // q*(1-eta)
                sumIn += qEff;
                sumInC += qEff * cOld[idxOf(sv)];
            }
            if (sumIn > 0.0) cNew[i] = sumInC / sumIn;
            continue;
        }

        const double preC = cOld[i];
        const double beta = tGraph[v].current_beta; // [1/s]
        const double m = [&]() -> double {
            auto itG = genByVertex.find(v);
            return (itG == genByVertex.end()) ? 0.0 : itG->second;
        }(); // [個/s]

        // old_vtsim の k1/k2 定義（k1: [個/m3/s], k2: [1/s]）
        double k1 = m / V;
        double k2 = beta;
        k2 += outSum[i] / V;
        for (const auto& in : inflowCoeff[i]) {
            const Vertex sv = in.first;
            const double qEff = in.second; // q*(1-eta)
            k1 += (qEff / V) * cOld[idxOf(sv)];
        }

        if (k2 == 0.0) {
            cNew[i] = preC + k1 * dt;
        } else {
            const double k = k1 / k2;
            cNew[i] = (preC - k) * std::exp(-k2 * dt) + k;
        }
    }

    // graph へ反映（thermal/vent 両方に入れておく）
    for (Vertex v : updateVertices) {
        const size_t i = idxOf(v);
        tGraph[v].current_c = cNew[i];
        auto itV = vKeyToV.find(tGraph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_c = cNew[i];
        }
    }
}

} // namespace transport


