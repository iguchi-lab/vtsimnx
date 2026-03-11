#include "transport/concentration_solver.h"

#include "network/contaminant_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <cmath>
#include <vector>

#include <boost/range/iterator_range.hpp>

namespace transport {

void updateConcentrationIfEnabled(const SimulationConstants& constants,
                                  VentilationNetwork& ventNetwork,
                                  ThermalNetwork& thermalNetwork,
                                  ContaminantNetwork& contaminantNetwork,
                                  std::ostream& logs,
                                  TimingList& timings,
                                  const std::string& meta) {
    (void)logs;
    if (!constants.concentrationCalc) return;

    ScopedTimer timer(timings, "concentration_update", meta);

    auto& tGraph = thermalNetwork.getGraph();
    auto& vGraph = ventNetwork.getGraph();
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

    ContaminantNetworkTerms terms;
    contaminantNetwork.buildTerms(tGraph, thermalNetwork, ventNetwork, terms);

    std::vector<double> cNew = cOld;
    for (Vertex v : terms.updateVertices) {
        const size_t i = idxOf(v);
        const double V = tGraph[v].v; // [m3]
        if (!(V > 0.0)) {
            // v<=0 は安全側: 流入混合のみ（沈着・発生は無視）
            double sumIn = 0.0;
            double sumInC = 0.0;
            for (const auto& in : terms.inflowCoeff[i]) {
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
            auto itG = terms.genByVertex.find(v);
            return (itG == terms.genByVertex.end()) ? 0.0 : itG->second;
        }(); // [個/s]

        // old_vtsim の k1/k2 定義（k1: [個/m3/s], k2: [1/s]）
        double k1 = m / V;
        double k2 = beta;
        k2 += terms.outSum[i] / V;
        for (const auto& in : terms.inflowCoeff[i]) {
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
    for (Vertex v : terms.updateVertices) {
        const size_t i = idxOf(v);
        tGraph[v].current_c = cNew[i];
        auto itV = vKeyToV.find(tGraph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_c = cNew[i];
        }
    }
}

} // namespace transport


