#include "core/humidity/humidity_solver.h"

#include "core/humidity/humidity_coupling.h"
#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "simulation_metrics.h"

#include <boost/range/iterator_range.hpp>
#include <vector>

namespace core::humidity {

HumiditySolveStats updateHumidityIfEnabled(const SimulationConstants& constants,
                                           VentilationNetwork& ventNetwork,
                                           Graph& nodeGraph,
                                           ConstNodeStateView nodeState,
                                           HumidityNetwork& humidityNetwork,
                                           const FlowRateMap& flowRates,
                                           std::ostream& logs,
                                           TimingList& timings,
                                           const std::string& meta,
                                           const std::vector<double>* xN,
                                           simulation::TimestepSolveMetrics* metrics) {
    (void)logs;
    HumiditySolveStats stats{};
    if (!constants.humidityCalc) return stats;

    ScopedTimer timer(timings, "humidity_update", meta);

    auto& tGraph = nodeGraph;
    auto& vGraph = ventNetwork.getGraph();
    const auto& vKeyToV = ventNetwork.getKeyToVertex();

    const double dt = static_cast<double>(constants.timestep);
    if (!(dt > 0.0)) return stats;

    (void)flowRates; // エッジ直接走査方式に統一したため FlowRateMap は不使用
    HumidityNetworkTerms terms;
    humidityNetwork.buildTerms(nodeState, ventNetwork, terms);
    stats.activeVertices = static_cast<int>(terms.updateVertices.size());
    if (stats.activeVertices == 0) return stats;

    auto& ctx = humidityNetwork.humiditySolverContext();
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    ctx.xIterate.resize(nV);
    ctx.xN.resize(nV);
    ctx.xSolved.resize(nV);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        const size_t i = static_cast<size_t>(v);
        const double xk = tGraph[v].current_x;
        ctx.xIterate[i] = xk;
        ctx.xSolved[i] = xk;
        if (xN != nullptr && i < xN->size()) {
            ctx.xN[i] = (*xN)[i];
        } else {
            ctx.xN[i] = xk;
        }
    }

    const SolveStats solve = solveHumidityImplicitStep(
        tGraph,
        terms,
        dt,
        constants.humiditySolverTolerance,
        ctx);
    stats.iterations = solve.iterations;
    stats.finalRelativeResidual = solve.finalRelativeResidual;
    stats.converged = solve.converged;
    stats.patternAnalyzes = solve.patternAnalyzes;
    stats.factorizes = solve.factorizes;
    stats.rhsOnlySolves = solve.rhsOnlySolves;
    stats.solutionReuse = solve.solutionReuse;

    if (metrics) {
        metrics->humidityPatternAnalyzes += solve.patternAnalyzes;
        metrics->humidityFactorizes += solve.factorizes;
        metrics->humidityRhsOnlySolves += solve.rhsOnlySolves;
        metrics->humiditySolutionReuse += solve.solutionReuse;
    }

    // 未収束・非有限・非物理解はグラフへ反映しない（連成の偽収束を防ぐ）
    if (!solve.converged) {
        stats.updated = false;
        return stats;
    }
    applyHumidityStateToGraphs(tGraph, vGraph, vKeyToV, terms.updateVertices, ctx.xSolved);
    stats.updated = true;
    return stats;
}

} // namespace core::humidity
