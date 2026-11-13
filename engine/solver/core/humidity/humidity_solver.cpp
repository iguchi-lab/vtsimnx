#include "core/humidity/humidity_solver.h"

#include "core/humidity/humidity_coupling.h"
#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "simulation_metrics.h"
#include "aircon/aircon_humidity_control.h"

#include <boost/range/iterator_range.hpp>
#include <vector>

namespace core::humidity {

namespace {
// Solve two boundary responses of the existing water-balance system. This
// includes storage, generation, outdoor air, material transport and remote
// set rooms; changing a room RH result directly would lose that balance.
void updateHumidistatRequests(Graph& graph, const HumidityNetworkTerms& terms,
                              double dt, double tolerance,
                              const HumiditySolverContext& current) {
    bool active = false;
    for (auto v : boost::make_iterator_range(boost::vertices(graph)))
        active = active || (graph[v].type == "aircon" && aircon::humidity_control::enabled(graph[v]));
    if (!active) return;
    std::unordered_map<std::string, Vertex> byKey;
    for (auto v : boost::make_iterator_range(boost::vertices(graph)))
        byKey.emplace(graph[v].key, v);
    for (auto ac : boost::make_iterator_range(boost::vertices(graph))) {
        auto& node = graph[ac];
        if (node.type != "aircon" || !aircon::humidity_control::enabled(node)) continue;
        const auto set = byKey.at(node.set_node);
        const auto inlet = byKey.at(node.in_node);
        const double targetT = aircon::humidity_control::simultaneous(node)
            ? node.current_requested_pre_temp : graph[set].current_t;
        const double targetX = archenv::absolute_humidity(targetT, node.current_pre_rh);
        node.humidity_demand = graph[set].current_x > targetX + 1e-8;
        node.humidity_supply_request = targetX;
        if (!(terms.outSum[ac] > 1e-9)) continue; // restart fan before probing
        HumidityNetworkTerms probeTerms = terms;
        auto& vertices = probeTerms.updateVertices;
        vertices.erase(std::remove(vertices.begin(), vertices.end(), ac), vertices.end());
        HumiditySolverContext probe;
        probe.xN = current.xN;
        probe.xIterate = current.xIterate;
        probe.xSolved = current.xSolved;
        probe.xIterate[ac] = 0.0;
        auto first = solveHumidityImplicitStep(graph, probeTerms, dt, tolerance, probe);
        if (!first.converged) throw std::runtime_error("Humidistat water-balance probe failed: " + node.key);
        const double baseSet = probe.xSolved[set], baseIn = probe.xSolved[inlet];
        constexpr double delta = 0.01;
        probe.xIterate[ac] = delta;
        auto second = solveHumidityImplicitStep(graph, probeTerms, dt, tolerance, probe);
        if (!second.converged) throw std::runtime_error("Humidistat water-balance response failed: " + node.key);
        const double gain = (probe.xSolved[set] - baseSet) / delta;
        const double inletGain = (probe.xSolved[inlet] - baseIn) / delta;
        if (!(gain > 1e-10)) continue; // disconnected target: report unmet RH
        const double request = (targetX - baseSet) / gain;
        const double noRemoval = inletGain < 1.0 - 1e-10
            ? baseIn / (1.0 - inletGain) : graph[inlet].current_x;
        node.humidity_supply_request = std::max(0.0, request);
        node.humidity_demand = request < noRemoval - 1e-8;
    }
}
} // namespace

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
    updateHumidistatRequests(tGraph, terms, dt, constants.humiditySolverTolerance, ctx);

    MoistureBalanceTerms bal;
    evaluateMoistureBalanceTerms(tGraph, terms, ctx.xN, dt, bal);
    humidityNetwork.setLastMoistureBalance(std::move(bal));
    stats.maxMoistureBalanceResidual = humidityNetwork.lastMoistureBalance().maxAbsResidual;

    stats.updated = true;
    return stats;
}

} // namespace core::humidity
