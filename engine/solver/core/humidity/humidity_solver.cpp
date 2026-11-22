#include "core/humidity/humidity_solver.h"

#include "core/humidity/humidity_coupling.h"
#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "simulation_metrics.h"
#include "simulation_error.h"
#include <Eigen/QR>
#include "aircon/aircon_humidity_control.h"
#include "archenv/include/archenv.h"
#include "types/common_types.h"

#include <boost/range/iterator_range.hpp>
#include <unordered_map>
#include <cmath>
#include <vector>

namespace core::humidity {

namespace {
constexpr double kIdealHumidityTol = 1e-9;

void clearIdealRoomLoadLatent(Graph& graph) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        auto& n = graph[v];
        if (n.getTypeCode() != VertexProperties::TypeCode::Aircon) continue;
        if (!aircon::humidity_control::idealRoomLoad(n)) continue;
        n.required_moisture_kg_s = 0.0;
        n.required_latent_w = 0.0;
    }
}

std::unordered_map<Vertex, double> idealRoomLoadHumidityFixes(const Graph& graph) {
    std::unordered_map<std::string, Vertex> byKey;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        byKey.emplace(graph[v].key, v);
    }
    std::unordered_map<Vertex, double> fixes;
    for (auto ac : boost::make_iterator_range(boost::vertices(graph))) {
        const auto& node = graph[ac];
        if (!aircon::humidity_control::idealRoomLoadHumidityActive(node)) continue;
        auto it = byKey.find(node.set_node);
        if (it == byKey.end()) continue;
        const Vertex room = it->second;
        const double targetX = archenv::absolute_humidity(
            node.current_pre_temp, node.current_pre_rh);
        if (!(targetX >= 0.0) || !std::isfinite(targetX)) continue;
        // Dehumidify only: fix when free-floating humidity exceeds the setpoint.
        if (graph[room].current_x > targetX + kIdealHumidityTol) {
            fixes[room] = targetX;
        }
    }
    return fixes;
}

void assignIdealRoomLoadLatent(Graph& graph, const HumidityNetworkTerms& terms,
                               const std::vector<double>& xN, double dt,
                               MoistureBalanceTerms& bal) {
    clearIdealRoomLoadLatent(graph);
    std::unordered_map<std::string, Vertex> byKey;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        byKey.emplace(graph[v].key, v);
    }
    for (auto ac : boost::make_iterator_range(boost::vertices(graph))) {
        auto& node = graph[ac];
        if (!aircon::humidity_control::idealRoomLoadHumidityActive(node)) continue;
        auto it = byKey.find(node.set_node);
        if (it == byKey.end()) continue;
        const size_t i = static_cast<size_t>(it->second);
        if (i >= bal.residual.size()) continue;
        // residual = S_ac [kg/s] into air; removal is positive when S_ac < 0.
        const double removal = std::max(0.0, -bal.residual[i]);
        node.required_moisture_kg_s = removal;
        node.required_latent_w = removal * PhysicalConstants::LATENT_HEAT_VAPORIZATION;
        bal.residual[i] = 0.0;
    }
    bal.maxAbsResidual = 0.0;
    for (Vertex v : terms.updateVertices) {
        const size_t i = static_cast<size_t>(v);
        if (i < bal.residual.size()) {
            bal.maxAbsResidual = std::max(bal.maxAbsResidual, std::abs(bal.residual[i]));
        }
    }
}

// Probe all controlled supply boundaries together. One factorization serves the
// baseline and every response column; the small AC response system accounts for
// cross-room interference before the outer physical constraint iteration.
void updateHumidistatRequests(Graph& graph, const HumidityNetworkTerms& terms,
                              double dt, double tolerance,
                              const HumiditySolverContext& current,
                              HumiditySolverContext& probe,
                              simulation::TimestepSolveMetrics* metrics) {
    bool active = false;
    for (auto v : boost::make_iterator_range(boost::vertices(graph)))
        active = active || (graph[v].type == "aircon" && aircon::humidity_control::enabled(graph[v]));
    if (!active) return;
    std::unordered_map<std::string, Vertex> byKey;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) byKey.emplace(graph[v].key, v);
    std::vector<Vertex> units, rooms, inlets;
    std::vector<double> targets;
    for (auto ac : boost::make_iterator_range(boost::vertices(graph))) {
        auto& node = graph[ac];
        if (node.type != "aircon" || !aircon::humidity_control::enabled(node)) continue;
        const auto room = byKey.at(node.set_node);
        const double targetT = aircon::humidity_control::simultaneous(node)
            ? node.current_requested_pre_temp : graph[room].current_t;
        const double targetX = archenv::absolute_humidity(targetT, node.current_pre_rh);
        node.humidity_demand = graph[room].current_x > targetX + 1e-8;
        node.humidity_supply_request = targetX;
        if (!(terms.outSum[ac] > 1e-9)) continue; // restart fan before probing
        units.push_back(ac); rooms.push_back(room); inlets.push_back(byKey.at(node.in_node));
        targets.push_back(targetX);
    }
    const Eigen::Index n = static_cast<Eigen::Index>(units.size());
    if (n == 0) return;
    HumidityNetworkTerms probeTerms = terms;
    auto& vertices = probeTerms.updateVertices;
    vertices.erase(std::remove_if(vertices.begin(), vertices.end(), [&](Vertex v) {
        return std::find(units.begin(), units.end(), v) != units.end();
    }), vertices.end());
    probe.xN = current.xN;
    probe.xIterate = current.xIterate;
    probe.xSolved = current.xSolved;
    Eigen::VectorXd base(n), baseSet(n), baseIn(n), target(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        // A hypothetical zero boundary can violate prescribed moisture sinks.
        base[i] = std::max({0.0, graph[units[i]].current_x, graph[inlets[i]].current_x});
        probe.xIterate[units[i]] = base[i];
        target[i] = targets[i];
    }
    auto solveProbe = [&]() {
        const auto stats = solveHumidityImplicitStep(graph, probeTerms, dt, tolerance, probe);
        if (metrics) {
            metrics->humidityPatternAnalyzes += stats.patternAnalyzes;
            metrics->humidityFactorizes += stats.factorizes;
            metrics->humidityRhsOnlySolves += stats.rhsOnlySolves;
            metrics->humiditySolutionReuse += stats.solutionReuse;
        }
        if (!stats.converged) throw simulation::Error(simulation::ErrorCode::HumidityNotConverged,
            "Humidistat water-balance response failed");
    };
    solveProbe();
    for (Eigen::Index i = 0; i < n; ++i) {
        baseSet[i] = probe.xSolved[rooms[i]]; baseIn[i] = probe.xSolved[inlets[i]];
    }
    constexpr double delta = 0.01;
    Eigen::MatrixXd gains(n,n), inletGains(n,n);
    for (Eigen::Index column = 0; column < n; ++column) {
        probe.xIterate[units[column]] = base[column] + delta;
        solveProbe();
        for (Eigen::Index row = 0; row < n; ++row) {
            gains(row,column) = (probe.xSolved[rooms[row]] - baseSet[row]) / delta;
            inletGains(row,column) = (probe.xSolved[inlets[row]] - baseIn[row]) / delta;
        }
        probe.xIterate[units[column]] = base[column];
    }
    Eigen::VectorXd adjustment = gains.completeOrthogonalDecomposition().solve(target-baseSet);
    Eigen::VectorXd request = base + adjustment;
    // A constrained unit cannot provide its hypothetical target boundary.
    // Keep that actual boundary in the free units' response equations, while
    // retaining its own unconstrained request for capacity/release control.
    std::vector<bool> limited(static_cast<std::size_t>(n), false);
    for (Eigen::Index pass = 0; pass <= n; ++pass) {
        bool added = false;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (limited[i]) continue;
            const auto& node = graph[units[i]];
            const double inletX = graph[inlets[i]].current_x;
            const bool simultaneous = aircon::humidity_control::simultaneous(node);
            const bool temperatureHeld = simultaneous ||
                node.current_pre_temp >= node.current_requested_pre_temp-1e-4;
            const double upperX = std::min(inletX,
                aircon::latent::saturationHumidityUpperBound(node.current_t));
            const double lowerX = std::min(inletX,
                aircon::latent::saturationHumidityUpperBound(node.min_coil_temperature));
            const bool bound = node.aircon_control_state == AirconControlState::CapacityLimited ||
                node.humidity_capacity_limited ||
                (node.control_mode == "humidity_priority" && node.humidity_overcool_hold &&
                 node.current_pre_temp <= node.current_requested_pre_temp-node.max_overcooling+1e-4) ||
                (node.on && temperatureHeld && request[i] > upperX+1e-8) ||
                (node.on && simultaneous && request[i] < lowerX-1e-8);
            if (bound) { limited[i] = true; added = true; }
        }
        if (!added) break;
        std::vector<Eigen::Index> freeUnits;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (limited[i]) adjustment[i] = graph[units[i]].current_x-base[i];
            else { freeUnits.push_back(i); adjustment[i] = 0.0; }
        }
        if (freeUnits.empty()) break;
        const Eigen::Index m = static_cast<Eigen::Index>(freeUnits.size());
        Eigen::MatrixXd freeGains(m,m);
        Eigen::VectorXd residual = target-baseSet-gains*adjustment, freeRhs(m);
        for (Eigen::Index i = 0; i < m; ++i) {
            freeRhs[i] = residual[freeUnits[i]];
            for (Eigen::Index j = 0; j < m; ++j) freeGains(i,j) = gains(freeUnits[i],freeUnits[j]);
        }
        const Eigen::VectorXd freeAdjustment = freeGains.completeOrthogonalDecomposition().solve(freeRhs);
        for (Eigen::Index i = 0; i < m; ++i) {
            adjustment[freeUnits[i]] = freeAdjustment[i];
            request[freeUnits[i]] = base[freeUnits[i]]+freeAdjustment[i];
        }
    }
    const Eigen::VectorXd intake = baseIn + inletGains * adjustment;
    if (!request.allFinite() || !intake.allFinite()) throw simulation::Error(
        simulation::ErrorCode::HumidityNotConverged, "Nonfinite humidistat response solution");
    for (Eigen::Index i = 0; i < n; ++i) {
        if (gains.row(i).norm() <= 1e-10) continue; // disconnected target remains visibly unmet
        auto& node = graph[units[i]];
        node.humidity_supply_request = std::max(0.0, request[i]);
        node.humidity_demand = request[i] < intake[i] - 1e-8;
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

    clearIdealRoomLoadLatent(tGraph);

    auto accumulateSolve = [&](const SolveStats& solve) {
        stats.iterations += solve.iterations;
        stats.finalRelativeResidual = solve.finalRelativeResidual;
        stats.converged = solve.converged;
        stats.patternAnalyzes += solve.patternAnalyzes;
        stats.factorizes += solve.factorizes;
        stats.rhsOnlySolves += solve.rhsOnlySolves;
        stats.solutionReuse += solve.solutionReuse;
        if (metrics) {
            metrics->humidityPatternAnalyzes += solve.patternAnalyzes;
            metrics->humidityFactorizes += solve.factorizes;
            metrics->humidityRhsOnlySolves += solve.rhsOnlySolves;
            metrics->humiditySolutionReuse += solve.solutionReuse;
        }
    };

    SolveStats solve = solveHumidityImplicitStep(
        tGraph, terms, dt, constants.humiditySolverTolerance, ctx, nullptr);
    accumulateSolve(solve);
    if (!solve.converged) {
        stats.updated = false;
        return stats;
    }
    applyHumidityStateToGraphs(tGraph, vGraph, vKeyToV, terms.updateVertices, ctx.xSolved);

    const auto idealFixes = idealRoomLoadHumidityFixes(tGraph);
    if (!idealFixes.empty()) {
        // Neighbor states after the free solve become the iterate for the fixed pass.
        for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
            ctx.xIterate[static_cast<size_t>(v)] = tGraph[v].current_x;
        }
        solve = solveHumidityImplicitStep(
            tGraph, terms, dt, constants.humiditySolverTolerance, ctx, &idealFixes);
        accumulateSolve(solve);
        if (!solve.converged) {
            stats.updated = false;
            return stats;
        }
        applyHumidityStateToGraphs(tGraph, vGraph, vKeyToV, terms.updateVertices, ctx.xSolved);
    }

    updateHumidistatRequests(tGraph, terms, dt, constants.humiditySolverTolerance, ctx,
        humidityNetwork.humidistatSolverContext(), metrics);

    MoistureBalanceTerms bal;
    evaluateMoistureBalanceTerms(tGraph, terms, ctx.xN, dt, bal);
    assignIdealRoomLoadLatent(tGraph, terms, ctx.xN, dt, bal);
    humidityNetwork.setLastMoistureBalance(std::move(bal));
    stats.maxMoistureBalanceResidual = humidityNetwork.lastMoistureBalance().maxAbsResidual;

    stats.updated = true;
    return stats;
}

} // namespace core::humidity
