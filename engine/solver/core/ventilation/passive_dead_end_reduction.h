#pragma once

#include "core/ventilation/flow_calculation.h"
#include "core/ventilation/pressure_solver.h"

namespace ventilation {

// A pressure-unknown leaf with one passive connection has exactly zero flow.
// Eliminate such branches before the nonlinear solve, then back-substitute
// their hydrostatic pressures. This avoids fitting a near-zero square root
// flow at a stopped fan, without changing the physical acceptance tolerance.
class PassiveDeadEndReduction {
    struct Leaf {
        Vertex vertex;
        Vertex parent;
        Edge edge;
    };
    Graph& graph_;
    std::vector<Leaf> leaves_;
    bool restored_ = false;

public:
    explicit PassiveDeadEndReduction(Graph& graph) : graph_(graph) {
        std::vector<Edge> activeEdges;
        std::vector<std::vector<std::size_t>> incident(boost::num_vertices(graph));
        std::vector<std::size_t> degree(incident.size(), 0);
        std::vector<Vertex> pending;
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const auto& edge = graph[e];
            if (!edge.current_enabled ||
                (FlowCalculation::isFixedVolumeFlowEdge(edge) && edge.current_vol == 0.0) ||
                (edge.type == "fan" && edge.q_max == 0.0 && edge.q1 == 0.0)) continue;
            const auto index = activeEdges.size();
            activeEdges.push_back(e);
            for (auto v : {boost::source(e, graph), boost::target(e, graph)}) {
                incident[v].push_back(index);
                ++degree[v];
            }
        }
        std::vector<bool> removed(activeEdges.size(), false);
        for (auto v : boost::make_iterator_range(boost::vertices(graph)))
            if (degree[v] == 1) pending.push_back(v);
        for (std::size_t i = 0; i < pending.size(); ++i) {
            const auto v = pending[i];
            if (!graph[v].calc_p || degree[v] != 1) continue;
            for (auto index : incident[v]) {
                if (removed[index]) continue;
                const auto e = activeEdges[index];
                const auto& edge = graph[e];
                if (FlowCalculation::isFixedVolumeFlowEdge(edge) ||
                    (edge.type != "simple_opening" && edge.type != "gap" &&
                     edge.type != "pressure_loss")) continue;
                const auto source = boost::source(e, graph);
                const auto target = boost::target(e, graph);
                leaves_.push_back({v, source == v ? target : source, e});
                removed[index] = true;
                for (auto endpoint : {source, target})
                    if (--degree[endpoint] == 1) pending.push_back(endpoint);
            }
        }
        // Allocate the reduction first; no partial mutation if allocation fails.
        for (const auto& leaf : leaves_) {
            graph[leaf.vertex].calc_p = false;
            graph[leaf.edge].current_enabled = false;
        }
    }

    PassiveDeadEndReduction(const PassiveDeadEndReduction&) = delete;
    PassiveDeadEndReduction& operator=(const PassiveDeadEndReduction&) = delete;
    ~PassiveDeadEndReduction() { restore(); }

    void restore() noexcept {
        if (restored_) return;
        for (const auto& leaf : leaves_) {
            graph_[leaf.vertex].calc_p = true;
            graph_[leaf.edge].current_enabled = true;
        }
        restored_ = true;
    }

    void expand(PressureSolveResult& result, double tolerance) {
        restore();
        if (!result.accepted || leaves_.empty()) return;
        for (auto it = leaves_.rbegin(); it != leaves_.rend(); ++it) {
            const auto& edge = graph_[it->edge];
            const bool isSource = boost::source(it->edge, graph_) == it->vertex;
            const double leafHead = calculateDensity(graph_[it->vertex].current_t) *
                archenv::GRAVITY * (isSource ? edge.h_from : edge.h_to);
            const double parentHead = calculateDensity(graph_[it->parent].current_t) *
                archenv::GRAVITY * (isSource ? edge.h_to : edge.h_from);
            const double parentTotal = result.pressures.at(graph_[it->parent].key) - parentHead;
            const double p = parentTotal + leafHead;
            result.pressures[graph_[it->vertex].key] = p;
            // Only floating-point cancellation may remain in the eliminated
            // branch law. Reject nonfinite or materially inconsistent recovery.
            const double dp = (p - leafHead) - parentTotal;
            const double q = FlowCalculation::calculateUnifiedFlow(isSource ? dp : -dp, edge);
            if (!std::isfinite(p) || !std::isfinite(q) || std::abs(q) > tolerance)
                result.accepted = false;
        }
        // Reduced flows already include exact zeros for the eliminated edges.
        // Recheck ALL original pressure unknowns, including the restored leaves.
        result.metrics = computePressureUnknownBalanceMetrics(result.balances, graph_);
        result.accepted = result.accepted && acceptMassBalance(result.metrics, tolerance);
    }

    void applyExactZeroFlows() {
        for (const auto& leaf : leaves_) graph_[leaf.edge].flow_rate = 0.0;
    }
};

} // namespace ventilation
