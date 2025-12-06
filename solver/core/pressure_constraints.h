#pragma once

#include "../vtsim_solver.h"
#include "core/flow_calculation.h"
#include <map>
#include <ostream>
#include <vector>

class FlowBalanceConstraint {
public:
    FlowBalanceConstraint(
        const std::string& nodeName,
        const Graph& graph,
        const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
        const std::map<Vertex, size_t>& vertexToParameterIndex,
        std::ostream& logFile
    ) : nodeName_(nodeName),
        graph_(graph),
        nodeKeyToVertex_(nodeKeyToVertex),
        vertexToParameterIndex_(vertexToParameterIndex),
        logFile_(logFile) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residual) const {
        auto nodeIt = nodeKeyToVertex_.find(nodeName_);
        if (nodeIt == nodeKeyToVertex_.end()) {
            residual[0] = T(0.0);
            return true;
        }

        Vertex nodeVertex = nodeIt->second;
        T inflow = T(0.0);
        T outflow = T(0.0);

        auto edge_range = boost::edges(graph_);
        for (auto edge : boost::make_iterator_range(edge_range)) {
            auto sourceVertex = boost::source(edge, graph_);
            auto targetVertex = boost::target(edge, graph_);
            const auto& edgeData = graph_[edge];

            bool isOutEdge = (sourceVertex == nodeVertex);
            bool isInEdge  = (targetVertex == nodeVertex);
            if (!isOutEdge && !isInEdge) continue;

            auto itS = vertexToParameterIndex_.find(sourceVertex);
            auto itT = vertexToParameterIndex_.find(targetVertex);
            if (itS != vertexToParameterIndex_.end() && itT != vertexToParameterIndex_.end()
                && itS->second == itT->second) {
                continue;
            }

            T sourcePressure = getNodePressure(sourceVertex, parameters);
            T targetPressure = getNodePressure(targetVertex, parameters);

            const auto& sourceNode = graph_[sourceVertex];
            const auto& targetNode = graph_[targetVertex];
            T rho_source = T(calculateDensity(sourceNode.current_t));
            T rho_target = T(calculateDensity(targetNode.current_t));

            T source_total_pressure = sourcePressure - rho_source * T(archenv::GRAVITY) * T(edgeData.h_from);
            T target_total_pressure = targetPressure - rho_target * T(archenv::GRAVITY) * T(edgeData.h_to);
            T dp = source_total_pressure - target_total_pressure;

            T flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
            if (isOutEdge) outflow += flow;
            if (isInEdge) inflow += flow;
        }

        residual[0] = inflow - outflow;
        return true;
    }

private:
    std::string nodeName_;
    const Graph& graph_;
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex_;
    const std::map<Vertex, size_t>& vertexToParameterIndex_;
    std::ostream& logFile_;

    template <typename T>
    T getNodePressure(Vertex v, T const* const* parameters) const {
        auto it = vertexToParameterIndex_.find(v);
        if (it != vertexToParameterIndex_.end()) {
            return parameters[0][it->second];
        }
        return T(graph_[v].current_p);
    }
};

class GroupFlowBalanceConstraint {
public:
    GroupFlowBalanceConstraint(
        const std::vector<Vertex>& groupVertices,
        const Graph& graph,
        const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
        const std::map<Vertex, size_t>& vertexToParameterIndex,
        std::ostream& logFile
    ) : groupVertices_(groupVertices),
        graph_(graph),
        nodeKeyToVertex_(nodeKeyToVertex),
        vertexToParameterIndex_(vertexToParameterIndex),
        logFile_(logFile) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residual) const {
        T inflow = T(0.0);
        T outflow = T(0.0);

        auto edge_range = boost::edges(graph_);
        for (auto edge : boost::make_iterator_range(edge_range)) {
            auto sv = boost::source(edge, graph_);
            auto tv = boost::target(edge, graph_);
            const auto& ep = graph_[edge];

            bool sIn = std::find(groupVertices_.begin(), groupVertices_.end(), sv) != groupVertices_.end();
            bool tIn = std::find(groupVertices_.begin(), groupVertices_.end(), tv) != groupVertices_.end();
            if (sIn == tIn) continue;

            T pS = getNodePressure(sv, parameters);
            T pT = getNodePressure(tv, parameters);

            const auto& sNode = graph_[sv];
            const auto& tNode = graph_[tv];
            T rhoS = T(calculateDensity(sNode.current_t));
            T rhoT = T(calculateDensity(tNode.current_t));

            T sTotal = pS - rhoS * T(archenv::GRAVITY) * T(ep.h_from);
            T tTotal = pT - rhoT * T(archenv::GRAVITY) * T(ep.h_to);
            T dp = sTotal - tTotal;

            T q = FlowCalculation::calculateUnifiedFlow(dp, ep);
            if (sIn) outflow += q;
            if (tIn) inflow += q;
        }

        residual[0] = inflow - outflow;
        return true;
    }

private:
    std::vector<Vertex> groupVertices_;
    const Graph& graph_;
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex_;
    const std::map<Vertex, size_t>& vertexToParameterIndex_;
    std::ostream& logFile_;

    template <typename T>
    T getNodePressure(Vertex v, T const* const* parameters) const {
        auto it = vertexToParameterIndex_.find(v);
        if (it != vertexToParameterIndex_.end()) return parameters[0][it->second];
        return T(graph_[v].current_p);
    }
};

class SoftAnchorConstraint {
public:
    SoftAnchorConstraint(size_t parameterIndex, double targetPressure, double weight)
        : parameterIndex_(parameterIndex), targetPressure_(targetPressure), weight_(weight) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residual) const {
        T p = parameters[0][parameterIndex_];
        residual[0] = T(weight_) * (p - T(targetPressure_));
        return true;
    }

private:
    size_t parameterIndex_;
    double targetPressure_;
    double weight_;
};


