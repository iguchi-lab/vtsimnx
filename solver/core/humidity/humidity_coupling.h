#pragma once

#include "types/common_types.h"
#include "types/graph_types.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::humidity {

using WeightedVertexLinks = std::vector<std::vector<std::pair<Vertex, double>>>;

struct NetworkTerms {
    std::unordered_map<Vertex, double> genByVertex;
    std::vector<double> outSum;
    WeightedVertexLinks inflow;
    WeightedVertexLinks moistureLinks;
    std::vector<Vertex> updateVertices;
};

struct SolveStats {
    int iterations = 0;
    double finalMaxDiff = 0.0;
    bool converged = true;
};

void buildHumidityNetworkTerms(const Graph& vGraph,
                               const Graph& tGraph,
                               const std::unordered_map<std::string, Vertex>& tKeyToV,
                               NetworkTerms& terms);

void initializeHumidityState(const Graph& tGraph,
                             std::vector<double>& xOld,
                             std::vector<double>& xNew);

SolveStats solveHumidityImplicitStep(const Graph& tGraph,
                                     const NetworkTerms& terms,
                                     double dt,
                                     int maxIter,
                                     double tolerance,
                                     std::vector<double>& xNew,
                                     const std::vector<double>& xOld);

void applyHumidityStateToGraphs(Graph& tGraph,
                                Graph& vGraph,
                                const std::unordered_map<std::string, Vertex>& vKeyToV,
                                const std::vector<Vertex>& updateVertices,
                                const std::vector<double>& xNew);

} // namespace core::humidity

