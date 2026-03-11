#pragma once

#include "types/common_types.h"
#include "types/graph_types.h"
#include "network/humidity_network.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::humidity {

struct SolveStats {
    int iterations = 0;
    double finalMaxDiff = 0.0;
    bool converged = true;
};

void initializeHumidityState(const Graph& tGraph,
                             std::vector<double>& xOld,
                             std::vector<double>& xNew);

SolveStats solveHumidityImplicitStep(const Graph& tGraph,
                                     const HumidityNetworkTerms& terms,
                                     double dt,
                                     double tolerance,
                                     std::vector<double>& xNew,
                                     const std::vector<double>& xOld);

void applyHumidityStateToGraphs(Graph& tGraph,
                                Graph& vGraph,
                                const std::unordered_map<std::string, Vertex>& vKeyToV,
                                const std::vector<Vertex>& updateVertices,
                                const std::vector<double>& xNew);

} // namespace core::humidity

