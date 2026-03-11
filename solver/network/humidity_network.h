#pragma once

#include "types/graph_types.h"

#include <unordered_map>
#include <utility>
#include <vector>

using WeightedVertexLinks = std::vector<std::vector<std::pair<Vertex, double>>>;

struct HumidityNetworkTerms {
    std::unordered_map<Vertex, double> genByVertex;
    std::vector<double> outSum;
    WeightedVertexLinks inflow;
    WeightedVertexLinks moistureLinks;
    std::vector<Vertex> updateVertices;
};

