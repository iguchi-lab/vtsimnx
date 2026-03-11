#pragma once

#include "types/graph_types.h"

#include <unordered_map>
#include <utility>
#include <vector>

class VentilationNetwork;

using WeightedContaminantLinks = std::vector<std::vector<std::pair<Vertex, double>>>;

struct ContaminantNetworkTerms {
    std::unordered_map<Vertex, double> genByVertex;
    std::vector<double> outSum;
    WeightedContaminantLinks inflowCoeff;
    std::vector<Vertex> updateVertices;
};

// 汚染物質濃度(c)ネットワークの項組み立て責務を集約するヘルパー。
class ContaminantNetwork {
public:
    static void buildTerms(const Graph& nodeGraph,
                           const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
                           const VentilationNetwork& ventNetwork,
                           ContaminantNetworkTerms& terms);
};

