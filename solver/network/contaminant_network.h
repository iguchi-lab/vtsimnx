#pragma once

#include "types/graph_types.h"

#include <unordered_map>
#include <utility>
#include <vector>

class VentilationNetwork;
class ThermalNetwork;

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
    void buildTerms(const Graph& nodeGraph,
                    const ThermalNetwork& thermalNetwork,
                    const VentilationNetwork& ventNetwork,
                    ContaminantNetworkTerms& terms) const;

    // 汚染物質濃度(c)の出力取得窓口
    const std::vector<std::string>& getOutputKeys(const ThermalNetwork& thermalNetwork) const;
    std::vector<double> collectOutputValues(const ThermalNetwork& thermalNetwork) const;
    void invalidateCaches();

private:
    void ensureNodeIndex(const ThermalNetwork& thermalNetwork) const;

    mutable bool nodeIndexInitialized = false;
    mutable std::unordered_map<std::string, Vertex> nodeKeyToVertex;
    mutable bool outputCacheInitialized = false;
    mutable std::vector<Vertex> outputVerticesOrdered;
    mutable std::vector<std::string> outputKeysOrdered;
};

