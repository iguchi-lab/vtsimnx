#pragma once

#include "network/node_state_view.h"
#include "types/graph_types.h"

#include <unordered_map>
#include <utility>
#include <vector>

class VentilationNetwork;

using WeightedVertexLinks = std::vector<std::vector<std::pair<Vertex, double>>>;

struct HumidityNetworkTerms {
    std::unordered_map<Vertex, double> genByVertex;
    std::vector<double> outSum;
    WeightedVertexLinks inflow;
    WeightedVertexLinks moistureLinks;
    std::vector<Vertex> updateVertices;
};

// 湿気ネットワーク固有の組み立て責務を集約するヘルパー。
// ノード状態は呼び出し元（現状: ThermalNetwork）が保持し、ここでは参照のみ行う。
class HumidityNetwork {
public:
    // 2) Build / Update / Sync
    void buildTerms(ConstNodeStateView nodeState,
                    const VentilationNetwork& ventNetwork,
                    HumidityNetworkTerms& terms) const;

    // 4) Output APIs
    const std::vector<std::string>& getOutputKeys(ConstNodeStateView nodeState) const;
    std::vector<double> collectOutputValues(ConstNodeStateView nodeState) const;

    // 5) Diagnostics / cache controls
    void invalidateCaches();

private:
    void ensureNodeIndex(ConstNodeStateView nodeState) const;

    mutable bool nodeIndexInitialized = false;
    mutable std::unordered_map<std::string, Vertex> nodeKeyToVertex;
    mutable bool outputCacheInitialized = false;
    mutable std::vector<Vertex> outputVerticesOrdered;
    mutable std::vector<std::string> outputKeysOrdered;
};

