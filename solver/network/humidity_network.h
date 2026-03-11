#pragma once

#include "types/graph_types.h"

#include <unordered_map>
#include <utility>
#include <vector>

class VentilationNetwork;
class ThermalNetwork;

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
    void buildTerms(const Graph& nodeGraph,
                    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
                    const VentilationNetwork& ventNetwork,
                    HumidityNetworkTerms& terms) const;

    // 絶対湿度(x)の出力取得窓口
    const std::vector<std::string>& getOutputKeys(const ThermalNetwork& thermalNetwork) const;
    std::vector<double> collectOutputValues(const ThermalNetwork& thermalNetwork) const;
    void invalidateOutputCache();

private:
    mutable bool outputCacheInitialized = false;
    mutable std::vector<Vertex> outputVerticesOrdered;
    mutable std::vector<std::string> outputKeysOrdered;
};

