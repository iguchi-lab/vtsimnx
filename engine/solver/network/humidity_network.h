#pragma once

#include "network/node_state_view.h"
#include "types/graph_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

class VentilationNetwork;

namespace core::humidity {
struct HumiditySolverContext;
}

using WeightedVertexLinks = std::vector<std::vector<std::pair<Vertex, double>>>;

struct HumidityNetworkTerms {
    std::unordered_map<Vertex, double> genByVertex;
    std::vector<double> outSum;
    WeightedVertexLinks inflow;
    WeightedVertexLinks moistureLinks;      // 全 moisture_conductance（湿度方程式）
    WeightedVertexLinks phaseChangeLinks;   // moisture_transfer_type=phase_change のみ
    // 有効換気枝の無向隣接（流量・向き非依存）。疎行列の固定上位パターン用。
    std::vector<std::vector<Vertex>> ventNeighbors;
    std::vector<Vertex> updateVertices;
};

// 湿気収支の内訳 [kg/s]（診断用。ソルバ数値には影響しない）
struct MoistureBalanceTerms {
    std::vector<double> ventilationTransport;
    std::vector<double> vaporGeneration;
    std::vector<double> materialPhaseChange; // moisture_conductance による正味水蒸気流入
    // 空調ノードの除湿 [kg/s]（負=空気から除去）。吹出 x=supplyX 境界と対応。
    std::vector<double> airconCondensation;
    std::vector<double> storage;             // C*(x_new-x_n)/dt
    std::vector<double> residual;            // storage - (vent+gen+material+aircon)
    double maxAbsResidual = 0.0;
};

// 湿気ネットワーク固有の組み立て責務を集約するヘルパー。
// ノード状態は呼び出し元（現状: ThermalNetwork）が保持し、ここでは参照のみ行う。
class HumidityNetwork {
public:
    HumidityNetwork();
    ~HumidityNetwork();
    HumidityNetwork(const HumidityNetwork&) = delete;
    HumidityNetwork& operator=(const HumidityNetwork&) = delete;
    HumidityNetwork(HumidityNetwork&&) noexcept;
    HumidityNetwork& operator=(HumidityNetwork&&) noexcept;

    // 2) Build / Update / Sync
    void buildTerms(ConstNodeStateView nodeState,
                    const VentilationNetwork& ventNetwork,
                    HumidityNetworkTerms& terms) const;

    // 4) Output APIs
    const std::vector<std::string>& getOutputKeys(ConstNodeStateView nodeState) const;
    std::vector<double> collectOutputValues(ConstNodeStateView nodeState) const;

    // 5) Diagnostics / cache controls
    void invalidateCaches();

    core::humidity::HumiditySolverContext& humiditySolverContext();

    const MoistureBalanceTerms& lastMoistureBalance() const { return lastMoistureBalance_; }
    void setLastMoistureBalance(MoistureBalanceTerms bal) {
        lastMoistureBalance_ = std::move(bal);
    }

private:
    void ensureNodeIndex(ConstNodeStateView nodeState) const;

    mutable bool nodeIndexInitialized = false;
    mutable const Graph* cachedGraphPtr_ = nullptr;
    mutable size_t cachedNumVertices_ = 0;
    mutable size_t cachedNumEdges_ = 0;
    mutable std::uint64_t cachedTopologyRevision_ = 0;
    mutable std::unordered_map<std::string, Vertex> nodeKeyToVertex;
    mutable bool outputCacheInitialized = false;
    mutable std::vector<Vertex> outputVerticesOrdered;
    mutable std::vector<std::string> outputKeysOrdered;

    std::unique_ptr<core::humidity::HumiditySolverContext> humiditySolverContext_;
    MoistureBalanceTerms lastMoistureBalance_;
};
