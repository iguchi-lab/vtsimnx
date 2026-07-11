#pragma once

#include "types/graph_types.h"

#include <cstdint>
#include <string>
#include <unordered_map>

struct NodeStateView {
    Graph& graph;
    const std::unordered_map<std::string, Vertex>& keyToVertex;
    std::uint64_t topologyRevision = 0;
};

// network 間で共有するノード状態の read-only view。
// network 層同士の直接依存を減らし、runner から明示的に受け渡すために使う。
// topologyRevision は ThermalNetwork::buildFromData 等での再構築検出に使う。
struct ConstNodeStateView {
    const Graph& graph;
    const std::unordered_map<std::string, Vertex>& keyToVertex;
    std::uint64_t topologyRevision = 0;
};

