#pragma once

#include "types/graph_types.h"

#include <string>
#include <unordered_map>

struct NodeStateView {
    Graph& graph;
    const std::unordered_map<std::string, Vertex>& keyToVertex;
};

// network 間で共有するノード状態の read-only view。
// network 層同士の直接依存を減らし、runner から明示的に受け渡すために使う。
struct ConstNodeStateView {
    const Graph& graph;
    const std::unordered_map<std::string, Vertex>& keyToVertex;
};

