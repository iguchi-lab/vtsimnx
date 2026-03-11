#pragma once

#include "types/graph_types.h"

#include <string>
#include <unordered_map>

struct NodeStateView {
    Graph& graph;
    const std::unordered_map<std::string, Vertex>& keyToVertex;
};

struct ConstNodeStateView {
    const Graph& graph;
    const std::unordered_map<std::string, Vertex>& keyToVertex;
};

