#include "network/contaminant_network.h"
#include "network/thermal_network.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/iterator_range.hpp>

const std::vector<std::string>& ContaminantNetwork::getOutputKeys(const ThermalNetwork& thermalNetwork) const {
    if (!outputCacheInitialized) {
        const auto& graph = thermalNetwork.getGraph();
        std::vector<std::pair<std::string, Vertex>> items;
        items.reserve(boost::num_vertices(graph) / 4 + 1);
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            if (!nd.calc_c) continue;
            items.emplace_back(nd.key, v);
        }
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        outputVerticesOrdered.clear();
        outputKeysOrdered.clear();
        outputVerticesOrdered.reserve(items.size());
        outputKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            outputKeysOrdered.push_back(kv.first);
            outputVerticesOrdered.push_back(kv.second);
        }
        outputCacheInitialized = true;
    }
    return outputKeysOrdered;
}

std::vector<double> ContaminantNetwork::collectOutputValues(const ThermalNetwork& thermalNetwork) const {
    const auto& keys = getOutputKeys(thermalNetwork);
    (void)keys;
    const auto& graph = thermalNetwork.getGraph();
    std::vector<double> values;
    values.resize(outputVerticesOrdered.size());
    for (size_t i = 0; i < outputVerticesOrdered.size(); ++i) {
        values[i] = graph[outputVerticesOrdered[i]].current_c;
    }
    return values;
}

void ContaminantNetwork::invalidateOutputCache() {
    outputCacheInitialized = false;
    outputVerticesOrdered.clear();
    outputKeysOrdered.clear();
}

