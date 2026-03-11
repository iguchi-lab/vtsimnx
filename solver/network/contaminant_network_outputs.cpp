#include "network/thermal_network.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/iterator_range.hpp>

const std::vector<std::string>& ThermalNetwork::getConcentrationKeys() const {
    if (!concentrationCacheInitialized) {
        std::vector<std::pair<std::string, Vertex>> items;
        items.reserve(boost::num_vertices(graph) / 4 + 1);
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            if (!nd.calc_c) continue;
            items.emplace_back(nd.key, v);
        }
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        concentrationVerticesOrdered.clear();
        concentrationKeysOrdered.clear();
        concentrationVerticesOrdered.reserve(items.size());
        concentrationKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            concentrationKeysOrdered.push_back(kv.first);
            concentrationVerticesOrdered.push_back(kv.second);
        }
        concentrationCacheInitialized = true;
    }
    return concentrationKeysOrdered;
}

std::vector<double> ThermalNetwork::collectConcentrationValues() const {
    const auto& keys = getConcentrationKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(concentrationVerticesOrdered.size());
    for (size_t i = 0; i < concentrationVerticesOrdered.size(); ++i) {
        values[i] = graph[concentrationVerticesOrdered[i]].current_c;
    }
    return values;
}

