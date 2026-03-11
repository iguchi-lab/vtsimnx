#include "network/thermal_network.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/iterator_range.hpp>

const std::vector<std::string>& ThermalNetwork::getHumidityKeys() const {
    if (!humidityCacheInitialized) {
        std::vector<std::pair<std::string, Vertex>> items;
        items.reserve(boost::num_vertices(graph) / 4 + 1);
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            const bool isCalcX = nd.calc_x;
            const bool isZeroVolume = !(nd.v > 0.0);
            if (!isCalcX && !isZeroVolume) continue;
            items.emplace_back(nd.key, v);
        }
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        humidityVerticesOrdered.clear();
        humidityKeysOrdered.clear();
        humidityVerticesOrdered.reserve(items.size());
        humidityKeysOrdered.reserve(items.size());
        for (const auto& kv : items) {
            humidityKeysOrdered.push_back(kv.first);
            humidityVerticesOrdered.push_back(kv.second);
        }
        humidityCacheInitialized = true;
    }
    return humidityKeysOrdered;
}

std::vector<double> ThermalNetwork::collectHumidityValues() const {
    const auto& keys = getHumidityKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(humidityVerticesOrdered.size());
    for (size_t i = 0; i < humidityVerticesOrdered.size(); ++i) {
        values[i] = graph[humidityVerticesOrdered[i]].current_x;
    }
    return values;
}

