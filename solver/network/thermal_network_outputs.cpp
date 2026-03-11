#include "network/thermal_network.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/iterator_range.hpp>

const std::vector<std::string>& ThermalNetwork::getTemperatureKeys() const {
    if (!temperatureCacheInitialized) {
        // directedS のため in_degree が使えない（in-edge リストを持たない）ので、
        // 全エッジを 1 回走査して「接続のある頂点」をマーキングする。
        std::vector<uint8_t> connected;
        connected.assign(boost::num_vertices(graph), 0);
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const Vertex sv = boost::source(e, graph);
            const Vertex tv = boost::target(e, graph);
            connected[static_cast<size_t>(sv)] = 1;
            connected[static_cast<size_t>(tv)] = 1;
        }

        // TemperatureMap は std::map でキーが昇順になるため、同じ順序（key昇順）で固定する
        std::vector<std::pair<std::string, Vertex>> itemsMain;
        std::vector<std::pair<std::string, Vertex>> itemsCap;
        std::vector<std::pair<std::string, Vertex>> itemsLayer;
        itemsMain.reserve(boost::num_vertices(graph));
        itemsCap.reserve(boost::num_vertices(graph) / 8 + 1);
        itemsLayer.reserve(boost::num_vertices(graph) / 2 + 1);

        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& nd = graph[v];
            if (static_cast<size_t>(v) >= connected.size() || connected[static_cast<size_t>(v)] == 0) continue;

            const auto tc = nd.getTypeCode();
            if (tc == VertexProperties::TypeCode::Capacity) {
                itemsCap.emplace_back(nd.key, v);
            } else if (tc == VertexProperties::TypeCode::Layer) {
                itemsLayer.emplace_back(nd.key, v);
            } else {
                // main: normal + aircon + unknown
                itemsMain.emplace_back(nd.key, v);
            }
        }
        std::sort(itemsMain.begin(), itemsMain.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::sort(itemsCap.begin(), itemsCap.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::sort(itemsLayer.begin(), itemsLayer.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        temperatureVerticesOrdered.clear();
        temperatureKeysOrdered.clear();
        temperatureVerticesOrderedCapacity.clear();
        temperatureKeysOrderedCapacity.clear();
        temperatureVerticesOrderedLayer.clear();
        temperatureKeysOrderedLayer.clear();

        temperatureVerticesOrdered.reserve(itemsMain.size());
        temperatureKeysOrdered.reserve(itemsMain.size());
        temperatureVerticesOrderedCapacity.reserve(itemsCap.size());
        temperatureKeysOrderedCapacity.reserve(itemsCap.size());
        temperatureVerticesOrderedLayer.reserve(itemsLayer.size());
        temperatureKeysOrderedLayer.reserve(itemsLayer.size());

        for (const auto& kv : itemsMain) {
            temperatureKeysOrdered.push_back(kv.first);
            temperatureVerticesOrdered.push_back(kv.second);
        }
        for (const auto& kv : itemsCap) {
            temperatureKeysOrderedCapacity.push_back(kv.first);
            temperatureVerticesOrderedCapacity.push_back(kv.second);
        }
        for (const auto& kv : itemsLayer) {
            temperatureKeysOrderedLayer.push_back(kv.first);
            temperatureVerticesOrderedLayer.push_back(kv.second);
        }
        temperatureCacheInitialized = true;
    }
    return temperatureKeysOrdered;
}

std::vector<double> ThermalNetwork::collectTemperatureValues() const {
    const auto& keys = getTemperatureKeys();
    (void)keys;
    std::vector<double> values;
    values.resize(temperatureVerticesOrdered.size());
    for (size_t i = 0; i < temperatureVerticesOrdered.size(); ++i) {
        values[i] = graph[temperatureVerticesOrdered[i]].current_t;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getTemperatureKeysCapacity() const {
    // main を呼べばキャッシュが構築される
    (void)getTemperatureKeys();
    return temperatureKeysOrderedCapacity;
}

std::vector<double> ThermalNetwork::collectTemperatureValuesCapacity() const {
    const auto& keys = getTemperatureKeysCapacity();
    (void)keys;
    std::vector<double> values;
    values.resize(temperatureVerticesOrderedCapacity.size());
    for (size_t i = 0; i < temperatureVerticesOrderedCapacity.size(); ++i) {
        values[i] = graph[temperatureVerticesOrderedCapacity[i]].current_t;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getTemperatureKeysLayer() const {
    // main を呼べばキャッシュが構築される
    (void)getTemperatureKeys();
    return temperatureKeysOrderedLayer;
}

std::vector<double> ThermalNetwork::collectTemperatureValuesLayer() const {
    const auto& keys = getTemperatureKeysLayer();
    (void)keys;
    std::vector<double> values;
    values.resize(temperatureVerticesOrderedLayer.size());
    for (size_t i = 0; i < temperatureVerticesOrderedLayer.size(); ++i) {
        values[i] = graph[temperatureVerticesOrderedLayer[i]].current_t;
    }
    return values;
}

namespace {
inline std::string heatRateOutputKeyFromUniqueId(const std::string& uniqueId) {
    std::string key = uniqueId;
    const std::string suffix = "_000";
    if (key.size() > suffix.size() &&
        key.rfind(suffix) == key.size() - suffix.size()) {
        key.erase(key.size() - suffix.size());
    }
    return key;
}
} // namespace

static void buildHeatRateCachesIfNeeded(const Graph& graph,
                                        bool& cacheInitialized,
                                        std::vector<Edge>& edgesAdvection,
                                        std::vector<std::string>& keysAdvection,
                                        std::vector<Edge>& edgesHeatGen,
                                        std::vector<std::string>& keysHeatGen,
                                        std::vector<Edge>& edgesSolar,
                                        std::vector<std::string>& keysSolar,
                                        std::vector<Edge>& edgesNoct,
                                        std::vector<std::string>& keysNoct,
                                        std::vector<Edge>& edgesConv,
                                        std::vector<std::string>& keysConv,
                                        std::vector<Edge>& edgesCond,
                                        std::vector<std::string>& keysCond,
                                        std::vector<Edge>& edgesRad,
                                        std::vector<std::string>& keysRad,
                                        std::vector<Edge>& edgesCap,
                                        std::vector<std::string>& keysCap) {
    if (cacheInitialized) return;

    std::vector<std::pair<std::string, Edge>> itemsAdvection;
    std::vector<std::pair<std::string, Edge>> itemsHeatGen;
    std::vector<std::pair<std::string, Edge>> itemsSolar;
    std::vector<std::pair<std::string, Edge>> itemsNoct;
    std::vector<std::pair<std::string, Edge>> itemsConv;
    std::vector<std::pair<std::string, Edge>> itemsCond;
    std::vector<std::pair<std::string, Edge>> itemsRad;
    std::vector<std::pair<std::string, Edge>> itemsCap;

    const size_t eCount = static_cast<size_t>(boost::num_edges(graph));
    itemsAdvection.reserve(eCount / 4 + 1);
    itemsHeatGen.reserve(eCount / 16 + 1);
    itemsSolar.reserve(eCount / 16 + 1);
    itemsNoct.reserve(eCount / 16 + 1);
    itemsConv.reserve(eCount / 8 + 1);
    itemsCond.reserve(eCount / 2 + 1);
    itemsRad.reserve(eCount / 8 + 1);
    itemsCap.reserve(eCount / 16 + 1);

    for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
        const auto& ep = graph[edge];
        const auto typeCode = ep.getTypeCode();
        const std::string key = heatRateOutputKeyFromUniqueId(ep.unique_id);

        if (typeCode == EdgeProperties::TypeCode::Advection) {
            itemsAdvection.emplace_back(key, edge);
            continue;
        }

        if (typeCode == EdgeProperties::TypeCode::HeatGeneration) {
            if (ep.subtype == "solar_gain") {
                itemsSolar.emplace_back(key, edge);
            } else if (ep.subtype == "nocturnal_loss") {
                itemsNoct.emplace_back(key, edge);
            } else {
                itemsHeatGen.emplace_back(key, edge);
            }
            continue;
        }

        if (ep.subtype == "convection") {
            itemsConv.emplace_back(key, edge);
        } else if (ep.subtype == "radiation") {
            itemsRad.emplace_back(key, edge);
        } else if (ep.subtype == "capacity") {
            itemsCap.emplace_back(key, edge);
        } else {
            itemsCond.emplace_back(key, edge);
        }
    }

    auto sortItems = [](auto& items) {
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    };
    sortItems(itemsAdvection);
    sortItems(itemsHeatGen);
    sortItems(itemsSolar);
    sortItems(itemsNoct);
    sortItems(itemsConv);
    sortItems(itemsCond);
    sortItems(itemsRad);
    sortItems(itemsCap);

    auto fillOut = [](const auto& items, std::vector<Edge>& edgesOut, std::vector<std::string>& keysOut) {
        edgesOut.clear();
        keysOut.clear();
        edgesOut.reserve(items.size());
        keysOut.reserve(items.size());
        for (const auto& kv : items) {
            keysOut.push_back(kv.first);
            edgesOut.push_back(kv.second);
        }
    };

    fillOut(itemsAdvection, edgesAdvection, keysAdvection);
    fillOut(itemsHeatGen, edgesHeatGen, keysHeatGen);
    fillOut(itemsSolar, edgesSolar, keysSolar);
    fillOut(itemsNoct, edgesNoct, keysNoct);
    fillOut(itemsConv, edgesConv, keysConv);
    fillOut(itemsCond, edgesCond, keysCond);
    fillOut(itemsRad, edgesRad, keysRad);
    fillOut(itemsCap, edgesCap, keysCap);

    cacheInitialized = true;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysAdvection() const {
    buildHeatRateCachesIfNeeded(graph,
                                heatRateCacheInitialized,
                                heatRateEdgesOrderedAdvection, heatRateKeysOrderedAdvection,
                                heatRateEdgesOrderedHeatGeneration, heatRateKeysOrderedHeatGeneration,
                                heatRateEdgesOrderedSolarGain, heatRateKeysOrderedSolarGain,
                                heatRateEdgesOrderedNocturnalLoss, heatRateKeysOrderedNocturnalLoss,
                                heatRateEdgesOrderedConvection, heatRateKeysOrderedConvection,
                                heatRateEdgesOrderedConduction, heatRateKeysOrderedConduction,
                                heatRateEdgesOrderedRadiation, heatRateKeysOrderedRadiation,
                                heatRateEdgesOrderedCapacity, heatRateKeysOrderedCapacity);
    return heatRateKeysOrderedAdvection;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesAdvection() const {
    (void)getHeatRateKeysAdvection();
    std::vector<double> values(heatRateEdgesOrderedAdvection.size());
    for (size_t i = 0; i < heatRateEdgesOrderedAdvection.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedAdvection[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysHeatGeneration() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedHeatGeneration;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesHeatGeneration() const {
    (void)getHeatRateKeysHeatGeneration();
    std::vector<double> values(heatRateEdgesOrderedHeatGeneration.size());
    for (size_t i = 0; i < heatRateEdgesOrderedHeatGeneration.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedHeatGeneration[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysSolarGain() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedSolarGain;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesSolarGain() const {
    (void)getHeatRateKeysSolarGain();
    std::vector<double> values(heatRateEdgesOrderedSolarGain.size());
    for (size_t i = 0; i < heatRateEdgesOrderedSolarGain.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedSolarGain[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysNocturnalLoss() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedNocturnalLoss;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesNocturnalLoss() const {
    (void)getHeatRateKeysNocturnalLoss();
    std::vector<double> values(heatRateEdgesOrderedNocturnalLoss.size());
    for (size_t i = 0; i < heatRateEdgesOrderedNocturnalLoss.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedNocturnalLoss[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysConvection() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedConvection;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesConvection() const {
    (void)getHeatRateKeysConvection();
    std::vector<double> values(heatRateEdgesOrderedConvection.size());
    for (size_t i = 0; i < heatRateEdgesOrderedConvection.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedConvection[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysConduction() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedConduction;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesConduction() const {
    (void)getHeatRateKeysConduction();
    std::vector<double> values(heatRateEdgesOrderedConduction.size());
    for (size_t i = 0; i < heatRateEdgesOrderedConduction.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedConduction[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysRadiation() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedRadiation;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesRadiation() const {
    (void)getHeatRateKeysRadiation();
    std::vector<double> values(heatRateEdgesOrderedRadiation.size());
    for (size_t i = 0; i < heatRateEdgesOrderedRadiation.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedRadiation[i]].heat_rate;
    }
    return values;
}

const std::vector<std::string>& ThermalNetwork::getHeatRateKeysCapacity() const {
    (void)getHeatRateKeysAdvection();
    return heatRateKeysOrderedCapacity;
}

std::vector<double> ThermalNetwork::collectHeatRateValuesCapacity() const {
    (void)getHeatRateKeysCapacity();
    std::vector<double> values(heatRateEdgesOrderedCapacity.size());
    for (size_t i = 0; i < heatRateEdgesOrderedCapacity.size(); ++i) {
        values[i] = graph[heatRateEdgesOrderedCapacity[i]].heat_rate;
    }
    return values;
}

