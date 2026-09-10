#include "hrv/hrv_controller.h"

#include "aircon/aircon_network_utils.h"
#include "archenv/include/archenv.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <boost/range/iterator_range.hpp>

namespace hrv {
namespace {

constexpr double kLatentHeatJPerKg = 2.501e6; // 水蒸気潜熱の代表値 [J/kg]

struct HrvParams {
    double eta_t = 0.0;
    double eta_x = 0.0;
    std::string recovery = "sensible";
    std::string exhaust_node;
};

HrvParams readParams(const VertexProperties& node) {
    HrvParams p;
    if (!node.ac_spec.is_object()) return p;
    if (node.ac_spec.contains("eta_t") && node.ac_spec["eta_t"].is_number()) {
        p.eta_t = node.ac_spec["eta_t"].get<double>();
    }
    if (node.ac_spec.contains("eta_x") && node.ac_spec["eta_x"].is_number()) {
        p.eta_x = node.ac_spec["eta_x"].get<double>();
    }
    if (node.ac_spec.contains("recovery") && node.ac_spec["recovery"].is_string()) {
        p.recovery = node.ac_spec["recovery"].get<std::string>();
    } else if (!node.model.empty()) {
        p.recovery = node.model;
    }
    if (node.ac_spec.contains("exhaust_node") && node.ac_spec["exhaust_node"].is_string()) {
        p.exhaust_node = node.ac_spec["exhaust_node"].get<std::string>();
    }
    if (p.recovery != "total") {
        p.eta_x = 0.0;
    }
    p.eta_t = std::clamp(p.eta_t, 0.0, 1.0);
    p.eta_x = std::clamp(p.eta_x, 0.0, 1.0);
    return p;
}

std::vector<std::string> listHrvKeys(const ThermalNetwork& thermalNetwork) {
    std::vector<std::string> keys;
    const auto& g = thermalNetwork.getGraph();
    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        if (g[v].type == "hrv") keys.push_back(g[v].key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

double absFlow(const FlowRateMap& flowRates, const std::string& a, const std::string& b) {
    return std::abs(aircon::network_utils::getFlowRate(flowRates, a, b));
}

} // namespace

void updateSupplyBoundaries(ThermalNetwork& thermalNetwork,
                            std::ostream& logs,
                            int logVerbosity) {
    auto& g = thermalNetwork.getGraph();
    const auto& keyToV = thermalNetwork.getKeyToVertex();

    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        auto& node = g[v];
        if (node.type != "hrv") continue;

        const HrvParams params = readParams(node);
        auto itOa = keyToV.find(node.outside_node);
        auto itRa = keyToV.find(node.in_node);
        if (itOa == keyToV.end() || itRa == keyToV.end()) continue;

        const auto& oa = g[itOa->second];
        const auto& ra = g[itRa->second];
        const double tSa = oa.current_t + params.eta_t * (ra.current_t - oa.current_t);
        const double xSa = oa.current_x + params.eta_x * (ra.current_x - oa.current_x);

        node.current_t = tSa;
        node.current_x = xSa;
        // 排気ジャンクションは室空気状態をそのまま持たせる（診断・移流上流用）
        if (!params.exhaust_node.empty()) {
            auto itEa = keyToV.find(params.exhaust_node);
            if (itEa != keyToV.end()) {
                g[itEa->second].current_t = ra.current_t;
                g[itEa->second].current_x = ra.current_x;
            }
        }

        if (logVerbosity >= 2) {
            std::ostringstream oss;
            oss << "　　[HRV] " << node.key
                << " recovery=" << params.recovery
                << " eta_t=" << params.eta_t
                << " eta_x=" << params.eta_x
                << " T_oa=" << oa.current_t << " T_ra=" << ra.current_t
                << " T_sa=" << tSa
                << " x_oa=" << oa.current_x << " x_ra=" << ra.current_x
                << " x_sa=" << xSa;
            logs << oss.str() << "\n";
        }
    }
}

const std::vector<std::string>& orderedKeys(const ThermalNetwork& thermalNetwork) {
    // 呼び出しごとに再構築（台数は少ない）。静的キャッシュはネットワーク再構築で不整合になる。
    static thread_local std::vector<std::string> cache;
    cache = listHrvKeys(thermalNetwork);
    return cache;
}

namespace {

std::pair<std::vector<double>, std::vector<double>>
collectRecovered(ThermalNetwork& thermalNetwork, const FlowRateMap& flowRates) {
    const auto& keys = orderedKeys(thermalNetwork);
    std::vector<double> sens(keys.size(), 0.0);
    std::vector<double> lat(keys.size(), 0.0);
    auto& g = thermalNetwork.getGraph();
    const auto& keyToV = thermalNetwork.getKeyToVertex();
    const double rho = archenv::DENSITY_DRY_AIR;
    const double cp = archenv::SPECIFIC_HEAT_AIR;

    for (size_t i = 0; i < keys.size(); ++i) {
        auto it = keyToV.find(keys[i]);
        if (it == keyToV.end()) continue;
        const auto& node = g[it->second];
        const HrvParams params = readParams(node);
        auto itOa = keyToV.find(node.outside_node);
        if (itOa == keyToV.end()) continue;
        const auto& oa = g[itOa->second];

        const double qSa = absFlow(flowRates, node.outside_node, node.key);
        double qEa = 0.0;
        if (!params.exhaust_node.empty()) {
            qEa = absFlow(flowRates, node.in_node, params.exhaust_node);
        }
        const double qEff = std::min(qSa, qEa > 0.0 ? qEa : qSa);
        const double dT = node.current_t - oa.current_t;
        const double dX = node.current_x - oa.current_x;
        sens[i] = rho * cp * qEff * dT;
        lat[i] = rho * kLatentHeatJPerKg * qEff * dX;
    }
    return {sens, lat};
}

} // namespace

std::vector<double> collectSensibleRecoveredW(ThermalNetwork& thermalNetwork,
                                              const FlowRateMap& flowRates) {
    return collectRecovered(thermalNetwork, flowRates).first;
}

std::vector<double> collectLatentRecoveredW(ThermalNetwork& thermalNetwork,
                                            const FlowRateMap& flowRates) {
    return collectRecovered(thermalNetwork, flowRates).second;
}

} // namespace hrv
