#include "aircon/aircon_network_utils.h"

#include "aircon/aircon_saturation.h"
#include <boost/range/iterator_range.hpp>
#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace aircon::network_utils {

OperationMode fanMode(const VertexProperties& node) {
    // OFF has no compressor mode. Prefer cooling fan data when available,
    // otherwise use a heating-only unit's fan specification.
    if (node.current_mode == "OFF" && node.ac_spec.contains("V_inner") &&
        !node.ac_spec["V_inner"].contains("cooling") && node.ac_spec["V_inner"].contains("heating"))
        return OperationMode::Heating;
    return resolveOperationModeForRuntime(node.current_mode, 0.0, 0.0);
}


std::vector<PassiveSupplyOutletConstraint> passiveSupplyOutletConstraints(
    const ThermalNetwork& network, const VertexProperties& ac) {
    const auto& graph = network.getGraph();
    const auto found = network.getKeyToVertex().find(ac.key);
    if (found == network.getKeyToVertex().end()) return {};
    const Vertex av = found->second;
    std::vector<Vertex> candidates;
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        const auto& edge = graph[e];
        if (!edge.current_enabled || edge.getTypeCode() != EdgeProperties::TypeCode::Advection ||
            edge.flow_rate == 0.0) continue;
        Vertex source = boost::source(e, graph), target = boost::target(e, graph);
        if (edge.flow_rate < 0.0) std::swap(source, target);
        if (source != av || target == av) continue;
        const auto& node = graph[target];
        if (node.key == ac.in_node || node.key == ac.set_node ||
            node.getTypeCode() != VertexProperties::TypeCode::Normal || !node.calc_x ||
            node.v > 0.0 || node.moisture_capacity > 0.0 || node.heat_source != 0.0) continue;
        if (std::find(candidates.begin(), candidates.end(), target) == candidates.end())
            candidates.push_back(target);
    }
    std::vector<PassiveSupplyOutletConstraint> result;
    for (Vertex outlet : candidates) {
        double inflow = 0.0, outflow = 0.0;
        bool passive = true;
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            const auto& edge = graph[e];
            Vertex source = boost::source(e, graph), target = boost::target(e, graph);
            if (!edge.current_enabled || (source != outlet && target != outlet)) continue;
            if (edge.getTypeCode() != EdgeProperties::TypeCode::Advection ||
                edge.moisture_conductance > 0.0 ||
                (target == outlet && edge.current_humidity_generation != 0.0)) {
                passive = false;
                break;
            }
            if (edge.flow_rate == 0.0) continue;
            const double flow = std::abs(edge.flow_rate);
            if (edge.flow_rate < 0.0) std::swap(source, target);
            if (target == outlet) {
                if (source != av) { passive = false; break; }
                inflow += flow;
            }
            if (source == outlet) outflow += flow;
        }
        if (!passive || !(inflow > std::numeric_limits<double>::epsilon()) ||
            !(outflow > std::numeric_limits<double>::epsilon())) continue;
        const double outletLimit = aircon::latent::saturationHumidityUpperBound(graph[outlet].current_t);
        // The humidity equation is q_out*x_out = q_in*x_AC. Pressure may be
        // accepted with a small mass residual, so x_out need not equal x_AC.
        // Enforce the outlet constraint upstream, then solve water again.
        const double upstreamLimit = outletLimit * outflow / inflow *
            (1.0 - 16.0 * std::numeric_limits<double>::epsilon());
        result.push_back({graph[outlet].key, upstreamLimit, outletLimit});
    }
    return result;
}

double getFlowRate(const FlowRateMap& flowRates,
                   const std::string& source,
                   const std::string& target) {
    const auto direct = flowRates.find({source, target});
    const auto reverse = flowRates.find({target, source});
    const bool hasDirect = (direct != flowRates.end());
    const bool hasReverse = (reverse != flowRates.end());
    if (!hasDirect && !hasReverse) return 0.0;
    double q = 0.0;
    if (hasDirect) q += direct->second;
    if (hasReverse) q -= reverse->second;
    return q;
}

double getAirconProcessFlowRate(const FlowRateMap& flowRates,
                                const std::string& inNode,
                                const std::string& airconNode) {
    if (inNode.empty() || airconNode.empty()) return 0.0;
    const auto ret = flowRates.find({inNode, airconNode});
    if (ret != flowRates.end()) return std::abs(ret->second);
    const auto supply = flowRates.find({airconNode, inNode});
    if (supply != flowRates.end()) return std::abs(supply->second);
    return 0.0;
}

bool tryGetTempFromThermalNetwork(const ThermalNetwork& thermalNetwork,
                                  const std::string& nodeKey,
                                  double& outTemp) {
    if (nodeKey.empty()) return false;
    const auto& keyToV = thermalNetwork.getKeyToVertex();
    auto it = keyToV.find(nodeKey);
    if (it == keyToV.end()) return false;
    outTemp = thermalNetwork.getGraph()[it->second].current_t;
    return true;
}

double getControlledRoomTemperature(const ThermalNetwork& thermalNetwork,
                                    const VertexProperties& airconNode) {
    double currentTemp = 0.0;
    if (!airconNode.set_node.empty()) {
        if (!tryGetTempFromThermalNetwork(thermalNetwork, airconNode.set_node, currentTemp)) {
            throw std::runtime_error(
                "aircon set_node temperature lookup failed: aircon=" + airconNode.key +
                ", set_node=" + airconNode.set_node);
        }
        return currentTemp;
    }
    if (!tryGetTempFromThermalNetwork(thermalNetwork, airconNode.key, currentTemp)) {
        return airconNode.current_t;
    }
    return currentTemp;
}

double getAbsoluteHumidityFromNode(const ThermalNetwork& thermalNetwork,
                                   const std::string& nodeKey) {
    if (nodeKey.empty()) return 0.0;
    const auto& keyToV = thermalNetwork.getKeyToVertex();
    auto it = keyToV.find(nodeKey);
    if (it == keyToV.end()) return 0.0;
    return thermalNetwork.getGraph()[it->second].current_x;
}

} // namespace aircon::network_utils
