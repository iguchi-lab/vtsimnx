#include "aircon/aircon_network_utils.h"

#include <cmath>
#include <stdexcept>

namespace aircon::network_utils {

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
