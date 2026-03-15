#include "aircon/aircon_network_utils.h"

namespace aircon::network_utils {

double getFlowRate(const FlowRateMap& flowRates,
                   const std::string& source,
                   const std::string& target) {
    auto direct = flowRates.find({source, target});
    if (direct != flowRates.end()) {
        return direct->second;
    }
    auto reverse = flowRates.find({target, source});
    if (reverse != flowRates.end()) {
        return -reverse->second;
    }
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

double getAbsoluteHumidityFromNode(const ThermalNetwork& thermalNetwork,
                                   const std::string& nodeKey) {
    if (nodeKey.empty()) return 0.0;
    const auto& keyToV = thermalNetwork.getKeyToVertex();
    auto it = keyToV.find(nodeKey);
    if (it == keyToV.end()) return 0.0;
    return thermalNetwork.getGraph()[it->second].current_x;
}

} // namespace aircon::network_utils
