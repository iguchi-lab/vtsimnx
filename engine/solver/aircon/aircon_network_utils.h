#pragma once

#include "network/thermal_network.h"
#include "vtsim_solver.h"

#include <string>

namespace aircon::network_utils {

double getFlowRate(const FlowRateMap& flowRates,
                   const std::string& source,
                   const std::string& target);

bool tryGetTempFromThermalNetwork(const ThermalNetwork& thermalNetwork,
                                  const std::string& nodeKey,
                                  double& outTemp);

double getAbsoluteHumidityFromNode(const ThermalNetwork& thermalNetwork,
                                   const std::string& nodeKey);

} // namespace aircon::network_utils
