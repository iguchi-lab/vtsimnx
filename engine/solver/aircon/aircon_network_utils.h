#pragma once

#include "network/thermal_network.h"
#include "vtsim_solver.h"

#include <string>

namespace aircon::network_utils {

double getFlowRate(const FlowRateMap& flowRates,
                   const std::string& source,
                   const std::string& target);

// 空調処理風量: 還気枝 (in→aircon) を優先し、無ければ吹出枝の絶対値。
// 還気+吹出の双方向固定流量を getFlowRate でネットすると 0 になるため分離する。
double getAirconProcessFlowRate(const FlowRateMap& flowRates,
                                const std::string& inNode,
                                const std::string& airconNode);

bool tryGetTempFromThermalNetwork(const ThermalNetwork& thermalNetwork,
                                  const std::string& nodeKey,
                                  double& outTemp);

double getAbsoluteHumidityFromNode(const ThermalNetwork& thermalNetwork,
                                   const std::string& nodeKey);

} // namespace aircon::network_utils
