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

// ON/OFF 判定用の制御室温。set_node があれば必須（欠落は例外）、無ければ aircon ノード、
// それも無ければ aircon.current_t。
double getControlledRoomTemperature(const ThermalNetwork& thermalNetwork,
                                    const VertexProperties& airconNode);

double getAbsoluteHumidityFromNode(const ThermalNetwork& thermalNetwork,
                                   const std::string& nodeKey);

} // namespace aircon::network_utils
