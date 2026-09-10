#pragma once

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "types/common_types.h"

#include <ostream>
#include <string>
#include <vector>

namespace hrv {

// 給気ノード (type=hrv) の T/x 境界を、室外・還気と交換効率から更新する。
void updateSupplyBoundaries(ThermalNetwork& thermalNetwork,
                            std::ostream& logs,
                            int logVerbosity);

// キー順（昇順）で回収顕熱 [W] を返す。潜熱は全熱のみ非ゼロ。
const std::vector<std::string>& orderedKeys(const ThermalNetwork& thermalNetwork);
std::vector<double> collectSensibleRecoveredW(ThermalNetwork& thermalNetwork,
                                              const FlowRateMap& flowRates);
std::vector<double> collectLatentRecoveredW(ThermalNetwork& thermalNetwork,
                                            const FlowRateMap& flowRates);

} // namespace hrv
