#pragma once

#include "aircon/aircon_operation_mode.h"
#include "vtsim_solver.h"

#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace aircon::capacity {

std::optional<double> resolveMaxHeatCapacity(const VertexProperties& nodeProps,
                                             OperationMode operationMode,
                                             std::string& source);

void applyExceededCapacityAdjustment(
    const std::string& airconKey,
    VertexProperties& nodeProps,
    OperationMode operationMode,
    double indoorTemp,
    double airFlowRate,
    double maxHeatCapacity,
    double currentTotal,
    std::unordered_map<std::string, std::pair<double, double>>& capacityLimitBracket,
    std::ostringstream& oss,
    bool& adjustmentMade);

void applyUnderCapacityBracketAdjustment(
    const std::string& airconKey,
    VertexProperties& nodeProps,
    OperationMode operationMode,
    double maxHeatCapacity,
    double currentTotal,
    std::unordered_map<std::string, std::pair<double, double>>& capacityLimitBracket,
    std::ostringstream& oss,
    bool& adjustmentMade);

} // namespace aircon::capacity
