#pragma once

#include "aircon/aircon_operation_mode.h"
#include "vtsim_solver.h"

#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace aircon::capacity {

/** 能力制限の設定温度探索 bracket。 */
struct CapacityBracket {
    double tLow = 0.0;
    double tHigh = 0.0;
    /** bracket 幅収束後、採用端点での熱計算をあと1回だけ待つ。 */
    bool finalVerificationPending = false;
};

using CapacityBracketMap = std::unordered_map<std::string, CapacityBracket>;

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
    CapacityBracketMap& capacityLimitBracket,
    std::ostringstream& oss,
    bool& adjustmentMade);

void applyUnderCapacityBracketAdjustment(
    const std::string& airconKey,
    VertexProperties& nodeProps,
    OperationMode operationMode,
    double indoorTemp,
    double airFlowRate,
    double maxHeatCapacity,
    double currentTotal,
    CapacityBracketMap& capacityLimitBracket,
    std::ostringstream& oss,
    bool& adjustmentMade);

} // namespace aircon::capacity
