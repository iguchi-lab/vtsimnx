#include "aircon/aircon_capacity.h"

#include "archenv/include/archenv.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kAirDensity = archenv::DENSITY_DRY_AIR;         // [kg/m^3]
constexpr double kAirSpecificHeat = archenv::SPECIFIC_HEAT_AIR;   // [J/(kg·K)]
constexpr int kSetpointSearchMaxIterations = 32;
constexpr double kSetpointSearchTolerance = 1e-3;                 // [degC]
constexpr double kSetpointFloor = 0.0;
constexpr double kSetpointCeiling = 50.0;
constexpr double kCapacityLimitInitialBracketWidth = 10.0;        // [degC]
constexpr double kCapacityConvergenceRelTol = 0.001;
constexpr double kCapacityConvergenceAbsTol = 1.0;                // [W]

inline double clampHeatCapacity(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

inline double estimateHeatCapacityForSetpoint(OperationMode operationMode,
                                              double inletTemp,
                                              double setpoint,
                                              double airFlowRate) {
    if (std::abs(airFlowRate) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    double deltaT = 0.0;
    if (isHeating(operationMode)) {
        deltaT = std::max(0.0, setpoint - inletTemp);
    } else {
        deltaT = std::max(0.0, inletTemp - setpoint);
    }
    return clampHeatCapacity(kAirDensity * kAirSpecificHeat * std::abs(airFlowRate) * deltaT);
}

inline std::optional<double> findCapacityLimitedSetpoint(OperationMode operationMode,
                                                         double inletTemp,
                                                         double currentSetpoint,
                                                         double airFlowRate,
                                                         double maxHeatCapacity) {
    if (!std::isfinite(inletTemp) || !std::isfinite(currentSetpoint) || !std::isfinite(maxHeatCapacity)) {
        return std::nullopt;
    }

    if (isHeating(operationMode)) {
        double feasible = std::min(inletTemp, currentSetpoint);
        double infeasible = std::max(inletTemp, currentSetpoint);
        if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, infeasible, airFlowRate) <= maxHeatCapacity) {
            return std::nullopt;
        }
        for (int i = 0; i < kSetpointSearchMaxIterations && (infeasible - feasible) > kSetpointSearchTolerance; ++i) {
            const double mid = 0.5 * (feasible + infeasible);
            if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, mid, airFlowRate) <= maxHeatCapacity) {
                feasible = mid;
            } else {
                infeasible = mid;
            }
        }
        return feasible;
    }

    double infeasible = std::min(inletTemp, currentSetpoint);
    double feasible = std::max(inletTemp, currentSetpoint);
    if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, infeasible, airFlowRate) <= maxHeatCapacity) {
        return std::nullopt;
    }
    for (int i = 0; i < kSetpointSearchMaxIterations && (feasible - infeasible) > kSetpointSearchTolerance; ++i) {
        const double mid = 0.5 * (feasible + infeasible);
        if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, mid, airFlowRate) <= maxHeatCapacity) {
            feasible = mid;
        } else {
            infeasible = mid;
        }
    }
    return feasible;
}

struct CapacityLimitBracketResult {
    double newSetpoint = 0.0;
    bool bracketConverged = false;
    bool capacityConverged = false;
};

void initCapacityLimitBracket(bool heating, double currentPreTemp, double& tLow, double& tHigh) {
    const double w = kCapacityLimitInitialBracketWidth;
    if (heating) {
        tLow = std::max(kSetpointFloor, currentPreTemp - w);
        tHigh = currentPreTemp;
    } else {
        tLow = currentPreTemp;
        tHigh = std::min(kSetpointCeiling, currentPreTemp + w);
    }
}

CapacityLimitBracketResult stepCapacityLimitBracket(bool heating, double maxQ, double currentQ,
                                                    double currentSetpoint,
                                                    double& tLow, double& tHigh) {
    if (currentQ > maxQ) {
        if (heating) {
            tHigh = currentSetpoint;
        } else {
            tLow = currentSetpoint;
        }
    } else {
        if (heating) {
            tLow = currentSetpoint;
        } else {
            tHigh = currentSetpoint;
        }
    }
    const double newSetpoint = 0.5 * (tLow + tHigh);
    const bool bracketConverged = (tHigh - tLow) <= kSetpointSearchTolerance;
    const bool capacityConverged =
        std::abs(currentQ - maxQ) <= (maxQ * kCapacityConvergenceRelTol + kCapacityConvergenceAbsTol);
    return {newSetpoint, bracketConverged, capacityConverged};
}

std::pair<double, double>& ensureCapacityLimitBracket(
    std::unordered_map<std::string, std::pair<double, double>>& brackets,
    const std::string& airconKey,
    bool heating,
    double currentPreTemp) {
    auto it = brackets.find(airconKey);
    if (it == brackets.end()) {
        std::pair<double, double> bracket;
        initCapacityLimitBracket(heating, currentPreTemp, bracket.first, bracket.second);
        brackets[airconKey] = bracket;
        it = brackets.find(airconKey);
    }
    return it->second;
}

} // namespace

namespace aircon::capacity {

std::optional<double> resolveMaxHeatCapacity(const VertexProperties& nodeProps,
                                             OperationMode operationMode,
                                             std::string& source) {
    const std::string mode = modeKey(operationMode);
    if (const auto* spec = nodeProps.getAirconSpec()) {
        auto value = spec->getCapacityMaxForMode(mode);
        if (value && *value > 0) {
            source = "Q." + mode + ".max";
            return *value * 1000.0;
        }
    }
    return std::nullopt;
}

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
    bool& adjustmentMade) {
    auto limitedSetpoint = findCapacityLimitedSetpoint(
        operationMode,
        indoorTemp,
        nodeProps.current_pre_temp,
        airFlowRate,
        maxHeatCapacity);
    const double previousSetpoint = nodeProps.current_pre_temp;
    if (limitedSetpoint) {
        nodeProps.current_pre_temp = *limitedSetpoint;
        adjustmentMade = true;
        oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << *limitedSetpoint << "°C";
        oss << ", 再計算要求";
        return;
    }

    const bool heating = isHeating(operationMode);
    auto& bracket = ensureCapacityLimitBracket(
        capacityLimitBracket, airconKey, heating, nodeProps.current_pre_temp);
    double& tLow = bracket.first;
    double& tHigh = bracket.second;
    const auto result = stepCapacityLimitBracket(heating, maxHeatCapacity, currentTotal,
                                                 nodeProps.current_pre_temp, tLow, tHigh);
    nodeProps.current_pre_temp = result.newSetpoint;
    if (result.capacityConverged) {
        oss << " → 二分探索収束 設定温度=" << result.newSetpoint << "°C（処理熱量≒最大能力）";
    } else if (result.bracketConverged) {
        adjustmentMade = true;
        oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint
            << "°C（bracket収束・最終解のため再計算1回）";
    } else {
        adjustmentMade = true;
        oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint
            << "°C（能力=" << maxHeatCapacity << "Wに合わせて二分探索）";
        oss << ", 再計算要求";
    }
}

void applyUnderCapacityBracketAdjustment(
    const std::string& airconKey,
    VertexProperties& nodeProps,
    OperationMode operationMode,
    double maxHeatCapacity,
    double currentTotal,
    std::unordered_map<std::string, std::pair<double, double>>& capacityLimitBracket,
    std::ostringstream& oss,
    bool& adjustmentMade) {
    const double previousSetpoint = nodeProps.current_pre_temp;
    oss << " → 不足（能力=" << maxHeatCapacity << "Wに合わせて二分探索継続）";
    const bool heating = isHeating(operationMode);
    auto& bracket = ensureCapacityLimitBracket(
        capacityLimitBracket, airconKey, heating, nodeProps.current_pre_temp);
    double& tLow = bracket.first;
    double& tHigh = bracket.second;
    const auto result = stepCapacityLimitBracket(heating, maxHeatCapacity, currentTotal,
                                                 nodeProps.current_pre_temp, tLow, tHigh);
    nodeProps.current_pre_temp = result.newSetpoint;
    if (result.capacityConverged) {
        oss << ", 二分探索収束 設定温度=" << result.newSetpoint << "°C（処理熱量≒最大能力）";
    } else if (result.bracketConverged) {
        adjustmentMade = true;
        oss << ", 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint
            << "°C（bracket収束・最終解のため再計算1回）";
    } else {
        adjustmentMade = true;
        oss << ", 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint << "°C, 再計算要求";
    }
}

} // namespace aircon::capacity
