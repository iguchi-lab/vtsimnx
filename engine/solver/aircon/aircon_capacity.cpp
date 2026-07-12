#include "aircon/aircon_capacity.h"

#include "archenv/include/archenv.h"
#include "simulation_error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {
constexpr double kAirDensity = archenv::DENSITY_DRY_AIR;         // [kg/m^3]
constexpr double kAirSpecificHeat = archenv::SPECIFIC_HEAT_AIR;   // [J/(kg·K)]
constexpr int kSetpointSearchMaxIterations = 32;
constexpr double kSetpointSearchTolerance = 1e-3;                 // [degC]
constexpr double kSetpointFloor = 0.0;
constexpr double kSetpointCeiling = 50.0;
constexpr double kCapacityLimitInitialBracketWidth = 10.0;        // [degC]
constexpr int kCapacityLimitMaxBracketExpansions = 8;
constexpr double kCapacityConvergenceRelTol = 0.001;
constexpr double kCapacityConvergenceAbsTol = 1.0;                // [W]

inline double clampHeatCapacity(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

inline double capacityTol(double maxQ) {
    return maxQ * kCapacityConvergenceRelTol + kCapacityConvergenceAbsTol;
}

inline bool capacityNearMax(double currentQ, double maxQ) {
    return std::abs(currentQ - maxQ) <= capacityTol(maxQ);
}

/** 最終検証用: わずかな超過も許容し、不足は成功扱い。 */
inline bool capacityNotExceeded(double currentQ, double maxQ) {
    return currentQ <= maxQ + capacityTol(maxQ);
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

inline double feasibleEndpoint(bool heating, double tLow, double tHigh) {
    return heating ? tLow : tHigh;
}

void initCapacityLimitBracket(bool heating,
                              double currentPreTemp,
                              double inletTemp,
                              double airFlowRate,
                              double maxHeatCapacity,
                              aircon::capacity::CapacityBracket& bracket) {
    double width = kCapacityLimitInitialBracketWidth;
    for (int expand = 0; expand <= kCapacityLimitMaxBracketExpansions; ++expand) {
        if (heating) {
            bracket.tLow = std::max(kSetpointFloor, currentPreTemp - width);
            bracket.tHigh = currentPreTemp;
            const double qFeasible =
                estimateHeatCapacityForSetpoint(OperationMode::Heating, inletTemp, bracket.tLow, airFlowRate);
            if (qFeasible <= maxHeatCapacity || bracket.tLow <= kSetpointFloor + 1e-12) {
                break;
            }
        } else {
            bracket.tLow = currentPreTemp;
            bracket.tHigh = std::min(kSetpointCeiling, currentPreTemp + width);
            const double qFeasible =
                estimateHeatCapacityForSetpoint(OperationMode::Cooling, inletTemp, bracket.tHigh, airFlowRate);
            if (qFeasible <= maxHeatCapacity || bracket.tHigh >= kSetpointCeiling - 1e-12) {
                break;
            }
        }
        width *= 2.0;
    }
    bracket.finalVerificationPending = false;
}

bool tryExpandCapacityBracket(bool heating,
                              double inletTemp,
                              double airFlowRate,
                              double maxHeatCapacity,
                              aircon::capacity::CapacityBracket& bracket) {
    const double prevWidth = std::max(0.0, bracket.tHigh - bracket.tLow);
    double width = std::max(kCapacityLimitInitialBracketWidth, prevWidth * 2.0);
    const double anchor = heating ? bracket.tHigh : bracket.tLow;  // 超過側の端
    bool expanded = false;
    for (int expand = 0; expand <= kCapacityLimitMaxBracketExpansions; ++expand) {
        const double oldLow = bracket.tLow;
        const double oldHigh = bracket.tHigh;
        if (heating) {
            bracket.tHigh = anchor;
            bracket.tLow = std::max(kSetpointFloor, anchor - width);
            const double qFeasible =
                estimateHeatCapacityForSetpoint(OperationMode::Heating, inletTemp, bracket.tLow, airFlowRate);
            expanded = (bracket.tLow < oldLow - 1e-12);
            if (qFeasible <= maxHeatCapacity || bracket.tLow <= kSetpointFloor + 1e-12) {
                break;
            }
        } else {
            bracket.tLow = anchor;
            bracket.tHigh = std::min(kSetpointCeiling, anchor + width);
            const double qFeasible =
                estimateHeatCapacityForSetpoint(OperationMode::Cooling, inletTemp, bracket.tHigh, airFlowRate);
            expanded = (bracket.tHigh > oldHigh + 1e-12);
            if (qFeasible <= maxHeatCapacity || bracket.tHigh >= kSetpointCeiling - 1e-12) {
                break;
            }
        }
        if (!expanded) {
            break;
        }
        width *= 2.0;
    }
    bracket.finalVerificationPending = false;
    return expanded || (bracket.tHigh - bracket.tLow) > prevWidth + 1e-12;
}

CapacityLimitBracketResult stepCapacityLimitBracket(bool heating, double maxQ, double currentQ,
                                                    double currentSetpoint,
                                                    double& tLow, double& tHigh) {
    // 現在点の処理熱量で先に収束判定する。
    if (capacityNearMax(currentQ, maxQ)) {
        return {currentSetpoint, /*bracketConverged=*/false, /*capacityConverged=*/true};
    }

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

    if (tHigh < tLow) {
        std::swap(tLow, tHigh);
    }

    const bool bracketConverged = (tHigh - tLow) <= kSetpointSearchTolerance;
    if (bracketConverged) {
        // 幅だけ収束したら「能力を超えない側」の端点を最終候補にする
        return {feasibleEndpoint(heating, tLow, tHigh), true, false};
    }
    const double newSetpoint = 0.5 * (tLow + tHigh);
    return {newSetpoint, false, false};
}

aircon::capacity::CapacityBracket& ensureCapacityLimitBracket(
    aircon::capacity::CapacityBracketMap& brackets,
    const std::string& airconKey,
    bool heating,
    double currentPreTemp,
    double inletTemp,
    double airFlowRate,
    double maxHeatCapacity) {
    auto it = brackets.find(airconKey);
    if (it == brackets.end()) {
        aircon::capacity::CapacityBracket bracket;
        initCapacityLimitBracket(heating, currentPreTemp, inletTemp, airFlowRate, maxHeatCapacity, bracket);
        brackets[airconKey] = bracket;
        it = brackets.find(airconKey);
    }
    return it->second;
}

void applyBracketStepResult(
    const std::string& airconKey,
    VertexProperties& nodeProps,
    bool heating,
    double maxHeatCapacity,
    double currentTotal,
    double previousSetpoint,
    double indoorTemp,
    double airFlowRate,
    aircon::capacity::CapacityBracketMap& capacityLimitBracket,
    aircon::capacity::CapacityBracket& bracket,
    const CapacityLimitBracketResult& result,
    std::ostringstream& oss,
    bool& adjustmentMade,
    bool underCapacityPath) {
    nodeProps.aircon_control_state = AirconControlState::CapacityLimited;

    // 最終検証待ちの再計算結果を評価する
    if (bracket.finalVerificationPending) {
        if (capacityNotExceeded(currentTotal, maxHeatCapacity)) {
            capacityLimitBracket.erase(airconKey);
            nodeProps.current_pre_temp = previousSetpoint;
            oss << (underCapacityPath ? ", " : " → ");
            oss << "bracket最終検証OK 設定温度=" << previousSetpoint
                << "°C（処理熱量=" << currentTotal << "W）";
            return;
        }
        // まだ超過: 可能なら bracket を広げて探索再開。広げられなければ打ち切り。
        if (tryExpandCapacityBracket(heating, indoorTemp, airFlowRate, maxHeatCapacity, bracket)) {
            const double mid = 0.5 * (bracket.tLow + bracket.tHigh);
            nodeProps.current_pre_temp = mid;
            adjustmentMade = true;
            oss << (underCapacityPath ? ", " : " → ");
            oss << "最終検証でも超過のため bracket 拡張 設定温度="
                << previousSetpoint << "→" << mid << "°C, 再計算要求";
            return;
        }
        capacityLimitBracket.erase(airconKey);
        nodeProps.current_pre_temp = previousSetpoint;
        oss << (underCapacityPath ? ", " : " → ");
        oss << "bracket最終検証でも能力超過のまま探索終了 設定温度=" << previousSetpoint
            << "°C（処理熱量=" << currentTotal << "W, 上限=" << maxHeatCapacity << "W）";
        throw simulation::Error(
            simulation::ErrorCode::CapacityConstraintUnresolved,
            "Aircon capacity constraint unresolved: aircon=" + airconKey +
                ", Q=" + std::to_string(currentTotal) + "W, Qmax=" + std::to_string(maxHeatCapacity) +
                "W, setpoint=" + std::to_string(previousSetpoint) + "C");
    }

    nodeProps.current_pre_temp = result.newSetpoint;
    const bool setpointChanged =
        std::abs(result.newSetpoint - previousSetpoint) > kSetpointSearchTolerance;
    if (setpointChanged) {
        adjustmentMade = true;
    }

    if (result.capacityConverged) {
        capacityLimitBracket.erase(airconKey);
        oss << (underCapacityPath ? ", " : " → ");
        oss << "二分探索収束 設定温度=" << result.newSetpoint << "°C（処理熱量≒最大能力）";
        if (setpointChanged) {
            oss << ", 再計算要求";
        }
        return;
    }

    if (result.bracketConverged) {
        // 幅収束: 能力非超過側端点を採用し、熱計算の最終確認をちょうど1回だけ要求する
        bracket.finalVerificationPending = true;
        adjustmentMade = true;
        oss << (underCapacityPath ? ", " : " → ");
        oss << "設定温度補正=" << previousSetpoint << "→" << result.newSetpoint
            << "°C（bracket幅収束・最終検証のため再計算1回）";
        return;
    }

    adjustmentMade = true;
    oss << (underCapacityPath ? ", " : " → ");
    if (underCapacityPath) {
        oss << "設定温度補正=" << previousSetpoint << "→" << result.newSetpoint << "°C, 再計算要求";
    } else {
        oss << "超過, 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint
            << "°C（能力=" << maxHeatCapacity << "Wに合わせて二分探索）, 再計算要求";
    }
}

} // namespace

namespace aircon::capacity {

std::optional<double> resolveMaxHeatCapacity(const VertexProperties& nodeProps,
                                             OperationMode operationMode,
                                             std::string& source) {
    const std::string mode = modeKey(operationMode);
    if (const auto* spec = nodeProps.getAirconSpec()) {
        auto maxV = spec->getCapacity(mode, "max");
        if (maxV && *maxV > 0) {
            source = "Q." + mode + ".max";
            return *maxV * 1000.0;
        }
        auto midV = spec->getCapacity(mode, "mid");
        if (midV && *midV > 0) {
            source = "Q." + mode + ".mid";
            return *midV * 1000.0;
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
    CapacityBracketMap& capacityLimitBracket,
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
        // 公式経路へ入る前に未完了の bracket を捨てる
        capacityLimitBracket.erase(airconKey);
        nodeProps.current_pre_temp = *limitedSetpoint;
        nodeProps.aircon_control_state = AirconControlState::CapacityLimited;
        adjustmentMade = true;
        oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << *limitedSetpoint << "°C";
        oss << ", 再計算要求";
        return;
    }

    const bool heating = isHeating(operationMode);
    auto& bracket = ensureCapacityLimitBracket(
        capacityLimitBracket, airconKey, heating, nodeProps.current_pre_temp,
        indoorTemp, airFlowRate, maxHeatCapacity);

    if (bracket.finalVerificationPending) {
        applyBracketStepResult(airconKey, nodeProps, heating, maxHeatCapacity, currentTotal,
                               previousSetpoint, indoorTemp, airFlowRate, capacityLimitBracket,
                               bracket, CapacityLimitBracketResult{}, oss, adjustmentMade,
                               /*underCapacityPath=*/false);
        return;
    }

    const auto result = stepCapacityLimitBracket(heating, maxHeatCapacity, currentTotal,
                                                 nodeProps.current_pre_temp, bracket.tLow, bracket.tHigh);
    applyBracketStepResult(airconKey, nodeProps, heating, maxHeatCapacity, currentTotal,
                           previousSetpoint, indoorTemp, airFlowRate, capacityLimitBracket,
                           bracket, result, oss, adjustmentMade, /*underCapacityPath=*/false);
}

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
    bool& adjustmentMade) {
    const double previousSetpoint = nodeProps.current_pre_temp;
    oss << " → 不足（能力=" << maxHeatCapacity << "Wに合わせて二分探索継続）";
    const bool heating = isHeating(operationMode);
    // under 経路では既存 bracket 前提。無い場合は何もしない（呼び出し側で count チェック済み）
    auto it = capacityLimitBracket.find(airconKey);
    if (it == capacityLimitBracket.end()) {
        return;
    }
    auto& bracket = it->second;

    if (bracket.finalVerificationPending) {
        applyBracketStepResult(airconKey, nodeProps, heating, maxHeatCapacity, currentTotal,
                               previousSetpoint, indoorTemp, airFlowRate, capacityLimitBracket,
                               bracket, CapacityLimitBracketResult{}, oss, adjustmentMade,
                               /*underCapacityPath=*/true);
        return;
    }

    const auto result = stepCapacityLimitBracket(heating, maxHeatCapacity, currentTotal,
                                                 nodeProps.current_pre_temp, bracket.tLow, bracket.tHigh);
    applyBracketStepResult(airconKey, nodeProps, heating, maxHeatCapacity, currentTotal,
                           previousSetpoint, indoorTemp, airFlowRate, capacityLimitBracket,
                           bracket, result, oss, adjustmentMade, /*underCapacityPath=*/true);
}

} // namespace aircon::capacity
