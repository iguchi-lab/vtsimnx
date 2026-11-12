#include "aircon/aircon_onoff.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aircon::onoff {

double restartBandK(double tolerance) {
    return std::max(tolerance, kRestartBandK);
}

bool temperatureWouldRestart(const std::string& mode,
                             double freeTempC,
                             double targetTempC,
                             double bandK) {
    if (!std::isfinite(freeTempC) || !std::isfinite(targetTempC)) return false;
    const double diff = freeTempC - targetTempC;
    if (std::abs(diff) <= bandK) return false;
    if (mode == "COOLING" || mode == "cooling") return diff > 0.0;
    if (mode == "AUTO") return true;
    return diff < 0.0;
}

const char* minCapacityModeKey(const std::string& currentMode, double requiredHeatW) {
    if (currentMode == "COOLING") return "cooling";
    if (currentMode == "AUTO" && requiredHeatW < 0.0) return "cooling";
    return "heating";
}

RequiredHeatTrust buildRequiredHeatTrust(double controlledRoomTempC,
                                         double effectivePreTempC,
                                         double requestedPreTempC,
                                         double setpointBandK,
                                         bool capacityLimited,
                                         std::optional<double> processFlowM3s) {
    RequiredHeatTrust trust;
    trust.nearEffectiveSetpoint =
        std::isfinite(effectivePreTempC) &&
        std::abs(controlledRoomTempC - effectivePreTempC) <= setpointBandK;
    trust.capacityLimited = capacityLimited;
    trust.effectiveDetunedFromRequest =
        std::isfinite(effectivePreTempC) && std::isfinite(requestedPreTempC) &&
        std::abs(effectivePreTempC - requestedPreTempC) > setpointBandK;
    if (processFlowM3s.has_value()) {
        trust.processFlowTooLow =
            !(*processFlowM3s > kMinMeaningfulProcessFlow);
    }
    return trust;
}

bool shouldUseRequiredHeat(bool currentlyOn,
                           double requiredHeatW,
                           const RequiredHeatTrust& trust) {
    return currentlyOn && std::isfinite(requiredHeatW) && trust.nearEffectiveSetpoint &&
           !trust.capacityLimited && !trust.effectiveDetunedFromRequest &&
           !trust.processFlowTooLow;
}

double resolveMinProcessHeatW(std::optional<double> qMinKW) {
    if (!qMinKW.has_value()) return 0.0;
    if (!std::isfinite(*qMinKW) || !(*qMinKW > 0.0)) return 0.0;
    return (*qMinKW) * 1000.0;
}

bool inMinProcessHoldBand(const char* minModeKey,
                          double requiredHeatW,
                          double minProcessHeatW,
                          double deadbandW,
                          bool previouslyHeld) {
    if (!std::isfinite(requiredHeatW) || !std::isfinite(minProcessHeatW)) return false;
    const double qTol = std::max(0.0, deadbandW);
    if (!(minProcessHeatW > qTol)) return false;
    const bool cooling = (minModeKey != nullptr && std::string(minModeKey) == "cooling");
    // 一度入ったら、同符号でデッドバンド超ならステップ内は出ない。
    // 風量を落とすと Qreq が増えるため、倍率の出口は固定点にならない。
    if (previouslyHeld) {
        return cooling ? (requiredHeatW < -qTol) : (requiredHeatW > qTol);
    }
    const double holdCeil = minProcessHoldCeilingW(minProcessHeatW);
    if (cooling) {
        return requiredHeatW < -qTol && requiredHeatW >= -holdCeil;
    }
    return requiredHeatW > qTol && requiredHeatW <= holdCeil;
}

double signedMinProcessHeatW(const char* minModeKey, double minProcessHeatW) {
    const bool cooling = (minModeKey != nullptr && std::string(minModeKey) == "cooling");
    return cooling ? -minProcessHeatW : minProcessHeatW;
}

namespace {

bool demandKeepsOn(const std::string& mode,
                   double requiredHeatW,
                   double qOn) {
    if (mode == "HEATING") {
        return requiredHeatW > qOn;
    }
    if (mode == "COOLING") {
        return requiredHeatW < -qOn;
    }
    if (mode == "AUTO") {
        return std::abs(requiredHeatW) > qOn;
    }
    throw std::runtime_error("エアコンのモードが不正です: " + mode);
}

Decision decideFromRequiredHeat(const std::string& mode,
                                double requiredHeatW,
                                double loadDeadbandW,
                                double minProcessHeatW,
                                bool holdAtMinimumCapacity,
                                bool previouslyHeldAtMinimum) {
    const double qTol = std::max(0.0, loadDeadbandW);
    const char* minMode = minCapacityModeKey(mode, requiredHeatW);
    const bool demandOn = demandKeepsOn(mode, requiredHeatW, qTol);
    // 逆符号・ほぼ零では hold を無視して OFF/通常判定へ。同符号ならステップ内は維持。
    const bool holdApplies =
        holdAtMinimumCapacity &&
        inMinProcessHoldBand(minMode, requiredHeatW, minProcessHeatW, qTol,
                             previouslyHeldAtMinimum);

    Decision decision;
    decision.shouldBeOn = holdApplies ? true : demandOn;

    std::ostringstream detail;
    detail << "Qreq=" << requiredHeatW << "W";
    if (holdApplies) {
        detail << " <= min_process=" << minProcessHeatW << "W, 最低能力で処理";
    }
    decision.detail = detail.str();
    return decision;
}

Decision decideFromTemperature(const std::string& mode,
                               bool currentlyOn,
                               double currentTemp,
                               double targetTemp,
                               double tolerance) {
    const double bandK = restartBandK(tolerance);
    const double diff = currentTemp - targetTemp;
    const bool withinBand = std::abs(diff) <= bandK;

    Decision decision;
    if (mode == "HEATING") {
        decision.shouldBeOn = withinBand ? currentlyOn : (diff < 0.0);
    } else if (mode == "COOLING") {
        decision.shouldBeOn = withinBand ? currentlyOn : (diff > 0.0);
    } else if (mode == "AUTO") {
        decision.shouldBeOn = withinBand ? currentlyOn : true;
    } else {
        throw std::runtime_error("エアコンのモードが不正です: " + mode);
    }

    std::ostringstream detail;
    detail << "T=" << currentTemp << "°C";
    decision.detail = detail.str();
    return decision;
}

} // namespace

Decision decide(const std::string& mode,
                bool currentlyOn,
                double currentTemp,
                double targetTemp,
                double tolerance,
                bool useRequiredHeat,
                double requiredHeatW,
                double loadDeadbandW,
                double minProcessHeatW,
                bool holdAtMinimumCapacity,
                bool previouslyHeldAtMinimum) {
    if (useRequiredHeat && currentlyOn && std::isfinite(requiredHeatW)) {
        return decideFromRequiredHeat(mode, requiredHeatW, loadDeadbandW, minProcessHeatW,
                                      holdAtMinimumCapacity, previouslyHeldAtMinimum);
    }
    return decideFromTemperature(mode, currentlyOn, currentTemp, targetTemp, tolerance);
}

std::string formatLog(const std::string& targetName,
                      bool currentlyOn,
                      bool shouldBeOn,
                      double targetTemp,
                      const std::string& detail) {
    std::ostringstream oss;
    if (shouldBeOn != currentlyOn) {
        const char* transition = currentlyOn ? "ON→OFF" : "OFF→ON";
        const char* action = shouldBeOn ? "起動" : "停止";
        oss << "　" << targetName << " エアコン " << transition << " (" << action << ")"
            << " : " << detail
            << ", 目標 " << targetTemp << "°C";
    } else {
        oss << "　" << targetName << " エアコン: "
            << (shouldBeOn ? "運転継続" : "停止維持")
            << " (" << detail
            << ", 目標 " << targetTemp << "°C)";
    }
    return oss.str();
}

} // namespace aircon::onoff
