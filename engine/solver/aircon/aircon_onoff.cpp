#include "aircon/aircon_onoff.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

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

namespace {

bool demandKeepsOn(const std::string& mode,
                   double requiredHeatW,
                   double qOn,
                   bool useMinCapacity) {
    if (mode == "HEATING") {
        return useMinCapacity ? (requiredHeatW >= qOn) : (requiredHeatW > qOn);
    }
    if (mode == "COOLING") {
        return useMinCapacity ? (requiredHeatW <= -qOn) : (requiredHeatW < -qOn);
    }
    if (mode == "AUTO") {
        return useMinCapacity ? (std::abs(requiredHeatW) >= qOn)
                              : (std::abs(requiredHeatW) > qOn);
    }
    throw std::runtime_error("エアコンのモードが不正です: " + mode);
}

Decision decideFromRequiredHeat(const std::string& mode,
                                double requiredHeatW,
                                double loadDeadbandW,
                                double minProcessHeatW,
                                bool holdAtMinimumCapacity) {
    const double qTol = std::max(0.0, loadDeadbandW);
    const bool useMinCapacity = std::isfinite(minProcessHeatW) && minProcessHeatW > qTol;
    const double qOn = useMinCapacity ? minProcessHeatW : qTol;

    Decision decision;
    if (holdAtMinimumCapacity && useMinCapacity) {
        decision.shouldBeOn = true;
    } else {
        decision.shouldBeOn = demandKeepsOn(mode, requiredHeatW, qOn, useMinCapacity);
    }

    std::ostringstream detail;
    detail << "Qreq=" << requiredHeatW << "W";
    if (holdAtMinimumCapacity && useMinCapacity) {
        detail << " < Q.min=" << qOn << "W, 再起動と共存のため最低能力で継続";
    } else if (useMinCapacity && !decision.shouldBeOn) {
        detail << " < Q.min=" << qOn << "W";
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
                bool holdAtMinimumCapacity) {
    if (useRequiredHeat && currentlyOn && std::isfinite(requiredHeatW)) {
        return decideFromRequiredHeat(mode, requiredHeatW, loadDeadbandW, minProcessHeatW,
                                      holdAtMinimumCapacity);
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
