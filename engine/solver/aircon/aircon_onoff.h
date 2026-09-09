#pragma once

#include <string>

namespace aircon::onoff {

// 停止後の再起動ヒステリシス。数値 tol が極小でも 1 K 未満では再起動しない。
constexpr double kRestartBandK = 1.0;
constexpr double kLoadDeadbandW = 1.0;

struct Decision {
    bool shouldBeOn = false;
    std::string detail;
};

double restartBandK(double tolerance);

// 暖房は要求より十分低い、冷房は十分高いときに再起動する。AUTO は幅の外なら再起動。
bool temperatureWouldRestart(const std::string& mode,
                             double freeTempC,
                             double targetTempC,
                             double bandK);

// Q.min を引くモードキー。AUTO は符号付き負荷が負なら冷却。
const char* minCapacityModeKey(const std::string& currentMode, double requiredHeatW);

// 設定維持中の室負荷、または停止中の温度バンドで ON/OFF を決める。
Decision decide(const std::string& mode,
                bool currentlyOn,
                double currentTemp,
                double targetTemp,
                double tolerance,
                bool useRequiredHeat,
                double requiredHeatW,
                double loadDeadbandW,
                double minProcessHeatW,
                bool holdAtMinimumCapacity);

std::string formatLog(const std::string& targetName,
                      bool currentlyOn,
                      bool shouldBeOn,
                      double targetTemp,
                      const std::string& detail);

} // namespace aircon::onoff
