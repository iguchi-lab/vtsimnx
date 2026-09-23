#pragma once

#include <optional>
#include <string>

namespace aircon::onoff {

// 停止後の再起動ヒステリシス。数値 tol が極小でも 1 K 未満では再起動しない。
constexpr double kRestartBandK = 1.0;
constexpr double kLoadDeadbandW = 1.0;
// 処理風量がこれ以下ならコイル由来の Qreq を信頼しない [m3/s]。
constexpr double kMinMeaningfulProcessFlow = 1e-4;

struct Decision {
    bool shouldBeOn = false;
    std::string detail;
};

// Qreq を ON/OFF に使うための信頼条件（いずれも満たす必要あり）。
struct RequiredHeatTrust {
    bool nearEffectiveSetpoint = false;   // |T_set - T_eff| <= restart band
    bool capacityLimited = false;         // 能力制限中は Qreq で止めない
    bool effectiveDetunedFromRequest = false;  // 実効設定が要求から外れている
    bool processFlowTooLow = false;       // ほぼ無風ではコイル熱が信用できない
};

double restartBandK(double tolerance);

// 暖房は要求より十分低い、冷房は十分高いときに再起動する。AUTO は幅の外なら再起動。
bool temperatureWouldRestart(const std::string& mode,
                             double freeTempC,
                             double targetTempC,
                             double bandK);

// Q.min / min_process を引くモードキー。AUTO は符号付き負荷が負なら冷却。
const char* minCapacityModeKey(const std::string& currentMode, double requiredHeatW);

// 制御室温・設定・能力状態・任意の処理風量から信頼条件を組み立てる。
// processFlowM3s が nullopt のときは風量不足判定をしない（現状どおり false）。
RequiredHeatTrust buildRequiredHeatTrust(double controlledRoomTempC,
                                         double effectivePreTempC,
                                         double requestedPreTempC,
                                         double setpointBandK,
                                         bool capacityLimited,
                                         std::optional<double> processFlowM3s = std::nullopt);

// ON 中かつ Qreq 有限で、信頼条件をすべて満たすときだけ負荷ベース ON/OFF を使う。
bool shouldUseRequiredHeat(bool currentlyOn,
                           double requiredHeatW,
                           const RequiredHeatTrust& trust);

// AirconSpec::getMinProcess の戻り [kW] を制御用 W へ。無効なら 0。
double resolveMinProcessHeatW(std::optional<double> qMinKW);

// 同符号の小負荷だけ最低能力ホールドする。
// 暖房/加熱側: deadband < Qreq <= minProcess
// 冷房側:     -minProcess <= Qreq < -deadband
// minModeKey は "heating" / "cooling"（minCapacityModeKey の戻り）。
bool inMinProcessHoldBand(const char* minModeKey,
                          double requiredHeatW,
                          double minProcessHeatW,
                          double deadbandW = kLoadDeadbandW);

// ホールド時に required_heat_w へ書き込む符号付き最低能力 [W]。
double signedMinProcessHeatW(const char* minModeKey, double minProcessHeatW);

// 設定維持中の室負荷、または停止中の温度バンドで ON/OFF を決める。
// minProcessHeatW はカタログ Q.min ではなく制御用 min_process [W]。
// holdAtMinimumCapacity のとき、同符号の小負荷なら停止せず最低能力で処理する。
// 逆符号の要求負荷では hold を無視して OFF にする。
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
