#pragma once

#include <optional>
#include <string>

namespace aircon::onoff {

// 停止後の再起動ヒステリシス。数値 tol が極小でも 1 K 未満では再起動しない。
constexpr double kRestartBandK = 1.0;
constexpr double kLoadDeadbandW = 1.0;
// 最低能力ホールド。入るのは 1.2 倍。一度入ったら、同符号でデッドバンド超なら
// その時間ステップの終わりまで出ない（倍率では出ない）。
// Qreq は風量を落とすと増えるため、1.5 倍でも 851 W ↔ 1254 W が縁をまたぐ。
constexpr double kMinProcessHoldRatio = 1.2;
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
// 入る: |Qreq| <= minProcess * kMinProcessHoldRatio（かつデッドバンド超）
// 維持: previouslyHeld なら同符号でデッドバンド超なら |Qreq| の大きさによらず維持
//       （ほぼ零・逆符号は出る。フラグはタイムステップ先頭で落ちる）
// 暖房は正、冷房は負。minModeKey は "heating" / "cooling"。
inline double minProcessHoldCeilingW(double minProcessHeatW) {
    return minProcessHeatW * kMinProcessHoldRatio;
}

bool inMinProcessHoldBand(const char* minModeKey,
                          double requiredHeatW,
                          double minProcessHeatW,
                          double deadbandW = kLoadDeadbandW,
                          bool previouslyHeld = false);

// DUCT以外のホールド時に required_heat_w へ書き込む符号付き最低能力 [W]。
double signedMinProcessHeatW(const char* minModeKey, double minProcessHeatW);

// 設定維持中の室負荷、または停止中の温度バンドで ON/OFF を決める。
// minProcessHeatW はカタログ Q.min ではなく制御用 min_process [W]。
// holdAtMinimumCapacity のとき、同符号の小負荷なら停止せず運転を維持する。
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
                bool holdAtMinimumCapacity,
                bool previouslyHeldAtMinimum = false);

std::string formatLog(const std::string& targetName,
                      bool currentlyOn,
                      bool shouldBeOn,
                      double targetTemp,
                      const std::string& detail);

} // namespace aircon::onoff
