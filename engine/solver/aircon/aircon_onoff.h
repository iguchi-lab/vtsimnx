#pragma once

#include <optional>
#include <string>

namespace aircon::onoff {

// 停止後の再起動ヒステリシス床（既定）。数値 tol が極小でもこの幅未満では再起動しない。
// simulation.tolerance.aircon_restart_band で上書き可。ideal_room_load では床 0 を使う。
constexpr double kRestartBandK = 1.0;
constexpr double kLoadDeadbandW = 1.0;
// 最低能力ホールド。入るのは 1.2 倍。一度入ったら、同符号でデッドバンド超なら
// その時間ステップの終わりまで出ない（倍率では出ない）。
// Qreq は風量を落とすと増えるため、1.5 倍でも 851 W ↔ 1254 W が縁をまたぐ。
// ※ ON/OFF 判定そのものには使わない。required_heat_w 上書きと風量下限用。
constexpr double kMinProcessHoldRatio = 1.2;
// 処理風量がこれ以下ならコイル由来の Qreq を信頼しない [m3/s]。
constexpr double kMinMeaningfulProcessFlow = 1e-4;

struct Decision {
    bool shouldBeOn = false;
    std::string detail;
};

// Qreq を ON/OFF に使うための信頼条件（いずれも満たす必要あり）。
struct RequiredHeatTrust {
    bool nearEffectiveSetpoint = false;   // |T_set - T_eff| <= trust band
    bool capacityLimited = false;         // 能力制限中は Qreq で止めない
    bool effectiveDetunedFromRequest = false;  // 実効設定が要求から外れている
    bool processFlowTooLow = false;       // ほぼ無風ではコイル熱が信用できない
};

// 機種・モードごとの ON/OFF 方針（ideal 特例をここに集約）。
struct OnOffPolicy {
    bool idealRoomLoad = false;
    double restartBandFloorK = kRestartBandK;
    double trustBandK = kRestartBandK;
    bool trustUsesProcessFlow = true;
    bool allowMinProcess = true;

    // temperatureTolK = ε_T,ac。configuredRestartFloorK = aircon_restart_band（ideal は床 0）。
    static OnOffPolicy make(bool idealRoomLoad,
                            double temperatureTolK,
                            double configuredRestartFloorK);
};

// ΔT_restart = max(tolerance, floorK)。floorK 既定は kRestartBandK。
double restartBandK(double tolerance, double floorK = kRestartBandK);

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

// 同符号の小負荷だけ最低能力ホールドする（風量下限・Qreq 上書き用。ON/OFF 判定では使わない）。
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

// 最低能力ホールドの計画（ON/OFF 後の Qreq 上書き・フラグ更新用）。
struct MinProcessPlan {
    const char* modeKey = "heating";
    double heatW = 0.0;
    bool hold = false;
};

// policy.allowMinProcess が false、または minProcessKW 無効なら hold=false。
MinProcessPlan planMinProcess(const OnOffPolicy& policy,
                              const std::string& currentMode,
                              double requiredHeatW,
                              std::optional<double> minProcessKW,
                              bool previouslyHeld,
                              double deadbandW = kLoadDeadbandW);

// humidistat 冷房要求（正の冷却量 W）向け。modeKey は常に "cooling"。
MinProcessPlan planMinProcessForCoolingDemand(double coolingDemandW,
                                              std::optional<double> minProcessKW,
                                              bool previouslyHeld,
                                              double deadbandW = kLoadDeadbandW);

// 通常機種・非 humidistat・非 DUCT で hold 中なら required_heat を置き換えてよい。
bool shouldOverwriteRequiredHeat(const OnOffPolicy& policy,
                                 const MinProcessPlan& plan,
                                 bool humidityControlEnabled,
                                 bool isDuctCentral,
                                 bool decidedOn);

// 設定維持中の室負荷、または停止中の温度バンドで ON/OFF を決める。
// 負荷経路の継続条件はデッドバンドのみ（最低能力ホールドはここでは見ない）。
Decision decide(const std::string& mode,
                bool currentlyOn,
                double currentTemp,
                double targetTemp,
                double tolerance,
                bool useRequiredHeat,
                double requiredHeatW,
                double loadDeadbandW = kLoadDeadbandW,
                double restartBandFloorK = kRestartBandK);

std::string formatLog(const std::string& targetName,
                      bool currentlyOn,
                      bool shouldBeOn,
                      double targetTemp,
                      const std::string& detail);

} // namespace aircon::onoff
