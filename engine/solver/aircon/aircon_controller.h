#pragma once

#include "vtsim_solver.h"
#include "aircon/aircon_operation_mode.h"
#include "aircon/aircon_capacity.h"
#include "types/aircon_control_state.h"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <utility>

#include "acmodel/acmodel.h"

// 前方宣言
class ThermalNetwork;
class VentilationNetwork;

/**
 * @brief エアコン制御の結果を表す構造体
 */
struct AirconControlResult {
    bool stateChanged;
    bool on;
    std::string logMessage;
};

/**
 * @brief エアコンバリデーションデータを表す構造体
 */
struct AirconValidationData {
    double outdoorTemp;
    double indoorTemp;
    double airconTemp;
    double setTemp;
    // 湿度（絶対湿度）: calc_flag.x が有効な場合に使用（無効時は0でよい）
    double outdoorX = 0.0;
    double indoorX = 0.0;
};

/**
 * @brief エアコン関連の処理を管理するクラス
 */
class AirconController {
private:
    // エアコンモデルを保存するマップ (エアコンキー -> acmodelインスタンス)
    std::unordered_map<std::string, std::unique_ptr<acmodel::AirconSpec>> airconModels;
    int logVerbosity_ = 1;
    // moist_enthalpy_enabled: 処理熱量を mDot*Δh で評価する
    // checkAndAdjustCapacity 等の const API から同期するため mutable
    mutable bool moistEnthalpyEnabled_ = false;

    struct RuntimeContext {
        AirconValidationData validData;
        double heatCapacity = 0.0;
        double airFlowRate = 0.0;
        OperationMode operationMode = OperationMode::Cooling;
    };

    // 出力用：エアコンキー順キャッシュ（airconModels のキーを昇順で保持）
    mutable bool airconKeysCacheInitialized_ = false;
    mutable std::vector<std::string> airconKeysOrdered_;

    // 能力超過時 nullopt 用の二分探索 bracket（タイムステップごとにクリア）
    mutable aircon::capacity::CapacityBracketMap capacityLimitBracket_;

    // 外側連成の forceMinTwo 判定用（ON/OFF・mode 署名）。initializeModels でクリア。
    std::uint64_t prevAirconStateSig_ = 0;
    bool havePrevAirconStateSig_ = false;

    /**
     * @brief エアコンの基本データをバリデーションして取得する
     * @throws std::runtime_error 必要なデータが見つからない場合
     */
    AirconValidationData validateAirconData(const std::string& airconKey,
                                            ThermalNetwork& thermalNetwork,
                                            const VertexProperties& nodeProps) const;

    RuntimeContext prepareRuntimeContext(const std::string& airconKey,
                                         ThermalNetwork& thermalNetwork,
                                         const VertexProperties& nodeProps,
                                         const FlowRateMap& flowRates) const;

    /** 1台分の電力[W]とCOPを推定する。logDetail が true のときのみ WARN と詳細1行を出力。失敗時は例外。 */
    std::pair<double, double> estimatePowerAndCOPForAircon(const std::string& airconKey,
                                                           ThermalNetwork& thermalNetwork,
                                                           const VertexProperties& nodeProps,
                                                           const FlowRateMap& flowRates,
                                                           std::ostream& logs,
                                                           bool logDetail) const;

    // returnPower=true: power[W] を返す, false: COP を返す
    std::vector<double> calculatePowerOrCOPValues(ThermalNetwork& thermalNetwork,
                                                  const FlowRateMap& flowRates,
                                                  std::ostream& logs,
                                                  bool returnPower) const;

public:
    struct LatentFeedbackStats {
        double maxAppliedHeatW = 0.0;
    };
    // === モデル管理 ===
    void initializeModels(ThermalNetwork& thermalNetwork, std::ostream& logs, int logVerbosity);
    void setMoistEnthalpyEnabled(bool enabled) { moistEnthalpyEnabled_ = enabled; }
    bool moistEnthalpyEnabled() const { return moistEnthalpyEnabled_; }
    acmodel::AirconSpec* getModel(const std::string& airconKey) const;
    ~AirconController();

    // === テスト用（本番コードの挙動は変えない）===
    // モデルを外部から登録できる注入口。単体テストでモックモデルを差し込むために使用する。
    void registerModelForTesting(const std::string& airconKey, std::unique_ptr<acmodel::AirconSpec> model);
    void clearModelsForTesting();

    // === 計算関数 ===
    // エアコンの処理熱量（顕熱）を計算する。
    // - 暖房/冷房どちらでも「処理熱量は +W（大きさ）」として扱う（COP推定/能力チェック用）
    // - mode は "heating" / "cooling"（prepareRuntimeContext が決める）
    double calculateHeatCapacity(ThermalNetwork& thermalNetwork,
                                 const std::string& mode,
                                 const std::string& inNode,
                                 const std::string& airconNode,
                                 const FlowRateMap& flowRates) const;

    // === 制御関数 ===
    template<typename NodeType>
    AirconControlResult controlAircon(const NodeType& nodeProps, double currentTemp,
                                      double targetTemp, double tolerance, [[maybe_unused]] std::ostream& logs,
                                      bool useRequiredHeat = false,
                                      double requiredHeatW = 0.0,
                                      double loadDeadbandW = 1.0) const {
        AirconControlResult result{false, nodeProps.on, ""};
        const std::string targetName = nodeProps.set_node.empty() ? nodeProps.key : nodeProps.set_node;
        if (nodeProps.current_mode == "OFF") {
            if (nodeProps.on) {
                result.stateChanged = true;
                result.on = false;
            }
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン: モードOFFのため制御対象外"
                << " (現在 " << currentTemp << "°C, 目標 " << targetTemp << "°C)";
            result.logMessage = oss.str();
            return result;
        }

        bool shouldBeOn = false;
        std::ostringstream detail;

        // ON かつ fixed-row 後の必要負荷が取れるときは符号付き負荷で active-set を決める。
        // （固定後の室温は常に設定値付近なので温度比較では停止判定できない）
        // 符号: Qrequired>0 = 暖房需要、Qrequired<0 = 冷房需要
        if (useRequiredHeat && nodeProps.on && std::isfinite(requiredHeatW)) {
            const double qTol = std::max(0.0, loadDeadbandW);
            if (nodeProps.current_mode == "HEATING") {
                shouldBeOn = (requiredHeatW > qTol);
            } else if (nodeProps.current_mode == "COOLING") {
                shouldBeOn = (requiredHeatW < -qTol);
            } else if (nodeProps.current_mode == "AUTO") {
                shouldBeOn = (std::abs(requiredHeatW) > qTol);
            } else {
                throw std::runtime_error("エアコンのモードが不正です: " + nodeProps.current_mode);
            }
            detail << "Qreq=" << requiredHeatW << "W";
        } else {
            // OFF 中（室温が自由）または負荷未評価: 従来の温度バンド
            const double diff = currentTemp - targetTemp;
            const bool withinBand = (std::abs(diff) <= tolerance);
            if (nodeProps.current_mode == "HEATING") {
                if (withinBand) shouldBeOn = nodeProps.on;
                else shouldBeOn = (diff < 0.0);
            } else if (nodeProps.current_mode == "COOLING") {
                if (withinBand) shouldBeOn = nodeProps.on;
                else shouldBeOn = (diff > 0.0);
            } else if (nodeProps.current_mode == "AUTO") {
                shouldBeOn = withinBand ? nodeProps.on : true;
            } else {
                throw std::runtime_error("エアコンのモードが不正です: " + nodeProps.current_mode);
            }
            detail << "T=" << currentTemp << "°C";
        }

        if (shouldBeOn != nodeProps.on) {
            result.stateChanged = true;
            result.on = shouldBeOn;
            const char* transition = nodeProps.on ? "ON→OFF" : "OFF→ON";
            const char* action = result.on ? "起動" : "停止";
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン " << transition << " (" << action << ")"
                << " : " << detail.str()
                << ", 目標 " << targetTemp << "°C";
            result.logMessage = oss.str();
        } else {
            std::ostringstream oss;
            oss << "　" << targetName << " エアコン: "
                << (shouldBeOn ? "運転継続" : "停止維持")
                << " (" << detail.str()
                << ", 目標 " << targetTemp << "°C)";
            result.logMessage = oss.str();
        }
        return result;
    }

    bool controlAllAircons(ThermalNetwork& thermalNetwork,
                           double tolerance,
                           std::ostream& logFile,
                           bool* supplyHumidityChanged = nullptr,
                           double humidityAbsTol = 1e-9,
                           std::vector<AirconStateProposal>* outProposals = nullptr) const;

    bool checkAndAdjustCapacity(ThermalNetwork& thermalNetwork, VentilationNetwork& ventNetwork,
                                const SimulationConstants& constants,
                                const FlowRateMap& flowRates,
                                std::ostream& logFile,
                                int& totalIterations,
                                bool* supplyHumidityChanged = nullptr,
                                double humidityAbsTol = 1e-9,
                                std::vector<AirconStateProposal>* outProposals = nullptr) const;

    // DUCT_CENTRAL 用: 処理熱量に応じて送風量を補正する。
    // - 処理熱量=0 -> 風量=0
    // - 処理熱量=Q.rtd -> 風量=V_inner.dsgn
    // 変更が入った場合は true（外側ループで再計算要求）。
    bool checkAndAdjustDuctCentralAirflow(ThermalNetwork& thermalNetwork,
                                          VentilationNetwork& ventNetwork,
                                          const FlowRateMap& flowRates,
                                          std::ostream& logs,
                                          bool* supplyHumidityChanged = nullptr,
                                          double humidityAbsTol = 1e-9,
                                          std::vector<AirconStateProposal>* outProposals = nullptr) const;

    // 潜熱処理量を熱方程式の heat_source へ反映する（冷房時のみ有効）。
    // relaxation=1.0 で全量反映。<1.0 で緩和反映。
    LatentFeedbackStats applyLatentFeedbackToThermal(
        ThermalNetwork& thermalNetwork,
        const FlowRateMap& flowRates,
        double relaxation,
        std::ostream& logs) const;

    // === データ収集・ログ ===
    const std::vector<std::string>& getAirconKeys() const;

    std::vector<double> collectAirconDataValues(ThermalNetwork& thermalNetwork,
                                                const FlowRateMap& flowRates,
                                                const std::string& dataType) const;

    std::vector<double> calculatePowerValues(ThermalNetwork& thermalNetwork,
                                             const FlowRateMap& flowRates,
                                             std::ostream& logFile) const;

    std::vector<double> calculateCOPValues(ThermalNetwork& thermalNetwork,
                                           const FlowRateMap& flowRates,
                                           std::ostream& logFile) const;

    // === 設定 ===
    void applyPreset(ThermalNetwork& thermalNetwork, std::ostream& logFile) const;

    /** 能力超過時の二分探索 bracket をクリア（タイムステップ先頭で呼ぶ） */
    void clearCapacityLimitBracket() const;

    /** 空調署名ウォームスタート状態をクリア（モデル再初期化時） */
    void clearCouplingWarmStart();
    /** 現署名を記録（タイムステップ終了時） */
    void observeAirconStateSignature(std::uint64_t sig);
    /** 初回または署名変化時は true（内側連成の最低2回を要求） */
    bool shouldForceMinTwoCouplingIters(std::uint64_t currentSig) const;
};


