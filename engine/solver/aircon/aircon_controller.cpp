#include "aircon/aircon_controller.h"
#include "aircon/aircon_capacity.h"
#include "aircon/aircon_airflow.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_network_utils.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "archenv/include/archenv.h"
#include "core/thermal/thermal_moist_air.h"
#include "utils/utils.h"
#include "acmodel/acmodel.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/range/iterator_range.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {
constexpr double kAirDensity = archenv::DENSITY_DRY_AIR;         // [kg/m^3]
constexpr double kAirSpecificHeat = archenv::SPECIFIC_HEAT_AIR;   // [J/(kg·K)]

inline double clampHeatCapacity(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

} // namespace

AirconController::~AirconController() = default;

void AirconController::initializeModels(ThermalNetwork& thermalNetwork,
                                        std::ostream& logs,
                                        int logVerbosity) {
    logVerbosity_ = logVerbosity;
    airconModels.clear();
    airconKeysCacheInitialized_ = false;
    airconKeysOrdered_.clear();
    lastAppliedMode_.clear();
    clearCouplingWarmStart();
    clearCapacityLimitBracket();

    // acmodel側のログ設定
    acmodel::setLogger([&logs](const std::string& message) {
        // acmodel::log 側で [acmodel] プレフィックスを付けるため、ここでは付けない
        writeDomainLog(logs, "空調", message);
    });
    acmodel::setLogVerbosity(logVerbosity_);

    const auto& graph = thermalNetwork.getGraph();
    auto vertices = boost::vertices(graph);
    int initialized = 0;

    for (auto vertex : boost::make_iterator_range(vertices)) {
        const auto& node = graph[vertex];
        if (node.type != "aircon") {
            continue;
        }
        // 制御対象キーはモデル有無に関わらず全 aircon を登録する
        airconKeysOrdered_.push_back(node.key);

        if (node.model == "IDEAL") {
            writeDomainLog(logs, "空調", "IDEAL モード（モデルなし）: " + node.key);
            continue;
        }
        if (node.ac_spec.empty()) {
            throw std::runtime_error(
                "aircon model init failed: ac_spec missing for key=" + node.key +
                " (use model=IDEAL for intentional model-less aircon)");
        }
        try {
            auto model = acmodel::AirconModelFactory::createModel(node.model, node.ac_spec);
            airconModels[node.key] = std::move(model);
            ++initialized;
            writeDomainLog(logs, "空調",
                     "モデル初期化完了: " + node.key +
                         " (タイプ: " + node.model + ")");
            // verbosity=1 でも、初期化の「最終サマリ（係数等）」だけは出す
            // （詳細ログは acmodel::setLogVerbosity により verbosity>=2 のときのみ）
            if (auto* m = getModel(node.key)) {
                const std::string s = m->getInitializationSummary();
                if (!s.empty()) {
                    // 初期化サマリは acmodel 側のログではないため、プレフィックスをここで付けて統一する
                    writeDomainLog(logs, "空調", std::string("　　[acmodel] ") + s);
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "aircon model init failed: key=" + node.key + " - " + e.what());
        }
    }
    std::sort(airconKeysOrdered_.begin(), airconKeysOrdered_.end());
    airconKeysCacheInitialized_ = true;

    // 同一 set_node を複数空調が制御すると fixed-row が頂点順依存になるため禁止する
    {
        std::unordered_map<std::string, std::string> setNodeOwner;
        for (auto vertex : boost::make_iterator_range(vertices)) {
            const auto& node = graph[vertex];
            if (node.type != "aircon" || node.set_node.empty()) {
                continue;
            }
            auto it = setNodeOwner.find(node.set_node);
            if (it != setNodeOwner.end()) {
                throw std::runtime_error(
                    "multiple aircons control the same set_node '" + node.set_node +
                    "': '" + it->second + "' and '" + node.key + "'");
            }
            setNodeOwner.emplace(node.set_node, node.key);
        }
    }

    writeDomainLog(logs, "空調", "モデル初期化総数: " + std::to_string(initialized) + "台");
}

void AirconController::registerModelForTesting(const std::string& airconKey,
                                               std::unique_ptr<acmodel::AirconSpec> model) {
    airconModels[airconKey] = std::move(model);
    if (std::find(airconKeysOrdered_.begin(), airconKeysOrdered_.end(), airconKey) ==
        airconKeysOrdered_.end()) {
        airconKeysOrdered_.push_back(airconKey);
        std::sort(airconKeysOrdered_.begin(), airconKeysOrdered_.end());
    }
    airconKeysCacheInitialized_ = true;
}

void AirconController::clearModelsForTesting() {
    airconModels.clear();
    airconKeysCacheInitialized_ = true;
    airconKeysOrdered_.clear();
    lastAppliedMode_.clear();
    clearCouplingWarmStart();
    clearCapacityLimitBracket();
}

acmodel::AirconSpec* AirconController::getModel(const std::string& airconKey) const {
    auto it = airconModels.find(airconKey);
    if (it == airconModels.end()) {
        return nullptr;
    }
    return it->second.get();
}

double AirconController::calculateHeatCapacity(ThermalNetwork& thermalNetwork,
                                               const std::string& mode,
                                               const std::string& inNode,
                                               const std::string& airconNode,
                                               const FlowRateMap& flowRates) const {
    if (inNode.empty()) {
        return 0.0;
    }

    double inletTemp = 0.0;
    double outletTemp = 0.0;
    if (!aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, inNode, inletTemp) ||
        !aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, airconNode, outletTemp)) {
        return 0.0;
    }

    double flowRate = aircon::network_utils::getAirconProcessFlowRate(flowRates, inNode, airconNode);
    if (std::abs(flowRate) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    // 処理熱量は「暖房/冷房どちらでも +W（大きさ）」として扱う。
    // - heating: 出口(吹出) > 入口 のときのみ加熱。出口 <= 入口なら 0（加熱していない）。
    // - cooling: 入口 > 出口 のときのみ除熱。入口 <= 出口なら 0。
    // モードと逆の向きの温度差は 0 とする。abs のみで正にするのは、熱ソルバが極端な設定で
    // 出口温度が暴れたときに巨大な処理熱量にならないようにするため。
    double deltaT = 0.0;
    if (isHeating(parseOperationModeOrDefaultCooling(mode))) {
        deltaT = outletTemp - inletTemp;
        if (deltaT <= 0.0) return 0.0;
    } else {
        deltaT = inletTemp - outletTemp;
        if (deltaT <= 0.0) return 0.0;
    }
    double heatCapacity = kAirDensity * kAirSpecificHeat * std::abs(flowRate) * deltaT;
    return clampHeatCapacity(heatCapacity);
}

double AirconController::calculateSignedProcessedHeat(ThermalNetwork& thermalNetwork,
                                                      const std::string& inNode,
                                                      const std::string& airconNode,
                                                      const FlowRateMap& flowRates) const {
    if (inNode.empty() || airconNode.empty()) {
        return 0.0;
    }
    double inletTemp = 0.0;
    double outletTemp = 0.0;
    if (!aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, inNode, inletTemp) ||
        !aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, airconNode, outletTemp)) {
        return 0.0;
    }
    const double flowRate =
        aircon::network_utils::getAirconProcessFlowRate(flowRates, inNode, airconNode);
    const double xIn = aircon::network_utils::getAbsoluteHumidityFromNode(thermalNetwork, inNode);
    const double xOut = aircon::network_utils::getAbsoluteHumidityFromNode(thermalNetwork, airconNode);
    return thermal_moist_air::signedProcessedHeatW(
        inletTemp, xIn, outletTemp, xOut, flowRate, moistEnthalpyEnabled_);
}

AirconValidationData AirconController::validateAirconData(const std::string& airconKey,
                                                          ThermalNetwork& thermalNetwork,
                                                          const VertexProperties& nodeProps) const {
    AirconValidationData data{};
    auto getTemp = [&](const std::string& nodeName, const char* label) -> double {
        if (nodeName.empty()) {
            throw std::runtime_error(std::string(label) + " が設定されていません (" + airconKey + ")");
        }
        double t = 0.0;
        if (!aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, nodeName, t)) {
            throw std::runtime_error(std::string(label) + " '" + nodeName + "' の温度が見つかりません");
        }
        return t;
    };

    data.outdoorTemp = getTemp(nodeProps.outside_node, "outside_node");
    data.indoorTemp = getTemp(nodeProps.in_node, "in_node");
    data.airconTemp = getTemp(nodeProps.key, "aircon_node");
    // 設定温度は set_node の現在温度ではなく、実効設定温度 current_pre_temp を使う。
    // （要求設定は current_requested_pre_temp。能力制限時は両者が異なる）
    if (std::isfinite(nodeProps.current_pre_temp)) {
        data.setTemp = nodeProps.current_pre_temp;
    } else if (!nodeProps.set_node.empty()) {
        data.setTemp = getTemp(nodeProps.set_node, "set_node");
    } else {
        data.setTemp = data.indoorTemp;
    }

    // 湿度（絶対湿度）: thermalNetwork の current_x を参照する
    data.outdoorX = aircon::network_utils::getAbsoluteHumidityFromNode(thermalNetwork, nodeProps.outside_node);
    data.indoorX = aircon::network_utils::getAbsoluteHumidityFromNode(thermalNetwork, nodeProps.in_node);
    return data;
}

AirconController::RuntimeContext AirconController::prepareRuntimeContext(
    const std::string& airconKey,
    ThermalNetwork& thermalNetwork,
    const VertexProperties& nodeProps,
    const FlowRateMap& flowRates) const {
    RuntimeContext context{};
    context.validData = validateAirconData(airconKey, thermalNetwork, nodeProps);
    context.airFlowRate = aircon::network_utils::getAirconProcessFlowRate(
        flowRates, nodeProps.in_node, nodeProps.key);

    context.operationMode = resolveOperationModeForRuntime(
        nodeProps.current_mode, context.validData.indoorTemp, context.validData.airconTemp);
    context.heatCapacity = calculateHeatCapacity(thermalNetwork, modeKey(context.operationMode),
                                                 nodeProps.in_node, nodeProps.key, flowRates);
    return context;
}

void AirconController::syncHumidityBoundariesBeforeSolve(ThermalNetwork& thermalNetwork) const {
    // 戻り値（再計算要求）は捨てる。ここは湿度ソルバ用の境界整備のみ。
    constexpr double kNoRecomputeTol = 1e30;
    // 未初期化判定床: 極小の正値（過乾燥の残留）を supplyX として残さない
    constexpr double kUninitializedHumidityFloor = 1e-4; // kg/kg(DA)
    for (const auto& airconKey : getAirconKeys()) {
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        const bool scheduleOff = (nodeProps.current_mode == "OFF");
        // 冷房以外は除湿しない。吸込へパススルーする。
        const bool nonCooling = (nodeProps.current_mode != "COOLING");
        const bool uninitialized =
            !std::isfinite(nodeProps.current_x) ||
            !(nodeProps.current_x > kUninitializedHumidityFloor);
        if (!nodeProps.on || scheduleOff || nonCooling || uninitialized) {
            (void)aircon::latent::applyPassthroughHumidityToAirconNode(
                thermalNetwork, airconKey, kNoRecomputeTol);
        }
    }
}

bool AirconController::controlAllAircons(ThermalNetwork& thermalNetwork,
                                         double tolerance,
                                         std::ostream& logFile,
                                         bool* supplyHumidityChanged,
                                         double humidityAbsTol,
                                         std::vector<AirconStateProposal>* outProposals,
                                         const FlowRateMap* flowRates) const {
    bool allControlled = true;
    constexpr double kMinMeaningfulProcessFlow = 1e-4; // [m3/s]

    // 順序を決定的にしてログ/挙動の再現性を上げる
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        double currentTemp = 0.0;
        if (!nodeProps.set_node.empty()) {
            if (!aircon::network_utils::tryGetTempFromThermalNetwork(
                    thermalNetwork, nodeProps.set_node, currentTemp)) {
                throw std::runtime_error(
                    "aircon set_node temperature lookup failed: aircon=" + airconKey +
                    ", set_node=" + nodeProps.set_node);
            }
        } else {
            if (!aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.key, currentTemp)) {
                currentTemp = nodeProps.current_t;
            }
        }
        double targetTemp = nodeProps.current_requested_pre_temp;

        // Qreq は set_node が実効設定近傍にあるときだけ信頼する。
        // fixed-row が効いていない解（室温が大きく外れている）では、
        // 容量ノードからの見かけの加熱などで符号が反転し ON/OFF が振動しうる。
        //
        // 能力制限中（実効設定を要求からずらしている）も Qreq で OFF しない。
        // 暖房で実効設定を下げた直後は、自然室温より低い温度を拘束するため
        // Qreq が負になり、OFF→要求温度で再ONの振動になる。
        //
        // 処理風量がほぼ 0 のときも Qreq（コイル熱）は信頼しない。
        // 固定温度行で室温だけ目標付近に見えて Qreq≈0→OFF→再ON の振動になる。
        const double setpointBandK = std::max(tolerance, 1.0);
        const bool nearSetpoint =
            std::isfinite(nodeProps.current_pre_temp) &&
            std::abs(currentTemp - nodeProps.current_pre_temp) <= setpointBandK;
        const bool capacityLimited =
            nodeProps.aircon_control_state == AirconControlState::CapacityLimited;
        const bool setpointDetuned =
            std::isfinite(nodeProps.current_pre_temp) &&
            std::isfinite(nodeProps.current_requested_pre_temp) &&
            std::abs(nodeProps.current_pre_temp - nodeProps.current_requested_pre_temp) > setpointBandK;
        bool processFlowTooLow = false;
        if (flowRates && !nodeProps.in_node.empty()) {
            const double processFlow = std::abs(aircon::network_utils::getAirconProcessFlowRate(
                *flowRates, nodeProps.in_node, nodeProps.key));
            processFlowTooLow = !(processFlow > kMinMeaningfulProcessFlow);
        }
        const bool useRequiredHeat =
            nodeProps.on && std::isfinite(nodeProps.required_heat_w) && nearSetpoint &&
            !capacityLimited && !setpointDetuned && !processFlowTooLow;
        auto result = controlAircon(nodeProps, currentTemp, targetTemp, tolerance, logFile,
                                    useRequiredHeat, nodeProps.required_heat_w,
                                    /*loadDeadbandW=*/1.0);
        writeDomainLog(logFile, "空調", result.logMessage);
        AirconRecomputeReason unitReasons = AirconRecomputeReason::None;
        if (result.stateChanged) {
            allControlled = false;
            nodeProps.on = result.on;
            unitReasons |= AirconRecomputeReason::OnOffChanged;
            if (!nodeProps.on) {
                nodeProps.required_heat_w = std::numeric_limits<double>::quiet_NaN();
            }
            // NOTE:
            // set_node の calc_t を ON/OFF で切り替えると、
            // 熱ソルバ側の「固定温度行（fixed row）」の適用条件（= set_node が未知数）を満たさず、
            // エアコンON直後に未収束/バランス超過になりやすい。
            //
            // setpoint 制御は thermal_solver_linear_direct.cpp の fixed row ロジックで行うため、
            // ここでは set_node.calc_t を変更しない。
        }

        if (!nodeProps.on) {
            nodeProps.aircon_control_state = AirconControlState::Off;
            // OFF 中は実効設定を要求値へ戻す（次に ON したときの起点を明確化）
            nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
            // OFF 中（遷移直後含む）は吹出湿度を入口へ追従。送風継続時の乾燥空気残留を防ぐ。
            if (aircon::latent::applyPassthroughHumidityToAirconNode(
                    thermalNetwork, airconKey, humidityAbsTol)) {
                if (supplyHumidityChanged) *supplyHumidityChanged = true;
                unitReasons |= AirconRecomputeReason::SupplyHumidityChanged;
            }
        } else if (nodeProps.aircon_control_state != AirconControlState::CapacityLimited) {
            nodeProps.aircon_control_state = AirconControlState::SetpointControlled;
            // 能力制限中でなければ実効設定=要求設定
            nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
        }

        if (outProposals) {
            auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
            proposal.reasons = unitReasons;
            outProposals->push_back(std::move(proposal));
        }
    }

    return allControlled;
}

bool AirconController::checkAndAdjustCapacity(ThermalNetwork& thermalNetwork,
                                              VentilationNetwork& /*ventNetwork*/,
                                              const SimulationConstants& constants,
                                              const FlowRateMap& flowRates,
                                              std::ostream& logs,
                                              int& /*totalIterations*/,
                                              bool* supplyHumidityChanged,
                                              double humidityAbsTol,
                                              std::vector<AirconStateProposal>* outProposals) const {
    moistEnthalpyEnabled_ = constants.moistEnthalpyEnabled;
    const double xTol = (humidityAbsTol > 0.0) ? humidityAbsTol : 1e-9;
    // 分岐: (1) 超過 → 公式で補正可能なら limitedSetpoint 適用、でなければ bracket 二分探索
    //       (2) 不足かつ bracket あり → 二分探索継続（設定温度を上げて再計算）
    //       (3) それ以外 → OK
    bool adjustmentMade = false;
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) {
            continue;
        }
        {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            const auto loads = aircon::latent::estimateLatentProcess(
                context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
                nodeProps, moistEnthalpyEnabled_);
            AirconRecomputeReason unitReasons = AirconRecomputeReason::None;
            if (aircon::latent::applySupplyHumidityToAirconNode(
                    thermalNetwork, airconKey, loads, xTol)) {
                if (supplyHumidityChanged) *supplyHumidityChanged = true;
                unitReasons |= AirconRecomputeReason::SupplyHumidityChanged;
            }
            std::string sourceLabel = "unknown";
            auto maxHeatCapacity = aircon::capacity::resolveMaxHeatCapacity(
                nodeProps, context.operationMode, sourceLabel);
            const double sensibleQ = std::max(0.0, loads.sensibleHeatCapacity);
            const double latentQ = std::max(0.0, loads.latentHeatCapacity);
            double current = aircon::latent::totalHeatCapacity(loads);

            std::ostringstream oss;
            oss << "　" << airconKey << " 最大処理熱量=";
            if (maxHeatCapacity) {
                oss << std::fixed << std::setprecision(2) << *maxHeatCapacity << "W";
                oss << " (" << sourceLabel << " 基準)";
            } else {
                oss << "N/A";
            }
            oss << ", 現在処理熱量(全熱)=" << std::fixed << std::setprecision(2) << current
                << "W (顕熱=" << sensibleQ << "W, 潜熱=" << latentQ << "W)";

            bool unitAdjusted = false;
            if (maxHeatCapacity && current > *maxHeatCapacity) {
                aircon::capacity::applyExceededCapacityAdjustment(
                    airconKey,
                    nodeProps,
                    context.operationMode,
                    context.validData.indoorTemp,
                    context.airFlowRate,
                    *maxHeatCapacity,
                    current,
                    capacityLimitBracket_,
                    oss,
                    unitAdjusted);
            } else if (maxHeatCapacity && current < *maxHeatCapacity && capacityLimitBracket_.count(airconKey)) {
                aircon::capacity::applyUnderCapacityBracketAdjustment(
                    airconKey,
                    nodeProps,
                    context.operationMode,
                    context.validData.indoorTemp,
                    context.airFlowRate,
                    *maxHeatCapacity,
                    current,
                    capacityLimitBracket_,
                    oss,
                    unitAdjusted);
            } else {
                oss << " → OK";
                // CapacityLimited 中は実効設定を維持（要求値へ勝手に戻さない）
                if (nodeProps.aircon_control_state != AirconControlState::CapacityLimited &&
                    !capacityLimitBracket_.count(airconKey)) {
                    nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
                    nodeProps.aircon_control_state = AirconControlState::SetpointControlled;
                }
            }
            if (unitAdjusted) {
                adjustmentMade = true;
                unitReasons |= AirconRecomputeReason::CapacitySetpointChanged;
            }
            writeDomainLog(logs, "空調", oss.str());

            if (outProposals) {
                auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
                proposal.processedHeatW = current;
                if (maxHeatCapacity) {
                    proposal.maxCapacityW = *maxHeatCapacity;
                    proposal.hasMaxCapacity = true;
                }
                proposal.currentFlowRate = context.airFlowRate;
                proposal.proposedFlowRate = context.airFlowRate;
                proposal.reasons = unitReasons;
                outProposals->push_back(std::move(proposal));
            }
        }
    }
    return adjustmentMade;
}

bool AirconController::checkAndAdjustDuctCentralAirflow(ThermalNetwork& thermalNetwork,
                                                        VentilationNetwork& ventNetwork,
                                                        const FlowRateMap& flowRates,
                                                        std::ostream& logs,
                                                        bool* supplyHumidityChanged,
                                                        double humidityAbsTol,
                                                        std::vector<AirconStateProposal>* outProposals) const {
    bool adjustmentMade = false;
    constexpr double kMinFlowTol = 1e-6;        // [m3/s]
    constexpr double kRelativeFlowTol = 1e-3;   // [-]
    const double xTol = (humidityAbsTol > 0.0) ? humidityAbsTol : 1e-9;

    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on || !aircon::airflow::isDuctCentralModel(nodeProps)) {
            continue;
        }

        auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
        const auto loads = aircon::latent::estimateLatentProcess(
            context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
            nodeProps, moistEnthalpyEnabled_);
        AirconRecomputeReason unitReasons = AirconRecomputeReason::None;
        if (aircon::latent::applySupplyHumidityToAirconNode(
                thermalNetwork, airconKey, loads, xTol)) {
            if (supplyHumidityChanged) *supplyHumidityChanged = true;
            unitReasons |= AirconRecomputeReason::SupplyHumidityChanged;
        }
        const double measuredHeatW = aircon::latent::totalHeatCapacity(loads);

        // 風量比の基準熱量:
        // - 能力制限中・要求設定未達: Q_max（追いつくまで設計風量側）
        // - 設定を保持できている: |required_heat_w|（室負荷。コイル熱 Q∝V は使わない）
        //   室負荷が無いときだけ Q_max に戻し、0 縮小を防ぐ
        // - set_node なし等で室温が自由: 計測処理熱
        double heatForFlowW = measuredHeatW;
        const char* heatBasis = "計測処理熱";
        const double unmetBandK = 0.5;
        double controlledRoomTemp = context.validData.setTemp;
        if (!nodeProps.set_node.empty()) {
            (void)aircon::network_utils::tryGetTempFromThermalNetwork(
                thermalNetwork, nodeProps.set_node, controlledRoomTemp);
        }
        const bool heatingUnmet =
            isHeating(context.operationMode) &&
            std::isfinite(nodeProps.current_requested_pre_temp) &&
            (controlledRoomTemp < nodeProps.current_requested_pre_temp - unmetBandK);
        const bool coolingUnmet =
            !isHeating(context.operationMode) &&
            std::isfinite(nodeProps.current_requested_pre_temp) &&
            (controlledRoomTemp > nodeProps.current_requested_pre_temp + unmetBandK);
        const bool capacityLimited =
            nodeProps.aircon_control_state == AirconControlState::CapacityLimited;
        const bool setpointHeld = nodeProps.on && !nodeProps.set_node.empty() &&
                                   !capacityLimited && !heatingUnmet && !coolingUnmet;
        const bool useExogenousHeatForFlow = capacityLimited || heatingUnmet || coolingUnmet || setpointHeld;
        std::string maxSource;
        const auto qMax = aircon::capacity::resolveMaxHeatCapacity(
            nodeProps, context.operationMode, maxSource);
        if (capacityLimited || heatingUnmet || coolingUnmet) {
            if (qMax) {
                heatForFlowW = *qMax;
                heatBasis = capacityLimited ? "能力上限" : "未達→能力上限";
            }
        } else if (setpointHeld) {
            if (std::isfinite(nodeProps.required_heat_w)) {
                heatForFlowW = std::abs(nodeProps.required_heat_w);
                if (qMax && heatForFlowW > *qMax) heatForFlowW = *qMax;
                heatBasis = "設定維持負荷";
            } else if (qMax) {
                heatForFlowW = *qMax;
                heatBasis = "設定固定→能力上限";
            }
        }

        const auto targetFlowOpt = aircon::airflow::computeTargetFlowFromProcessedHeat(
            nodeProps, context.operationMode, heatForFlowW);
        if (!targetFlowOpt) {
            continue;
        }
        // 極小目標は 0 にスナップし、0.00→0.00 の再計算ループを防ぐ。
        // flowRates 側に数値残差が残っても、実質ゼロ同士なら更新しない。
        constexpr double kAbsFlowSnap = 1e-4;  // [m3/s]
        constexpr double kAbsFlowMatch = 1e-3; // [m3/s] これ未満の差は無視
        double targetFlow = *targetFlowOpt;
        if (targetFlow < kAbsFlowSnap) {
            targetFlow = 0.0;
        }
        const double currentFlow = std::isfinite(context.airFlowRate) ? std::abs(context.airFlowRate) : 0.0;
        const double flowTol = std::max({kMinFlowTol, kAbsFlowMatch * 0.1,
                                         std::max({targetFlow, currentFlow, kAbsFlowSnap}) * kRelativeFlowTol});

        // 吸込が既に目標でも吹出がずれていれば更新する。一致判定は枝側で行う。
        bool edgeUpdated = false;
        if (!nodeProps.in_node.empty()) {
            edgeUpdated = aircon::airflow::updateDuctCentralCircuitFixedFlows(
                ventNetwork, nodeProps.in_node, nodeProps.key, targetFlow, flowTol);
        }

        if (!edgeUpdated) {
            if (outProposals && unitReasons != AirconRecomputeReason::None) {
                auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
                proposal.processedHeatW = measuredHeatW;
                if (useExogenousHeatForFlow) {
                    proposal.maxCapacityW = heatForFlowW;
                    proposal.hasMaxCapacity = true;
                }
                proposal.currentFlowRate = context.airFlowRate;
                proposal.proposedFlowRate = targetFlow;
                proposal.reasons = unitReasons;
                outProposals->push_back(std::move(proposal));
            }
            continue;
        }

        adjustmentMade = true;
        unitReasons |= AirconRecomputeReason::AirflowChanged;
        std::ostringstream oss;
        oss << "　" << airconKey
            << " DUCT_CENTRAL風量補正: 基準熱量(" << heatBasis << ")="
            << std::fixed << std::setprecision(2) << heatForFlowW << "W"
            << " (計測=" << measuredHeatW << "W)"
            << ", 風量 " << context.airFlowRate << "→" << targetFlow << " m3/s, 再計算要求";
        writeDomainLog(logs, "空調", oss.str());

        if (outProposals) {
            auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
            proposal.processedHeatW = measuredHeatW;
            if (useExogenousHeatForFlow) {
                proposal.maxCapacityW = heatForFlowW;
                proposal.hasMaxCapacity = true;
            }
            proposal.currentFlowRate = context.airFlowRate;
            proposal.proposedFlowRate = targetFlow;
            proposal.reasons = unitReasons;
            outProposals->push_back(std::move(proposal));
        }
    }
    return adjustmentMade;
}

const std::vector<std::string>& AirconController::getAirconKeys() const {
    if (!airconKeysCacheInitialized_) {
        airconKeysOrdered_.clear();
        airconKeysOrdered_.reserve(airconModels.size());
        for (const auto& kv : airconModels) {
            airconKeysOrdered_.push_back(kv.first);
        }
        std::sort(airconKeysOrdered_.begin(), airconKeysOrdered_.end());
        airconKeysCacheInitialized_ = true;
    }
    return airconKeysOrdered_;
}

std::vector<double> AirconController::collectAirconDataValues(ThermalNetwork& thermalNetwork,
                                                              const FlowRateMap& flowRates,
                                                              const std::string& dataType) const {
    const auto& keys = getAirconKeys();
    std::vector<double> values(keys.size(), 0.0);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& airconKey = keys[i];
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        try {
            if (dataType == "airconTemp") {
                double t = 0.0;
                if (aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.key, t)) values[i] = t;
            } else if (dataType == "inTemp") {
                if (!nodeProps.in_node.empty()) {
                    double t = 0.0;
                    if (aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.in_node, t)) values[i] = t;
                }
            } else if (dataType == "flow") {
                values[i] = aircon::network_utils::getAirconProcessFlowRate(
                    flowRates, nodeProps.in_node, nodeProps.key);
            } else if (dataType == "sensibleHeatCapacity" || dataType == "latentHeatCapacity") {
                // 処理熱量は「実機出力」として扱うため、OFF時は 0 を返す。
                // 出力収集は副作用なし（supplyX のグラフ適用は外側ループ側の正本）。
                if (!nodeProps.on) {
                    values[i] = 0.0;
                    continue;
                }
                auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
                const auto loads = aircon::latent::estimateLatentProcess(
                    context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
                    nodeProps, moistEnthalpyEnabled_);
                values[i] = (dataType == "sensibleHeatCapacity")
                                ? loads.sensibleHeatCapacity
                                : loads.latentHeatCapacity;
            }
        } catch (...) {
            values[i] = 0.0;
        }
    }
    return values;
}

std::pair<double, double> AirconController::estimatePowerAndCOPForAircon(
    const std::string& airconKey,
    ThermalNetwork& thermalNetwork,
    const VertexProperties& nodeProps,
    const FlowRateMap& flowRates,
    std::ostream& logs,
    bool logDetail) const {
    auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
    auto* model = getModel(airconKey);
    if (!model) {
        throw std::runtime_error("初期化済みモデルがありません");
    }
    const auto loads = aircon::latent::estimateLatentProcess(
        context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
        nodeProps, moistEnthalpyEnabled_);
    // 出力・COP 計算はグラフ状態を変更しない（supplyX 適用は外側ループ側）
    acmodel::InputData input =
        aircon::latent::buildAcmodelInput(context.validData,
                                          loads.sensibleHeatCapacity, loads.latentHeatCapacity,
                                          context.airFlowRate);

    const bool heating = isHeating(context.operationMode);
    bool usedFallbackEx = false;
    bool usedFallbackIn = false;
    if (!(input.X_ex > 0.0)) {
        usedFallbackEx = true;
        input.X_ex = heating ? archenv::jis::X_H_EX : archenv::jis::X_C_EX;
    }
    if (!(input.X_in > 0.0)) {
        usedFallbackIn = true;
        input.X_in = heating ? archenv::jis::X_H_IN : archenv::jis::X_C_IN;
    }
    if (logDetail && logVerbosity_ >= 1 && (usedFallbackEx || usedFallbackIn)) {
        std::ostringstream warn;
        warn << "　　[WARN] エアコン湿度入力が不足のためJIS条件で補完: " << airconKey
             << " [" << modeKey(context.operationMode) << "]";
        if (usedFallbackIn) warn << " in_node=" << nodeProps.in_node << " X_in=JIS";
        if (usedFallbackEx) warn << " outside_node=" << nodeProps.outside_node << " X_ex=JIS";
        writeDomainLog(logs, "空調", warn.str());
    }
    if (logDetail && logVerbosity_ >= 1 && loads.usedRh95Fallback) {
        std::ostringstream warn;
        warn << "　　[WARN] bf法の吹出点相対湿度が100%を超えたためRH95法へフォールバック: " << airconKey
             << " RH(bf)=" << std::fixed << std::setprecision(2) << loads.bfRhPercentBeforeFallback
             << "% -> RH(out)=" << std::fixed << std::setprecision(2) << loads.supplyRhPercent
             << "% (Tout=" << context.validData.airconTemp << "°C, X_out=" << loads.supplyX
             << ", T_coil=" << loads.coilTemp << "°C, X_coil=" << loads.coilX << ")";
        writeDomainLog(logs, "空調", warn.str());
    }
    auto result = model->estimateCOP(modeKey(context.operationMode), input);
    if (logVerbosity_ >= 2) {
        for (const auto& msg : result.logMessages) {
            writeDomainLog(logs, "空調", msg);
        }
    }
    if (!result.valid) {
        throw std::runtime_error("COP推定に失敗しました");
    }
    const double powerW = result.power * 1000.0; // kW -> W
    if (logDetail && logVerbosity_ >= 1) {
        std::ostringstream detail;
        detail << "　　エアコン電力計算: " << airconKey
               << " [" << modeKey(context.operationMode) << "]"
               << " 顕熱=" << std::fixed << std::setprecision(2) << loads.sensibleHeatCapacity << "W"
               << " 潜熱=" << std::fixed << std::setprecision(2) << loads.latentHeatCapacity << "W"
               << " 合計=" << std::fixed << std::setprecision(2) << aircon::latent::totalHeatCapacity(loads) << "W"
               << " 風量=" << context.airFlowRate << "m³/s"
               << " 外気=" << context.validData.outdoorTemp << "°C"
               << " 室内=" << context.validData.indoorTemp << "°C"
               << " COP=" << result.COP
               << " 電力=" << powerW << "W";
        writeDomainLog(logs, "空調", detail.str());
    }
    return {powerW, result.COP};
}

std::vector<double> AirconController::calculatePowerOrCOPValues(ThermalNetwork& thermalNetwork,
                                                                const FlowRateMap& flowRates,
                                                                std::ostream& logs,
                                                                bool returnPower) const {
    const auto& keys = getAirconKeys();
    std::vector<double> values(keys.size(), 0.0);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& airconKey = keys[i];
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) {
            continue;
        }
        try {
            const auto pair =
                estimatePowerAndCOPForAircon(airconKey, thermalNetwork, nodeProps, flowRates, logs,
                                             returnPower /* power のみ詳細ログ */);
            values[i] = returnPower ? pair.first : pair.second;
        } catch (const std::exception& e) {
            writeDomainLog(logs, "空調", std::string("[ERROR] ") + airconKey + " - " + e.what());
        }
    }
    return values;
}

std::vector<double> AirconController::calculatePowerValues(ThermalNetwork& thermalNetwork,
                                                           const FlowRateMap& flowRates,
                                                           std::ostream& logs) const {
    return calculatePowerOrCOPValues(thermalNetwork, flowRates, logs, true);
}

std::vector<double> AirconController::calculateCOPValues(ThermalNetwork& thermalNetwork,
                                                         const FlowRateMap& flowRates,
                                                         std::ostream& logs) const {
    return calculatePowerOrCOPValues(thermalNetwork, flowRates, logs, false);
}

AirconController::LatentFeedbackStats
AirconController::applyLatentFeedbackToThermal(ThermalNetwork& thermalNetwork,
                                               const FlowRateMap& flowRates,
                                               double relaxation,
                                               std::ostream& logs) const {
    LatentFeedbackStats stats{};
    if (!(relaxation > 0.0)) return stats;
    // moist enthalpy 経路と二重になるため無効
    if (moistEnthalpyEnabled_) return stats;
    const double alpha = std::min(1.0, relaxation);

    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) continue;
        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            if (context.operationMode != OperationMode::Cooling) continue;
            const auto loads = aircon::latent::estimateLatentProcess(
                context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
                nodeProps, moistEnthalpyEnabled_);
            aircon::latent::applySupplyHumidityToAirconNode(thermalNetwork, airconKey, loads);
            const double latentQ = std::max(0.0, loads.latentHeatCapacity);
            if (!(latentQ > 0.0)) continue;
            if (nodeProps.in_node.empty()) continue;

            // set_node が固定温度行で拘束されるケースでは、
            // in_node への heat_source 注入が熱収支残差として残りやすい。
            // （制御ノードの熱負荷は別経路で評価されるため、ここでは注入を抑止）
            bool targetIsActiveSetpointNode = false;
            for (const auto& key : getAirconKeys()) {
                const auto& ac = thermalNetwork.getNode(key);
                if (!ac.on) continue;
                if (!ac.set_node.empty() && ac.set_node == nodeProps.in_node) {
                    targetIsActiveSetpointNode = true;
                    break;
                }
            }
            if (targetIsActiveSetpointNode) {
                if (logVerbosity_ >= 1) {
                    std::ostringstream oss;
                    oss << "　潜熱フィードバック: " << airconKey
                        << " は setpoint 固定ノード(" << nodeProps.in_node
                        << ")への注入をスキップ";
                    writeDomainLog(logs, "空調", oss.str());
                }
                continue;
            }

            auto& inNode = thermalNetwork.getNode(nodeProps.in_node);
            const double deltaQ = -alpha * latentQ; // 冷房除湿は室側の熱源としては負（除熱）
            inNode.heat_source += deltaQ;
            stats.maxAppliedHeatW = std::max(stats.maxAppliedHeatW, std::abs(deltaQ));

            if (logVerbosity_ >= 2) {
                std::ostringstream oss;
                oss << "　潜熱フィードバック: " << airconKey
                    << " in_node=" << nodeProps.in_node
                    << " latent=" << latentQ << "W"
                    << " alpha=" << alpha
                    << " applied=" << deltaQ << "W";
                writeDomainLog(logs, "空調", oss.str());
            }
        } catch (const std::exception& e) {
            writeDomainLog(logs, "空調", std::string("[ERROR] 潜熱フィードバック ") + airconKey + " - " + e.what());
        }
    }
    return stats;
}

void AirconController::clearCapacityLimitBracket() const {
    capacityLimitBracket_.clear();
}

void AirconController::clearCouplingWarmStart() {
    prevAirconStateSig_ = 0;
    havePrevAirconStateSig_ = false;
}

void AirconController::observeAirconStateSignature(std::uint64_t sig) {
    prevAirconStateSig_ = sig;
    havePrevAirconStateSig_ = true;
}

bool AirconController::shouldForceMinTwoCouplingIters(std::uint64_t currentSig) const {
    if (!havePrevAirconStateSig_) {
        return true;
    }
    return currentSig != prevAirconStateSig_;
}

void AirconController::applyPreset(ThermalNetwork& thermalNetwork,
                                   std::ostream& logs) const {
    auto& graph = thermalNetwork.getGraph();
    // 順序を決定的にしてログ/挙動の再現性を上げる
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        const std::string& mode = nodeProps.current_mode;
        if (mode == "OFF") {
            nodeProps.on = false;
        } else {
            const auto it = lastAppliedMode_.find(airconKey);
            const bool firstOrWasOff =
                (it == lastAppliedMode_.end() || it->second == "OFF");
            const bool modeSwitched =
                (it != lastAppliedMode_.end() && it->second != "OFF" && it->second != mode);
            // 初回・OFF→運転・暖房↔冷房切替は ON で開始。モード継続時は前ステップの ON/OFF を維持。
            if (firstOrWasOff || modeSwitched) {
                nodeProps.on = true;
            }
        }
        lastAppliedMode_[airconKey] = mode;
        std::string target = nodeProps.set_node.empty() ? nodeProps.key : nodeProps.set_node;
        writeDomainLog(logs, "空調",
                 std::string("　エアコン設定（初期化）: ") + target +
                     " ON/OFF=" + (nodeProps.on ? "ON" : "OFF"));
        if (!nodeProps.set_node.empty()) {
            const auto& mapping = thermalNetwork.getKeyToVertex();
            auto it = mapping.find(nodeProps.set_node);
            if (it != mapping.end()) {
                graph[it->second].calc_t = true;
            }
        }
    }
}
