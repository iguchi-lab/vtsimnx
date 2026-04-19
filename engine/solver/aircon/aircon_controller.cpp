#include "aircon/aircon_controller.h"
#include "aircon/aircon_capacity.h"
#include "aircon/aircon_airflow.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_network_utils.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "archenv/include/archenv.h"
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

    // acmodel側のログ設定
    acmodel::setLogger([&logs](const std::string& message) {
        // acmodel::log 側で [acmodel] プレフィックスを付けるため、ここでは付けない
        writeLog(logs, message);
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
        if (node.ac_spec.empty()) {
            writeLog(logs, "　エアコンモデル初期化スキップ: " + node.key + " (ac_spec 未設定)");
            continue;
        }
        try {
            auto model = acmodel::AirconModelFactory::createModel(node.model, node.ac_spec);
            airconModels[node.key] = std::move(model);
            ++initialized;
            writeLog(logs,
                     "　エアコンモデル初期化完了: " + node.key +
                         " (タイプ: " + node.model + ")");
            // verbosity=1 でも、初期化の「最終サマリ（係数等）」だけは出す
            // （詳細ログは acmodel::setLogVerbosity により verbosity>=2 のときのみ）
            if (auto* m = getModel(node.key)) {
                const std::string s = m->getInitializationSummary();
                if (!s.empty()) {
                    // 初期化サマリは acmodel 側のログではないため、プレフィックスをここで付けて統一する
                    writeLog(logs, std::string("　　[acmodel] ") + s);
                }
            }
        } catch (const std::exception& e) {
            writeLog(logs,
                     "　エラー: エアコンモデル初期化に失敗" + node.key + " - " + e.what());
        }
    }
    writeLog(logs, "  エアコンモデル初期化総数: " + std::to_string(initialized) + "台");
}

void AirconController::registerModelForTesting(const std::string& airconKey,
                                               std::unique_ptr<acmodel::AirconSpec> model) {
    airconModels[airconKey] = std::move(model);
    airconKeysCacheInitialized_ = false;
    airconKeysOrdered_.clear();
}

void AirconController::clearModelsForTesting() {
    airconModels.clear();
    airconKeysCacheInitialized_ = false;
    airconKeysOrdered_.clear();
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

    double flowRate = aircon::network_utils::getFlowRate(flowRates, inNode, airconNode);
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
    // 設定温度は set_node の現在温度ではなく、airconノードの current_pre_temp を使う。
    // （set_node は制御対象室であり、設定値そのものではない）
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
    context.airFlowRate = std::abs(aircon::network_utils::getFlowRate(flowRates, nodeProps.in_node, nodeProps.key));

    context.operationMode = resolveOperationModeForRuntime(
        nodeProps.current_mode, context.validData.indoorTemp, context.validData.airconTemp);
    context.heatCapacity = calculateHeatCapacity(thermalNetwork, modeKey(context.operationMode),
                                                 nodeProps.in_node, nodeProps.key, flowRates);
    return context;
}

bool AirconController::controlAllAircons(ThermalNetwork& thermalNetwork,
                                         double tolerance,
                                         std::ostream& logFile) const {
    bool allControlled = true;

    // 順序を決定的にしてログ/挙動の再現性を上げる
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        double currentTemp = 0.0;
        if (!nodeProps.set_node.empty()) {
            (void)aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.set_node, currentTemp);
        } else {
            if (!aircon::network_utils::tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.key, currentTemp)) {
                currentTemp = nodeProps.current_t;
            }
        }
        double targetTemp = nodeProps.current_pre_temp;

        auto result = controlAircon(nodeProps, currentTemp, targetTemp, tolerance, logFile);
        writeLog(logFile, result.logMessage);
        if (result.stateChanged) {
            allControlled = false;
            nodeProps.on = result.on;
            // NOTE:
            // set_node の calc_t を ON/OFF で切り替えると、
            // 熱ソルバ側の「固定温度行（fixed row）」の適用条件（= set_node が未知数）を満たさず、
            // エアコンON直後に未収束/バランス超過になりやすい。
            //
            // setpoint 制御は thermal_solver_linear_direct.cpp の fixed row ロジックで行うため、
            // ここでは set_node.calc_t を変更しない。
            //
            // NOTE(将来拡張の忘備録):
            // 現状は「エアコンOFFでも送風（=in->aircon / aircon->out の換気流量）は入力の ventilation_branches に従う」
            // という前提で、aircon の ON/OFF と換気枝の enable/vol を自動連動していない。
            // もし「OFFなら送風も停止（流量=0）」を実現したくなった場合は、
            // - 入力JSON側で ventilation_branches.enable を時系列で制御する
            //   もしくは
            // - solver側で nodeProps.on と換気枝 current_enabled/flow_rate を連動させる
            // といった対応が必要になる。
        }
    }

    return allControlled;
}

bool AirconController::checkAndAdjustCapacity(ThermalNetwork& thermalNetwork,
                                              VentilationNetwork& /*ventNetwork*/,
                                              const SimulationConstants& /*constants*/,
                                              const FlowRateMap& flowRates,
                                              std::ostream& logs,
                                              int& /*totalIterations*/) const {
    // 分岐: (1) 超過 → 公式で補正可能なら limitedSetpoint 適用、でなければ bracket 二分探索
    //       (2) 不足かつ bracket あり → 二分探索継続（設定温度を上げて再計算）
    //       (3) それ以外 → OK
    bool adjustmentMade = false;
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) {
            continue;
        }
        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            const auto loads = aircon::latent::estimateLatentProcess(
                context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
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
                    adjustmentMade);
            } else if (maxHeatCapacity && current < *maxHeatCapacity && capacityLimitBracket_.count(airconKey)) {
                aircon::capacity::applyUnderCapacityBracketAdjustment(
                    airconKey,
                    nodeProps,
                    context.operationMode,
                    *maxHeatCapacity,
                    current,
                    capacityLimitBracket_,
                    oss,
                    adjustmentMade);
            } else {
                oss << " → OK";
            }
            writeLog(logs, oss.str());
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: エアコン ") + airconKey + " - " + e.what());
        }
    }
    return adjustmentMade;
}

bool AirconController::checkAndAdjustDuctCentralAirflow(ThermalNetwork& thermalNetwork,
                                                        VentilationNetwork& ventNetwork,
                                                        const FlowRateMap& flowRates,
                                                        std::ostream& logs) const {
    bool adjustmentMade = false;
    constexpr double kMinFlowTol = 1e-6;        // [m3/s]
    constexpr double kRelativeFlowTol = 1e-3;   // [-]

    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on || !aircon::airflow::isDuctCentralModel(nodeProps)) {
            continue;
        }

        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            const auto loads = aircon::latent::estimateLatentProcess(
                context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
            const double processedHeatW = aircon::latent::totalHeatCapacity(loads);

            const auto targetFlowOpt = aircon::airflow::computeTargetFlowFromProcessedHeat(
                nodeProps, context.operationMode, processedHeatW);
            if (!targetFlowOpt) {
                continue;
            }
            const double targetFlow = *targetFlowOpt;
            const double flowTol = std::max(kMinFlowTol,
                                            std::max(targetFlow, context.airFlowRate) * kRelativeFlowTol);

            if (!std::isfinite(context.airFlowRate) || std::abs(context.airFlowRate - targetFlow) <= flowTol) {
                continue;
            }

            bool edgeUpdated = false;
            if (!nodeProps.in_node.empty()) {
                edgeUpdated = aircon::airflow::updateFixedFlowEdgeByNodePair(
                    ventNetwork, nodeProps.in_node, nodeProps.key, targetFlow, flowTol);
            }

            if (!edgeUpdated) {
                continue;
            }

            adjustmentMade = true;
            std::ostringstream oss;
            oss << "　" << airconKey
                << " DUCT_CENTRAL風量補正: 処理熱量=" << std::fixed << std::setprecision(2)
                << processedHeatW << "W"
                << ", 風量 " << context.airFlowRate << "→" << targetFlow << " m3/s, 再計算要求";
            writeLog(logs, oss.str());
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: DUCT_CENTRAL風量補正に失敗 ")
                               + airconKey + " - " + e.what());
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
                values[i] = std::abs(aircon::network_utils::getFlowRate(flowRates, nodeProps.in_node, nodeProps.key));
            } else if (dataType == "sensibleHeatCapacity" || dataType == "latentHeatCapacity") {
                // 処理熱量は「実機出力」として扱うため、OFF時は 0 を返す。
                if (!nodeProps.on) {
                    values[i] = 0.0;
                    continue;
                }
                auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
                const auto loads = aircon::latent::estimateLatentProcess(
                    context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
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
        context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
    // 吹出絶対湿度をエアコンノードへ反映する（humidity_x 出力および次ステップ初期値に利用）。
    // ここでは冷房時のみ有効値が入り、暖房/無効時は入力湿度（実質据え置き）となる。
    thermalNetwork.getNode(airconKey).current_x = loads.supplyX;
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
        writeLog(logs, warn.str());
    }
    if (logDetail && logVerbosity_ >= 1 && loads.usedRh95Fallback) {
        std::ostringstream warn;
        warn << "　　[WARN] bf法の吹出点相対湿度が100%を超えたためRH95法へフォールバック: " << airconKey
             << " RH(bf)=" << std::fixed << std::setprecision(2) << loads.bfRhPercentBeforeFallback
             << "% -> RH(out)=" << std::fixed << std::setprecision(2) << loads.supplyRhPercent
             << "% (Tout=" << context.validData.airconTemp << "°C, X_out=" << loads.supplyX
             << ", T_coil=" << loads.coilTemp << "°C, X_coil=" << loads.coilX << ")";
        writeLog(logs, warn.str());
    }
    auto result = model->estimateCOP(modeKey(context.operationMode), input);
    if (logVerbosity_ >= 2) {
        for (const auto& msg : result.logMessages) {
            writeLog(logs, msg);
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
        writeLog(logs, detail.str());
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
            writeLog(logs, std::string("　　エラー: エアコン ") + airconKey + " - " + e.what());
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
    const double alpha = std::min(1.0, relaxation);

    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) continue;
        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            if (context.operationMode != OperationMode::Cooling) continue;
            const auto loads = aircon::latent::estimateLatentProcess(
                context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
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
                    writeLog(logs, oss.str());
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
                writeLog(logs, oss.str());
            }
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: 潜熱フィードバック ") + airconKey + " - " + e.what());
        }
    }
    return stats;
}

void AirconController::clearCapacityLimitBracket() const {
    capacityLimitBracket_.clear();
}

void AirconController::applyPreset(ThermalNetwork& thermalNetwork,
                                   std::ostream& logs) const {
    auto& graph = thermalNetwork.getGraph();
    // 順序を決定的にしてログ/挙動の再現性を上げる
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        // 既定は「モードOFF以外なら初期ON」。
        // 制御ループで不要運転はOFFへ落とす。
        // （毎ステップOFF開始より、再計算回数を抑えられるケースが多い）
        nodeProps.on = (nodeProps.current_mode != "OFF");
        std::string target = nodeProps.set_node.empty() ? nodeProps.key : nodeProps.set_node;
        writeLog(logs,
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
