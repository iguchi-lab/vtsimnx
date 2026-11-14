#include "simulation_error.h"
#include "network/humidity_network.h"
#include "aircon/aircon_controller.h"
#include "aircon/aircon_onoff.h"
#include "aircon/aircon_capacity.h"
#include "aircon/aircon_airflow.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_humidity_control.h"
#include "aircon/aircon_network_utils.h"
#include "aircon/aircon_saturation.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "archenv/include/archenv.h"
#include "core/thermal/thermal_moist_air.h"
#include "utils/utils.h"
#include "acmodel/acmodel.h"
#include "acmodel/duct_central_model.h"

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

void logLatentBoundary(std::ostream& logs, const char* stage, const std::string& key,
                       const AirconValidationData& data, const aircon::latent::LatentProcessResult& loads,
                       const VertexProperties& node, double flow, double oldX, bool changed) {
    std::ostringstream out;
    out << std::setprecision(10) << "latent_boundary stage=" << stage << " key=" << key
        << " mode=" << node.current_mode << " regime=" << loads.regime
        << " T_in=" << data.indoorTemp << " T_supply=" << data.airconTemp
        << " x_in=" << data.indoorX << " x_old=" << oldX << " x_new=" << loads.supplyX
        << " pre_rh=";
    // NaN is the documented sentinel for an omitted RH setpoint, not a failed solve.
    if (std::isnan(node.current_pre_rh)) out << "unset";
    else out << node.current_pre_rh;
    out << " flow=" << flow
        << " sensible_W=" << loads.sensibleHeatCapacity << " latent_W=" << loads.latentHeatCapacity
        << " saturation_limited=" << loads.saturationLimited
        << " RH_supply=" << loads.supplyRhPercent
        << " removal_kg_s=" << loads.condensationRateKgPerS
        << " recompute=" << changed;
    writeDomainLog(logs, "空調診断", out.str());
}

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

        if (node.control_mode != "temperature_priority") {
            if (node.set_node.empty()) throw std::runtime_error("Humidistat requires set_node: " + node.key);
            const auto& room = thermalNetwork.getNode(node.set_node);
            if (!room.calc_x || !room.calc_t)
                throw std::runtime_error("Humidistat set room requires calc_x and calc_t: " + node.key);
        }

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
    if (context.operationMode == OperationMode::Cooling) {
        for (const auto& outlet : aircon::network_utils::passiveSupplyOutletConstraints(thermalNetwork, nodeProps))
            context.validData.additionalSupplyXUpperBound = std::min(
                context.validData.additionalSupplyXUpperBound, outlet.supplyXUpperBound);
    }
    context.heatCapacity = calculateHeatCapacity(thermalNetwork, modeKey(context.operationMode),
                                                 nodeProps.in_node, nodeProps.key, flowRates);
    return context;
}

void AirconController::syncHumidityBoundariesBeforeSolve(ThermalNetwork& thermalNetwork) const {
    // Pass-through humidity is solved with the room, never copied from its
    // previous iterate. Only clear the obsolete condensation diagnostic here.
    for (const auto& airconKey : getAirconKeys()) {
        auto& node = thermalNetwork.getNode(airconKey);
        if (airconHumidityPassthrough(node)) {
            node.aircon_moisture_removal_kg_s = 0.0;
            node.aircon_humidity_transition = false;
        }
    }
}

void AirconController::prepareAirflowForSolve(ThermalNetwork& thermalNetwork,
                                              VentilationNetwork& ventNetwork) const {
    const auto flows = ventNetwork.collectFlowRateMap();
    for (const auto& key : getAirconKeys()) {
        const auto& node = thermalNetwork.getNode(key);
        if (!aircon::airflow::isDuctCentralModel(node)) continue;
        const double current = aircon::network_utils::getAirconProcessFlowRate(flows, node.in_node, key);
        if (node.on && std::abs(current) > 1e-4) continue;
        const auto mode = aircon::network_utils::fanMode(node);
        double target = node.vol_zero;
        if (node.on) {
            std::string source;
            const auto qmax = aircon::capacity::resolveMaxHeatCapacity(node, mode, source);
            if (!qmax) continue;
            const auto flow = aircon::airflow::computeTargetFlowFromProcessedHeat(node, mode, *qmax, target, 0.0);
            if (!flow) continue;
            target = *flow;
        }
        bool fan = false;
        aircon::airflow::updateDuctCentralFanAffinity(ventNetwork, node, mode, key, target,
                                                     1e-3, &fan, false, false);
        if (!fan) aircon::airflow::updateDuctCentralCircuitFixedFlows(
            ventNetwork, node.in_node, key, target, 1e-4);
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
    const double setpointBandK = aircon::onoff::restartBandK(tolerance);

    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        const double currentTemp =
            aircon::network_utils::getControlledRoomTemperature(thermalNetwork, nodeProps);
        const double targetTemp = nodeProps.current_requested_pre_temp;

        std::optional<double> processFlow;
        if (flowRates && !nodeProps.in_node.empty()) {
            processFlow = aircon::network_utils::getAirconProcessFlowRate(
                *flowRates, nodeProps.in_node, nodeProps.key);
        }
        const auto trust = aircon::onoff::buildRequiredHeatTrust(
            currentTemp,
            nodeProps.current_pre_temp,
            nodeProps.current_requested_pre_temp,
            setpointBandK,
            nodeProps.aircon_control_state == AirconControlState::CapacityLimited,
            processFlow);
        const bool useRequiredHeat =
            aircon::onoff::shouldUseRequiredHeat(nodeProps.on, nodeProps.required_heat_w, trust);
        if (!useRequiredHeat && std::isfinite(currentTemp)) {
            lastFreeSetTempC_[airconKey] = currentTemp;
        }

        const char* minMode = aircon::onoff::minCapacityModeKey(
            nodeProps.current_mode, nodeProps.required_heat_w);
        double minProcessHeatW = 0.0;
        if (useRequiredHeat) {
            if (const auto* spec = nodeProps.getAirconSpec()) {
                minProcessHeatW =
                    aircon::onoff::resolveMinProcessHeatW(spec->getMinProcess(minMode));
            }
        }
        const bool holdAtMinimumCapacity = useRequiredHeat &&
            aircon::onoff::inMinProcessHoldBand(minMode, nodeProps.required_heat_w,
                                                minProcessHeatW, aircon::onoff::kLoadDeadbandW,
                                                nodeProps.aircon_min_process_hold);

        auto result = controlAircon(nodeProps, currentTemp, targetTemp, tolerance, logFile,
                                    useRequiredHeat, nodeProps.required_heat_w,
                                    aircon::onoff::kLoadDeadbandW, minProcessHeatW,
                                    holdAtMinimumCapacity, nodeProps.aircon_min_process_hold);
        const bool reheatDemand = aircon::humidity_control::simultaneous(nodeProps) &&
            (currentTemp < targetTemp - setpointBandK ||
             (nodeProps.on && std::isfinite(nodeProps.required_heat_w) && nodeProps.required_heat_w > 1.0));
        // A humidity-triggered fan can itself satisfy RH by mixing dry return
        // air. Keep that trial active while solving its cooling/fan-only state;
        // turning it off merely because its own airflow removed the demand
        // would alternate between two different moisture networks forever.
        const bool humidityTrial = nodeProps.on && nodeProps.humidity_overcool_hold;
        if (aircon::humidity_control::enabled(nodeProps) &&
            (nodeProps.humidity_demand || reheatDemand || humidityTrial)) {
            const bool reheat = aircon::humidity_control::simultaneous(nodeProps);
            const double floor = targetTemp - nodeProps.max_overcooling;
            // A remote return room can be colder than the controlled room.
            // Use the coil's actual sensible sign, not just the set-room load:
            // maintaining the floor must never introduce unreported heating.
            const double heatingNeededW = flowRates
                ? calculateHeatCapacity(thermalNetwork, "HEATING", nodeProps.in_node, airconKey, *flowRates)
                : std::max(0.0, nodeProps.required_heat_w);
            if (!reheat && nodeProps.on && heatingNeededW > 1.0 &&
                nodeProps.current_pre_temp <= floor + 1e-4) {
                humidityFloorBlocked_.insert(airconKey);
            }
            // Do not heat an already cold room in humidity-priority mode.
            const bool mayCool = currentTemp >= floor - 1e-4 &&
                !humidityFloorBlocked_.count(airconKey);
            if (reheat || mayCool) {
                result.on = true;
                result.stateChanged = !nodeProps.on;
                result.logMessage = nodeProps.key + " humidistat: ON";
                nodeProps.humidity_overcool_hold = !reheat;
            }
        }
        if (aircon::humidity_control::enabled(nodeProps) &&
            humidityFloorBlocked_.count(airconKey)) {
            // Stopping the fan changes the free-room state. Do not retry the
            // same infeasible full-flow cooling state until the next timestep.
            result.on = false;
            result.stateChanged = nodeProps.on;
            result.logMessage = nodeProps.key + " humidistat: OFF (overcooling floor requires heating)";
            nodeProps.humidity_overcool_hold = false;
        }
        if (result.on && useRequiredHeat) {
            nodeProps.aircon_min_process_hold = holdAtMinimumCapacity;
        } else if (!result.on) {
            nodeProps.aircon_min_process_hold = false;
        }
        // DUCTの風量は生の設定維持負荷を参照する。最低能力は風量則側で
        // 下限として適用し、ON/OFFホールドで負荷を上書きしない。
        if (holdAtMinimumCapacity && result.on && std::isfinite(nodeProps.required_heat_w) &&
            !aircon::airflow::isDuctCentralModel(nodeProps)) {
            nodeProps.required_heat_w =
                aircon::onoff::signedMinProcessHeatW(minMode, minProcessHeatW);
        }
        writeDomainLog(logFile, "空調", result.logMessage);
        applyOnOffControlResult(thermalNetwork, airconKey, nodeProps, result, humidityAbsTol,
                                allControlled, supplyHumidityChanged, outProposals);
    }

    return allControlled;
}

void AirconController::applyOnOffControlResult(
    ThermalNetwork& thermalNetwork,
    const std::string& airconKey,
    VertexProperties& nodeProps,
    const AirconControlResult& result,
    double humidityAbsTol,
    bool& allControlled,
    bool* supplyHumidityChanged,
    std::vector<AirconStateProposal>* outProposals) const {
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
        capacityLimitBracket_.erase(airconKey);
        nodeProps.humidity_search_low = nodeProps.humidity_search_high =
            std::numeric_limits<double>::quiet_NaN();
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
        if (!nodeProps.humidity_overcool_hold)
            nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
    }

    if (outProposals) {
        auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
        proposal.reasons = unitReasons;
        outProposals->push_back(std::move(proposal));
    }
}

bool AirconController::acceptedHumidityConverged(ThermalNetwork& thermalNetwork,
                                                  const FlowRateMap& flowRates,
                                                  double humidityAbsTol,
                                                  std::ostream& logs) const {
    for (const auto& key : getAirconKeys()) {
        const auto& node = thermalNetwork.getNode(key);
        if (!node.on) continue;
        const auto context = prepareRuntimeContext(key, thermalNetwork, node, flowRates);
        if (context.operationMode != OperationMode::Cooling) continue;
        const auto loads = aircon::latent::estimateLatentProcess(
            context.validData, context.operationMode, context.heatCapacity,
            context.airFlowRate, node, moistEnthalpyEnabled_);
        if (node.current_x > loads.supplyXUpperBound ||
            ((loads.condensationRateKgPerS > 0.0 || node.aircon_humidity_transition) &&
             aircon::latent::supplyHumidityNeedsUpdate(node, loads, humidityAbsTol))) {
            writeDomainLog(logs, "受入検査", "空調吹出湿度未達: aircon=" + key);
            return false;
        }
        for (const auto& outlet : aircon::network_utils::passiveSupplyOutletConstraints(thermalNetwork, node)) {
            const auto& discharge = thermalNetwork.getNode(outlet.outletKey);
            if (!aircon::latent::unsaturatedBeforeAndAfterSerialization(
                    discharge.current_t, discharge.current_x)) {
                std::ostringstream detail;
                detail << std::setprecision(17) << "空調出口湿度未達: outlet=" << outlet.outletKey
                       << ", x=" << thermalNetwork.getNode(outlet.outletKey).current_x
                       << ", guarded_target=" << outlet.outletXUpperBound
                       << ", supply_x=" << node.current_x
                       << ", supply_limit=" << outlet.supplyXUpperBound;
                writeDomainLog(logs, "受入検査", detail.str());
                return false;
            }
        }
    }
    return true;
}

void AirconController::validateAcceptedCapacity(ThermalNetwork& thermalNetwork,
                                                const FlowRateMap& flowRates) const {
    for (const auto& key : getAirconKeys()) {
        const auto& node = thermalNetwork.getNode(key);
        if (!node.on) continue;
        const auto context = prepareRuntimeContext(key, thermalNetwork, node, flowRates);
        const auto loads = aircon::latent::estimateLatentProcess(
            context.validData, context.operationMode, context.heatCapacity,
            context.airFlowRate, node, moistEnthalpyEnabled_);
        const double total = aircon::latent::totalHeatCapacity(loads);
        std::string source;
        const auto maximum = aircon::capacity::resolveMaxHeatCapacity(node, context.operationMode, source);
        if (!std::isfinite(total) ||
            (maximum && aircon::capacity::capacityExceedsLimit(total, *maximum))) {
            throw simulation::Error(simulation::ErrorCode::CapacityConstraintUnresolved,
                "Accepted-state capacity constraint failed: aircon=" + key +
                ", Q=" + std::to_string(total) + ", Qmax=" +
                (maximum ? std::to_string(*maximum) : "unset"));
        }
        if (aircon::humidity_control::enabled(node) && loads.sensibleHeatCapacity > 1.0 &&
            context.validData.indoorTemp > node.min_coil_temperature &&
            loads.coilTemp < node.min_coil_temperature - 1e-3) {
            throw simulation::Error(simulation::ErrorCode::CapacityConstraintUnresolved,
                "Accepted-state coil temperature below limit: aircon=" + key);
        }
    }
}

bool AirconController::checkAndAdjustCapacity(ThermalNetwork& thermalNetwork,
                                              VentilationNetwork& ventNetwork,
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
    struct Candidate {
        std::string key;
        RuntimeContext context;
        aircon::latent::LatentProcessResult loads;
    };
    std::vector<Candidate> candidates;
    // Evaluate the whole fleet from one state, then apply boundaries together.
    // Updating one unit must not alter another unit's capacity observation.
    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) continue;
        auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
        auto loads = aircon::latent::estimateLatentProcess(
            context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
            nodeProps, moistEnthalpyEnabled_);
        candidates.push_back({airconKey, context, loads});
    }
    bool boundariesChanged = false;
    for (const auto& candidate : candidates) {
        const auto& airconKey = candidate.key;
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        const auto& context = candidate.context;
        const auto& loads = candidate.loads;
        const double oldSupplyX = nodeProps.current_x;
        const double residual = loads.supplyX - oldSupplyX;
        auto& history = boundaryIterations_[airconKey];
        double relaxation = 0.5;
        const bool sameCandidate = history.regime == loads.regime &&
            std::abs(history.setpoint - nodeProps.current_pre_temp) < 1e-8 &&
            std::abs(history.flow - context.airFlowRate) < 1e-7;
        const double df = residual - history.residual;
        // Safeguarded scalar secant acceleration of the boundary fixed
        // point. Restart on a changed control candidate or a growing defect.
        if (sameCandidate && std::abs(df) > 1e-14 &&
            std::abs(residual) <= 1.25 * std::abs(history.residual)) {
            const double alpha = -(oldSupplyX - history.x) / df;
            if (std::isfinite(alpha) && alpha > 0.0) relaxation = std::clamp(alpha, 0.1, 1.0);
        }
        if (aircon::humidity_control::enabled(nodeProps) && !aircon::humidity_control::simultaneous(nodeProps))
            relaxation = 1.0;
        history = {oldSupplyX, residual, nodeProps.current_pre_temp, context.airFlowRate, loads.regime};
        const bool boundaryChanged = aircon::latent::applySupplyHumidityToAirconNode(
            thermalNetwork, airconKey, loads, xTol, relaxation);
        if (logVerbosity_ > 0 && (boundaryChanged || logVerbosity_ >= 2)) {
            logLatentBoundary(logs, "capacity", airconKey, context.validData, loads,
                              nodeProps, context.airFlowRate, oldSupplyX, boundaryChanged);
        }
        if (boundaryChanged) {
            boundariesChanged = true;
            if (supplyHumidityChanged) *supplyHumidityChanged = true;
            if (outProposals) {
                auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
                proposal.reasons = AirconRecomputeReason::SupplyHumidityChanged;
                outProposals->push_back(std::move(proposal));
            }
            // This load belongs to the old room moisture state. Do not
            // narrow a capacity bracket until its supply boundary settles.
            continue;
        }
    }
    if (boundariesChanged) return false;
    // Keep other humidistats' thermal boundaries fixed while a capacity root
    // is being bracketed. Moving them simultaneously invalidates the bracket
    // observations and can repeatedly reopen the RH search.
    const bool capacitySearchPending = !capacityLimitBracket_.empty();
    std::unordered_map<std::string, double> pendingHumidityCapacityRecovery;
    for (const auto& candidate : candidates) {
        const auto& airconKey = candidate.key;
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        const auto& context = candidate.context;
        const auto& loads = candidate.loads;
        {
            AirconRecomputeReason unitReasons = AirconRecomputeReason::None;
            const bool searchHumidity = aircon::humidity_control::enabled(nodeProps) && nodeProps.humidity_overcool_hold &&
                nodeProps.aircon_control_state != AirconControlState::CapacityLimited &&
                context.airFlowRate > 1e-8 && std::isfinite(nodeProps.humidity_supply_request);
            if (searchHumidity && capacitySearchPending) {
                // Revisit the deferred controller even if the other capacity
                // bracket closes without changing its setpoint in this pass.
                adjustmentMade = true;
                continue;
            }
            if (searchHumidity) {
                const double requested = nodeProps.current_requested_pre_temp;
                const double floor = requested - nodeProps.max_overcooling;
                if (!std::isfinite(nodeProps.humidity_search_low)) {
                    nodeProps.humidity_search_error_low = nodeProps.humidity_search_error_high =
                        std::numeric_limits<double>::quiet_NaN();
                    nodeProps.humidity_search_low = floor;
                    nodeProps.humidity_search_high = requested;
                }
                const double old = nodeProps.current_pre_temp;
                const double fanOnlyError = context.validData.airconTemp - context.validData.indoorTemp;
                if (!humidityFanTrialDone_.count(airconKey) &&
                    (std::abs(fanOnlyError) <= 1e-4 ||
                     (old >= requested - 1e-4 && fanOnlyError < 0.0))) {
                    humidityFanTrialDone_.insert(airconKey);
                    nodeProps.humidity_search_error_low = nodeProps.humidity_search_error_high =
                        std::numeric_limits<double>::quiet_NaN();
                    nodeProps.humidity_search_low = floor;
                    nodeProps.humidity_search_high = old;
                }
                const double desiredCoil = humidityFanTrialDone_.count(airconKey) && nodeProps.humidity_demand
                    ? aircon::latent::humidityRequestedOutletTemperature(
                        context.validData, nodeProps, context.airFlowRate, moistEnthalpyEnabled_)
                    : context.validData.indoorTemp;
                // Without reheat the coil cannot warm the return air. When
                // mixing alone satisfies RH, search for T_out == T_in (fan
                // only), subject to the same room overcooling bound.
                const double error = context.validData.airconTemp - desiredCoil;
                double next = old;
                if (std::abs(error) > 1e-4) {
                    if (error > 0.0) {
                        nodeProps.humidity_search_high = old;
                        nodeProps.humidity_search_error_high = error;
                        next = (old + nodeProps.humidity_search_low) / 2.0;
                    } else {
                        nodeProps.humidity_search_low = old;
                        nodeProps.humidity_search_error_low = error;
                        next = (old + nodeProps.humidity_search_high) / 2.0;
                    }
                    if (nodeProps.humidity_search_error_low < 0.0 && nodeProps.humidity_search_error_high > 0.0) {
                        const double fraction = std::clamp(-nodeProps.humidity_search_error_low /
                            (nodeProps.humidity_search_error_high-nodeProps.humidity_search_error_low), 0.1, 0.9);
                        next = nodeProps.humidity_search_low + fraction *
                            (nodeProps.humidity_search_high-nodeProps.humidity_search_low);
                    }
                    if (error > 0.0 && old-floor < 1e-4) next = floor;
                }
                // A small trial step is not success when coupling moved the
                // root. Check after narrowing/interpolation, then reopen locally.
                if (std::abs(next-old) <= 1e-5 && std::abs(error) > 1e-3 &&
                    ((error > 0.0 && old > floor+1e-4) ||
                     (error < 0.0 && old < requested-1e-4))) {
                    nodeProps.humidity_search_error_low = nodeProps.humidity_search_error_high =
                        std::numeric_limits<double>::quiet_NaN();
                    const double width = std::max(1e-3, 2.0*std::abs(error));
                    nodeProps.humidity_search_low = error > 0.0 ? std::max(floor, old-width) : old;
                    nodeProps.humidity_search_high = error > 0.0 ? old : std::min(requested, old+width);
                    next = 0.5*(nodeProps.humidity_search_low+nodeProps.humidity_search_high);
                }
                if (logVerbosity_ >= 2) {
                    std::ostringstream searchLog;
                    searchLog << airconKey << " humidity search: room=" << old << "->" << next
                        << ", coil_error=" << error << ", target_coil=" << desiredCoil
                        << ", supply_request=" << nodeProps.humidity_supply_request
                        << ", bracket=[" << nodeProps.humidity_search_low << "," << nodeProps.humidity_search_high << "]";
                    writeDomainLog(logs, "空調", searchLog.str());
                }
                if (std::abs(next - old) > 1e-5) {
                    nodeProps.current_pre_temp = next;
                    adjustmentMade = true;
                    if (outProposals) {
                        auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
                        proposal.reasons = AirconRecomputeReason::CapacitySetpointChanged;
                        outProposals->push_back(std::move(proposal));
                    }
                    continue; // re-solve the network before evaluating loads
                }
            }
            std::string sourceLabel = "unknown";
            auto maxHeatCapacity = aircon::capacity::resolveMaxHeatCapacity(
                nodeProps, context.operationMode, sourceLabel);
            if (aircon::humidity_control::enabled(nodeProps)) {
                // The coldest permitted wet coil also bounds cooling capacity.
                // Feed that bound into the shared capacity search so a large
                // sensible load cannot bypass the coil-temperature limit.
                auto floorData = context.validData;
                floorData.airconTemp = aircon::latent::minimumCoolingOutletTemperature(
                    context.validData, nodeProps, context.airFlowRate, moistEnthalpyEnabled_);
                floorData.additionalSupplyXUpperBound = std::numeric_limits<double>::infinity();
                auto floorNode = nodeProps;
                floorNode.control_mode = "humidity_priority";
                floorNode.current_pre_rh = std::numeric_limits<double>::quiet_NaN();
                if (aircon::humidity_control::simultaneous(nodeProps))
                    floorNode.ac_spec["latent_method"] = "none";
                const double floorSensible = archenv::DENSITY_DRY_AIR * context.airFlowRate *
                    archenv::SPECIFIC_HEAT_AIR * std::max(0.0, floorData.indoorTemp - floorData.airconTemp);
                const double coilLimit = aircon::latent::totalHeatCapacity(
                    aircon::latent::estimateLatentProcess(floorData, context.operationMode, floorSensible,
                        context.airFlowRate, floorNode, moistEnthalpyEnabled_));
                if (coilLimit > 1e-6 && (!maxHeatCapacity || coilLimit < *maxHeatCapacity)) {
                    // The watt acceptance band must not permit a temperature
                    // below the physical coil floor. Reserve that band inside
                    // the coil-derived limit, without changing equipment Qmax.
                    maxHeatCapacity = std::max(0.0,
                        (coilLimit-aircon::capacity::kCapacityConvergenceAbsTol) /
                        (1.0+aircon::capacity::kCapacityConvergenceRelTol));
                    sourceLabel = "min_coil_temperature";
                }
            }
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

            // Re-bracket a settled humidistat near its current capacity root.
            // Other rooms may change the load; retain this unit's constrained
            // humidity boundary while recovering either excess or spare capacity.
            if (maxHeatCapacity && nodeProps.humidity_overcool_hold &&
                nodeProps.aircon_control_state == AirconControlState::CapacityLimited &&
                !capacityLimitBracket_.count(airconKey) && context.airFlowRate > 1e-8) {
                const double old = nodeProps.current_pre_temp;
                const double width = std::max(0.01, 2.0 * std::abs(current - *maxHeatCapacity) /
                    (archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * context.airFlowRate));
                if (aircon::capacity::capacityExceedsLimit(current, *maxHeatCapacity)) {
                    capacityLimitBracket_[airconKey] = {old, std::min(50.0, old + width), false};
                }
            }
            if (nodeProps.humidity_overcool_hold && capacityLimitBracket_.count(airconKey))
                capacityLimitBracket_.at(airconKey).interpolate = true;
            bool unitAdjusted = false;
            if (maxHeatCapacity && aircon::capacity::capacityExceedsLimit(current, *maxHeatCapacity)) {
                if (aircon::airflow::isDuctCentralModel(nodeProps) &&
                    capacityFlowBoost_.insert(airconKey).second) {
                    const auto flow = aircon::airflow::computeTargetFlowFromProcessedHeat(
                        nodeProps, context.operationMode, *maxHeatCapacity, nodeProps.vol_zero, 0.0);
                    if (flow) {
                        bool fan = false;
                        bool changed = aircon::airflow::updateDuctCentralFanAffinity(
                            ventNetwork, nodeProps, context.operationMode, airconKey, *flow,
                            1e-3, &fan, false, false);
                        if (!fan) changed = aircon::airflow::updateDuctCentralCircuitFixedFlows(
                            ventNetwork, nodeProps.in_node, airconKey, *flow, 1e-4);
                        if (changed) {
                            capacityLimitBracket_.erase(airconKey);
                            nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
                            adjustmentMade = true;
                            if (outProposals) {
                                auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
                                proposal.reasons = AirconRecomputeReason::AirflowChanged;
                                outProposals->push_back(std::move(proposal));
                            }
                            writeDomainLog(logs, "空調", "能力判定: 最大側風量で要求設定を再評価 " + airconKey);
                            continue;
                        }
                    }
                }
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
            } else if (maxHeatCapacity &&
                       aircon::capacity::capacityWellBelowLimit(current, *maxHeatCapacity) &&
                       capacityLimitBracket_.count(airconKey)) {
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
                // A settled load inside the capacity band closes the search,
                // including when a different controller moved its root.
                if (maxHeatCapacity) capacityLimitBracket_.erase(airconKey);
                const bool atRequested = std::abs(nodeProps.current_pre_temp -
                    nodeProps.current_requested_pre_temp) <= 1e-5;
                if ((nodeProps.aircon_control_state != AirconControlState::CapacityLimited || atRequested) &&
                    !capacityLimitBracket_.count(airconKey)) {
                    if (!nodeProps.humidity_overcool_hold)
                        nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
                    nodeProps.aircon_control_state = AirconControlState::SetpointControlled;
                } else if (maxHeatCapacity && aircon::capacity::capacityWellBelowLimit(current, *maxHeatCapacity) &&
                           !capacityLimitBracket_.count(airconKey) &&
                           !aircon::humidity_control::enabled(nodeProps) &&
                           capacityRecoveryAttempted_.insert(airconKey).second) {
                    // Trial, not unconditional release: re-solve at the request
                    // and retain the hard capacity guard if that trial fails.
                    nodeProps.current_pre_temp = nodeProps.current_requested_pre_temp;
                    unitAdjusted = true;
                    oss << ", 要求設定への復帰を試算";
                }
            }
            if (!unitAdjusted && maxHeatCapacity && nodeProps.humidity_overcool_hold &&
                nodeProps.aircon_control_state == AirconControlState::CapacityLimited &&
                !capacityLimitBracket_.count(airconKey) && context.airFlowRate > 1e-8 &&
                aircon::capacity::capacityWellBelowLimit(current, *maxHeatCapacity) &&
                nodeProps.current_pre_temp > nodeProps.current_requested_pre_temp-nodeProps.max_overcooling+1e-3) {
                pendingHumidityCapacityRecovery[airconKey] = *maxHeatCapacity;
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
    // Recover spare capacity only after the free humidistats settle. Chasing
    // their intermediate loads would restart the capacity root at every step.
    if (!adjustmentMade) {
        for (const auto& candidate : candidates) {
            const auto found = pendingHumidityCapacityRecovery.find(candidate.key);
            if (found == pendingHumidityCapacityRecovery.end()) continue;
            auto& node = thermalNetwork.getNode(candidate.key);
            const double desiredCoil = node.humidity_demand
                ? aircon::latent::humidityRequestedOutletTemperature(
                    candidate.context.validData, node, candidate.context.airFlowRate, moistEnthalpyEnabled_)
                : candidate.context.validData.indoorTemp;
            if (node.current_pre_temp <= node.current_requested_pre_temp+1e-4 &&
                candidate.context.validData.airconTemp <= desiredCoil+1e-4) {
                // Demand has fallen below capacity. Return to the humidistat
                // instead of forcing full output without a humidity need.
                node.aircon_control_state = AirconControlState::SetpointControlled;
                node.humidity_search_low = node.humidity_search_high = std::numeric_limits<double>::quiet_NaN();
                adjustmentMade = true;
                if (outProposals) {
                    auto proposal = makeAirconStateProposalBase(candidate.key, node);
                    proposal.reasons = AirconRecomputeReason::CapacitySetpointChanged;
                    outProposals->push_back(std::move(proposal));
                }
                continue;
            }
            const double floor = node.current_requested_pre_temp - node.max_overcooling;
            const double current = aircon::latent::totalHeatCapacity(candidate.loads);
            const double width = std::max(0.01, 2.0 * (found->second - current) /
                (archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * candidate.context.airFlowRate));
            capacityLimitBracket_[candidate.key] = {std::max(floor, node.current_pre_temp - width),
                                                   node.current_pre_temp, false};
            capacityLimitBracket_.at(candidate.key).interpolate = true;
            std::ostringstream oss;
            bool changed = false;
            aircon::capacity::applyUnderCapacityBracketAdjustment(
                candidate.key, node, candidate.context.operationMode, candidate.context.validData.indoorTemp,
                candidate.context.airFlowRate, found->second, current, capacityLimitBracket_, oss, changed);
            adjustmentMade |= changed;
            writeDomainLog(logs, "空調", candidate.key + oss.str());
            if (changed && outProposals) {
                auto proposal = makeAirconStateProposalBase(candidate.key, node);
                proposal.reasons = AirconRecomputeReason::CapacitySetpointChanged;
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
    (void)supplyHumidityChanged;
    (void)humidityAbsTol;

    for (const auto& airconKey : getAirconKeys()) {
        auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!aircon::airflow::isDuctCentralModel(nodeProps)) {
            continue;
        }

        if (!nodeProps.on) {
            bool fan = false;
            const auto mode = aircon::network_utils::fanMode(nodeProps);
            bool changed = aircon::airflow::updateDuctCentralFanAffinity(
                ventNetwork, nodeProps, mode, airconKey, nodeProps.vol_zero, 1e-3, &fan, false, false);
            if (!fan) changed = aircon::airflow::updateDuctCentralCircuitFixedFlows(
                ventNetwork, nodeProps.in_node, airconKey, nodeProps.vol_zero, 1e-4);
            if (changed) {
                adjustmentMade = true;
                if (outProposals) {
                    auto proposal = makeAirconStateProposalBase(airconKey, nodeProps);
                    proposal.reasons = AirconRecomputeReason::AirflowChanged;
                    outProposals->push_back(std::move(proposal));
                }
            }
            continue;
        }

        auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
        const auto loads = aircon::latent::estimateLatentProcess(
            context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
            nodeProps, moistEnthalpyEnabled_);
        AirconRecomputeReason unitReasons = AirconRecomputeReason::None;
        const double measuredHeatW = aircon::latent::totalHeatCapacity(loads);

        double controlledRoomTemp = context.validData.setTemp;
        if (!nodeProps.set_node.empty()) {
            (void)aircon::network_utils::tryGetTempFromThermalNetwork(
                thermalNetwork, nodeProps.set_node, controlledRoomTemp);
        }
        const auto heatBasis = aircon::airflow::selectFlowHeatBasis(
            nodeProps, context.operationMode, measuredHeatW, controlledRoomTemp,
            nodeProps.aircon_flow_unmet_boost);
        nodeProps.aircon_flow_unmet_boost = heatBasis.unmetBoost;
        double heatForFlowW = heatBasis.heatW;
        if (capacityFlowBoost_.count(airconKey)) {
            std::string source;
            if (auto maximum = aircon::capacity::resolveMaxHeatCapacity(nodeProps, context.operationMode, source))
                heatForFlowW = *maximum;
        }
        const bool useExogenousHeatForFlow = heatBasis.exogenous;

        bool heldAtMinimumFlow = false;
        double minProcessHeatW = 0.0;
        if (const auto* spec = nodeProps.getAirconSpec()) {
            const char* modeKeyStr =
                (context.operationMode == OperationMode::Cooling) ? "cooling" : "heating";
            if (const auto qMinKW = spec->getMinProcess(modeKeyStr)) {
                if (*qMinKW > 0.0) minProcessHeatW = *qMinKW * 1000.0;
            }
        }
        auto targetFlowOpt = aircon::airflow::computeTargetFlowFromProcessedHeat(
            nodeProps, context.operationMode, heatForFlowW, nodeProps.vol_zero, minProcessHeatW,
            &heldAtMinimumFlow);
        if (aircon::humidity_control::enabled(nodeProps)) {
            // A humidity-only load must not collapse the fan to zero because
            // sensible demand is zero. Hold design flow for the ideal modes.
            const auto& spec = nodeProps.ac_spec;
            if (spec.contains("V_inner") && spec["V_inner"].contains("cooling") &&
                spec["V_inner"]["cooling"].contains("dsgn"))
                targetFlowOpt = spec["V_inner"]["cooling"]["dsgn"].get<double>();
        }
        if (!targetFlowOpt) {
            continue;
        }
        // ON/OFFのホールド状態はcontrolAllAirconsだけが更新する。
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

        // ファン枝があれば速度比だけを載せ、固定風量は触らない。無いときだけ vol を書く。
        // 達成風量が目標と 1e-3 m3/s 以内なら速度比も触らない（0.05→0.05 の再計算防止）。
        bool fanPresent = false;
        bool edgeUpdated = false;
        if (std::abs(currentFlow - targetFlow) > kAbsFlowMatch) {
            edgeUpdated = aircon::airflow::updateDuctCentralFanAffinity(
                ventNetwork, nodeProps, context.operationMode, nodeProps.key, targetFlow,
                kRelativeFlowTol, &fanPresent, /*increaseOnly=*/false, /*dampUpdate=*/true);
            if (!fanPresent && !nodeProps.in_node.empty()) {
                edgeUpdated = aircon::airflow::updateDuctCentralCircuitFixedFlows(
                    ventNetwork, nodeProps.in_node, nodeProps.key, targetFlow, flowTol);
            }
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
            << " DUCT_CENTRAL風量補正: 基準熱量(" << heatBasis.label << ")="
            << std::fixed << std::setprecision(2) << heatForFlowW << "W"
            << " (計測=" << measuredHeatW << "W)"
            << (fanPresent ? ", ファン速度比を更新" : "")
            << ", 風量 " << context.airFlowRate << "→" << targetFlow << " m3/s"
            << (heldAtMinimumFlow ? ", 最低風量維持" : "")
            << ", 再計算要求";
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
            } else if (dataType == "roomRelativeHumidity" || dataType == "humidityUnmet" || dataType == "temperatureUnmet") {
                const auto& room = thermalNetwork.getNode(nodeProps.set_node);
                const double rh = 100.0 * archenv::STANDARD_ATMOSPHERIC_PRESSURE * room.current_x /
                    (0.622 + room.current_x) / archenv::saturation_vapor_pressure(room.current_t);
                if (dataType == "roomRelativeHumidity") values[i] = rh;
                else if (dataType == "humidityUnmet") values[i] =
                    nodeProps.current_mode == "COOLING" && std::isfinite(nodeProps.current_pre_rh)
                    ? std::max(0.0, rh - nodeProps.current_pre_rh) : 0.0;
                else values[i] = nodeProps.current_mode == "COOLING"
                    ? std::max(0.0, room.current_t - nodeProps.current_requested_pre_temp)
                    : nodeProps.current_mode == "HEATING"
                        ? std::max(0.0, nodeProps.current_requested_pre_temp - room.current_t)
                        : nodeProps.current_mode == "AUTO"
                            ? std::abs(room.current_t - nodeProps.current_requested_pre_temp) : 0.0;
            } else if (dataType == "sensibleHeatCapacity" || dataType == "latentHeatCapacity" || dataType == "reheatCapacity") {
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
                values[i] = dataType == "reheatCapacity" ? loads.reheatCapacity : (dataType == "sensibleHeatCapacity")
                                ? loads.sensibleHeatCapacity
                                : loads.latentHeatCapacity;
            }
        } catch (...) {
            values[i] = 0.0;
        }
    }
    return values;
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
    capacityFlowBoost_.clear();
    capacityRecoveryAttempted_.clear();
    humidityFloorBlocked_.clear();
    humidityFanTrialDone_.clear();
    boundaryIterations_.clear();
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
