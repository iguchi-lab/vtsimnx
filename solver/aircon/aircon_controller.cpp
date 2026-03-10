#include "aircon/aircon_controller.h"

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
constexpr double kDefaultOuterFlowRate = 25.5 / 60.0;             // m^3/s
constexpr double kDefaultBypassFactor = 0.2;                      // [-]
constexpr double kDefaultSupplyRhPercent = 95.0;                  // [%]
constexpr int kSetpointSearchMaxIterations = 32;
constexpr double kSetpointSearchTolerance = 1e-3;                 // [degC]
constexpr double kSetpointFloor = 0.0;
constexpr double kSetpointCeiling = 50.0;
// 能力超過時二分探索の初期 bracket 幅。いきなり極端な設定にせず、まず現在設定から幅だけ動かした範囲で探索する。
constexpr double kCapacityLimitInitialBracketWidth = 10.0;        // [degC]
// 処理熱量が最大能力に「十分近い」とみなす許容（相対＋絶対）。0.1% + 1W。
constexpr double kCapacityConvergenceRelTol = 0.001;
constexpr double kCapacityConvergenceAbsTol = 1.0;               // [W]

inline std::string toLowerCopy(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

inline double getFlowRate(const FlowRateMap& flowRates,
                          const std::string& source,
                          const std::string& target) {
    auto direct = flowRates.find({source, target});
    if (direct != flowRates.end()) {
        return direct->second;
    }
    auto reverse = flowRates.find({target, source});
    if (reverse != flowRates.end()) {
        return -reverse->second;
    }
    return 0.0;
}

inline double clampHeatCapacity(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

inline double estimateHeatCapacityForSetpoint(const std::string& operationMode,
                                              double inletTemp,
                                              double setpoint,
                                              double airFlowRate) {
    if (std::abs(airFlowRate) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    double deltaT = 0.0;
    if (operationMode == "heating") {
        deltaT = std::max(0.0, setpoint - inletTemp);
    } else {
        deltaT = std::max(0.0, inletTemp - setpoint);
    }
    return clampHeatCapacity(kAirDensity * kAirSpecificHeat * std::abs(airFlowRate) * deltaT);
}

inline std::optional<double> findCapacityLimitedSetpoint(const std::string& operationMode,
                                                         double inletTemp,
                                                         double currentSetpoint,
                                                         double airFlowRate,
                                                         double maxHeatCapacity) {
    if (!std::isfinite(inletTemp) || !std::isfinite(currentSetpoint) || !std::isfinite(maxHeatCapacity)) {
        return std::nullopt;
    }

    if (operationMode == "heating") {
        double feasible = std::min(inletTemp, currentSetpoint);
        double infeasible = std::max(inletTemp, currentSetpoint);
        if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, infeasible, airFlowRate) <= maxHeatCapacity) {
            return std::nullopt;
        }
        for (int i = 0; i < kSetpointSearchMaxIterations && (infeasible - feasible) > kSetpointSearchTolerance; ++i) {
            const double mid = 0.5 * (feasible + infeasible);
            if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, mid, airFlowRate) <= maxHeatCapacity) {
                feasible = mid;
            } else {
                infeasible = mid;
            }
        }
        return feasible;
    }

    double infeasible = std::min(inletTemp, currentSetpoint);
    double feasible = std::max(inletTemp, currentSetpoint);
    if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, infeasible, airFlowRate) <= maxHeatCapacity) {
        return std::nullopt;
    }
    for (int i = 0; i < kSetpointSearchMaxIterations && (feasible - infeasible) > kSetpointSearchTolerance; ++i) {
        const double mid = 0.5 * (feasible + infeasible);
        if (estimateHeatCapacityForSetpoint(operationMode, inletTemp, mid, airFlowRate) <= maxHeatCapacity) {
            feasible = mid;
        } else {
            infeasible = mid;
        }
    }
    return feasible;
}

static inline acmodel::InputData buildAcmodelInput(const std::string& /*operationMode*/,
                                                   const AirconValidationData& validData,
                                                   double sensibleHeatCapacity,
                                                   double latentHeatCapacity,
                                                   double airFlowRate) {
    acmodel::InputData input;
    input.T_ex = validData.outdoorTemp;
    input.T_in = validData.indoorTemp;
    // 湿度（絶対湿度）は呼び出し側で「欠損時の警告 + JISフォールバック」を行う
    input.X_ex = validData.outdoorX;
    input.X_in = validData.indoorX;
    input.Q_S = sensibleHeatCapacity;
    input.Q_L = latentHeatCapacity;
    input.Q = sensibleHeatCapacity + latentHeatCapacity;
    input.V_inner = airFlowRate;
    input.V_outer = kDefaultOuterFlowRate;
    return input;
}

struct LatentProcessResult {
    double sensibleHeatCapacity = 0.0; // [W]
    double latentHeatCapacity = 0.0;   // [W]
    double supplyX = 0.0;              // [kg/kg(DA)]
    double coilTemp = 0.0;             // [degC]
    double coilX = 0.0;                // [kg/kg(DA)]
    double supplyRhPercent = 0.0;      // [%]
    double bfRhPercentBeforeFallback = 0.0; // [%]
    bool rhExceeded = false;
    bool usedRh95Fallback = false;
};

inline double readBypassFactor(const VertexProperties& nodeProps) {
    const auto& spec = nodeProps.ac_spec;
    auto read = [&](const char* key, double& out) -> bool {
        if (!spec.is_object() || !spec.contains(key) || !spec[key].is_number()) return false;
        out = spec[key].get<double>();
        return std::isfinite(out);
    };
    double bf = kDefaultBypassFactor;
    if (!read("bf", bf) && !read("BF", bf) && !read("bypass_factor", bf)) {
        return kDefaultBypassFactor;
    }
    return std::clamp(bf, 0.0, 0.99);
}

inline std::string readLatentMethod(const VertexProperties& nodeProps) {
    const auto& spec = nodeProps.ac_spec;
    if (!spec.is_object() || !spec.contains("latent_method") || !spec["latent_method"].is_string()) {
        return "rh95";
    }
    return toLowerCopy(spec["latent_method"].get<std::string>());
}

inline LatentProcessResult estimateLatentByBypassFactor(const AirconValidationData& validData,
                                                        const std::string& operationMode,
                                                        double sensibleHeatCapacity,
                                                        double airFlowRate,
                                                        const VertexProperties& nodeProps) {
    LatentProcessResult result;
    result.sensibleHeatCapacity = std::max(0.0, sensibleHeatCapacity);
    result.supplyX = std::max(0.0, validData.indoorX);

    if (operationMode != "cooling") return result;
    if (!(airFlowRate > std::numeric_limits<double>::epsilon())) return result;

    const double tIn = validData.indoorTemp;
    // 潜熱処理量は「吸込(in)→吹出(out)」の空気状態差で評価する。
    // ここでの吹出温度は制御点(set)ではなく、aircon ノード温度（out相当）を使う。
    const double tOut = validData.airconTemp;
    const double xIn = std::max(0.0, validData.indoorX);
    if (!(tIn > tOut)) {
        result.supplyX = xIn;
        return result;
    }

    const std::string latentMethod = readLatentMethod(nodeProps);
    if (latentMethod == "none") {
        result.supplyX = xIn;
        return result;
    }

    auto applyRh95 = [&]() {
        const double x95 = std::max(0.0, archenv::absolute_humidity(tOut, kDefaultSupplyRhPercent));
        result.supplyX = std::min(xIn, x95);
        const double xSatOut = std::max(0.0, archenv::absolute_humidity(tOut, 100.0));
        if (xSatOut > std::numeric_limits<double>::epsilon()) {
            result.supplyRhPercent = 100.0 * result.supplyX / xSatOut;
        }
    };

    if (latentMethod == "bf") {
        const double bf = readBypassFactor(nodeProps);
        if (!(bf < 1.0)) {
            result.supplyX = xIn;
        } else {
            // θcoil = θin - (θin-θout)/(1-BF)
            const double tCoil = tIn - (tIn - tOut) / std::max(1e-9, (1.0 - bf));
            const double xCoil = std::max(0.0, archenv::absolute_humidity(tCoil, 100.0));
            result.coilTemp = tCoil;
            result.coilX = xCoil;

            // 空気線図上の直線（吸込点-コイル飽和点）と吹出温度との交点
            const double denom = (tIn - tCoil);
            double xOut = xIn;
            if (std::abs(denom) > 1e-9) {
                const double ratio = (tIn - tOut) / denom;
                xOut = xIn + (xCoil - xIn) * ratio;
            }
            if (!std::isfinite(xOut)) xOut = xIn;
            xOut = std::max(0.0, xOut);
            result.supplyX = std::min(xOut, xIn); // 冷房コイルで加湿はしない前提
        }

        const double xSatOut = std::max(0.0, archenv::absolute_humidity(tOut, 100.0));
        if (xSatOut > std::numeric_limits<double>::epsilon()) {
            result.supplyRhPercent = 100.0 * result.supplyX / xSatOut;
            result.bfRhPercentBeforeFallback = result.supplyRhPercent;
            result.rhExceeded = (result.supplyRhPercent > 100.0 + 1e-6);
        }
        if (result.rhExceeded) {
            applyRh95();
            result.usedRh95Fallback = true;
            result.rhExceeded = false;
        }
    } else {
        applyRh95();
    }

    const double deltaX = std::max(0.0, xIn - result.supplyX);
    result.latentHeatCapacity =
        std::max(0.0,
                 kAirDensity * std::abs(airFlowRate) *
                     archenv::vapor_latent_heat(tOut) * deltaX);
    return result;
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

namespace {
static inline bool tryGetTempFromThermalNetwork(const ThermalNetwork& thermalNetwork,
                                                const std::string& nodeKey,
                                                double& outTemp) {
    if (nodeKey.empty()) return false;
    const auto& keyToV = thermalNetwork.getKeyToVertex();
    auto it = keyToV.find(nodeKey);
    if (it == keyToV.end()) return false;
    outTemp = thermalNetwork.getGraph()[it->second].current_t;
    return true;
}
} // namespace

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
    if (!tryGetTempFromThermalNetwork(thermalNetwork, inNode, inletTemp) ||
        !tryGetTempFromThermalNetwork(thermalNetwork, airconNode, outletTemp)) {
        return 0.0;
    }

    double flowRate = getFlowRate(flowRates, inNode, airconNode);
    if (std::abs(flowRate) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    // 処理熱量は「暖房/冷房どちらでも +W（大きさ）」として扱う。
    // - heating: 出口(吹出) > 入口 のときのみ加熱。出口 <= 入口なら 0（加熱していない）。
    // - cooling: 入口 > 出口 のときのみ除熱。入口 <= 出口なら 0。
    // モードと逆の向きの温度差は 0 とする。abs のみで正にするのは、熱ソルバが極端な設定で
    // 出口温度が暴れたときに巨大な処理熱量にならないようにするため。
    double deltaT = 0.0;
    if (mode == "heating") {
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
        if (!tryGetTempFromThermalNetwork(thermalNetwork, nodeName, t)) {
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
    auto getX = [&](const std::string& nodeName) -> double {
        if (nodeName.empty()) return 0.0;
        const auto& keyToV = thermalNetwork.getKeyToVertex();
        auto it = keyToV.find(nodeName);
        if (it == keyToV.end()) return 0.0;
        return thermalNetwork.getGraph()[it->second].current_x;
    };
    data.outdoorX = getX(nodeProps.outside_node);
    data.indoorX = getX(nodeProps.in_node);
    return data;
}

AirconController::RuntimeContext AirconController::prepareRuntimeContext(
    const std::string& airconKey,
    ThermalNetwork& thermalNetwork,
    const VertexProperties& nodeProps,
    const FlowRateMap& flowRates) const {
    RuntimeContext context{};
    context.validData = validateAirconData(airconKey, thermalNetwork, nodeProps);
    context.airFlowRate = std::abs(getFlowRate(flowRates, nodeProps.in_node, nodeProps.key));

    auto mode = nodeProps.current_mode;
    if (mode == "AUTO") {
        mode = (context.validData.indoorTemp > context.validData.airconTemp) ? "COOLING" : "HEATING";
    }
    context.operationMode = (mode == "HEATING") ? "heating" : "cooling";
    context.heatCapacity = calculateHeatCapacity(thermalNetwork, context.operationMode,
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
            (void)tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.set_node, currentTemp);
        } else {
            if (!tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.key, currentTemp)) {
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

namespace {
// 能力超過時二分探索: bracket (tLow, tHigh) を1ステップ更新し、新 setpoint と収束フラグを返す。
struct CapacityLimitBracketResult {
    double newSetpoint = 0.0;
    bool bracketConverged = false;
    bool capacityConverged = false;
};

void initCapacityLimitBracket(bool heating, double currentPreTemp, double& tLow, double& tHigh) {
    const double w = kCapacityLimitInitialBracketWidth;
    if (heating) {
        tLow = std::max(kSetpointFloor, currentPreTemp - w);
        tHigh = currentPreTemp;
    } else {
        tLow = currentPreTemp;
        tHigh = std::min(kSetpointCeiling, currentPreTemp + w);
    }
}

CapacityLimitBracketResult stepCapacityLimitBracket(bool heating, double maxQ, double currentQ,
                                                    double currentSetpoint,
                                                    double& tLow, double& tHigh) {
    // 処理熱量 > 最大能力: bracket を狭める（暖房なら tHigh=現設定、冷房なら tLow=現設定）
    if (currentQ > maxQ) {
        if (heating) {
            tHigh = currentSetpoint;
        } else {
            tLow = currentSetpoint;
        }
    } else {
        // 処理熱量 < 最大能力: 設定を上げる方向に広げる（暖房: tLow=現設定、冷房: tHigh=現設定）
        if (heating) {
            tLow = currentSetpoint;
        } else {
            tHigh = currentSetpoint;
        }
    }
    const double newSetpoint = 0.5 * (tLow + tHigh);
    const bool bracketConverged = (tHigh - tLow) <= kSetpointSearchTolerance;
    const bool capacityConverged =
        std::abs(currentQ - maxQ) <= (maxQ * kCapacityConvergenceRelTol + kCapacityConvergenceAbsTol);
    return {newSetpoint, bracketConverged, capacityConverged};
}

std::optional<double> resolveMaxHeatCapacity(const VertexProperties& nodeProps,
                                             const std::string& operationMode,
                                             std::string& source) {
    auto modeKey = toLowerCopy(operationMode);
    if (const auto* spec = nodeProps.getAirconSpec()) {
        auto value = spec->getCapacityMaxForMode(modeKey);
        if (value && *value > 0) {
            source = "Q." + modeKey + ".max";
            return *value * 1000.0;
        }
    }
    return std::nullopt;
}
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
            const auto loads = estimateLatentByBypassFactor(
                context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
            std::string sourceLabel = "unknown";
            auto maxHeatCapacity = resolveMaxHeatCapacity(nodeProps, context.operationMode, sourceLabel);
            const double sensibleQ = std::max(0.0, loads.sensibleHeatCapacity);
            const double latentQ = std::max(0.0, loads.latentHeatCapacity);
            double current = sensibleQ + latentQ;

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
                // 目的: 処理熱量が最大能力（maxHeatCapacity）と等しくなる設定温度を求める。
                // このタイムステップのみ見かけ上設定温度を変えて再計算し、能力内に収まる解を得る。
                // current_pre_temp の変更は当該タイムステップの熱計算にのみ使われ、次ステップでは入力時系列の設定温度に戻る。
                auto limitedSetpoint = findCapacityLimitedSetpoint(
                    context.operationMode,
                    context.validData.indoorTemp,
                    nodeProps.current_pre_temp,
                    context.airFlowRate,
                    *maxHeatCapacity);
                const double previousSetpoint = nodeProps.current_pre_temp;
                if (limitedSetpoint) {
                    nodeProps.current_pre_temp = *limitedSetpoint;
                    adjustmentMade = true;
                    oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << *limitedSetpoint << "°C";
                    oss << ", 再計算要求";
                } else {
                    // 二分探索で有効な設定温度が見つからない場合。吹き込み温度は参照しない。
                    // 処理熱量が最大能力と同じになる設定温度を、熱ソルバの解を使った二分探索で求める。
                    const bool heating = (context.operationMode == "heating");
                    const double maxQ = *maxHeatCapacity;
                    const double currentQ = current;
                    auto it = capacityLimitBracket_.find(airconKey);
                    if (it == capacityLimitBracket_.end()) {
                        std::pair<double, double> bracket;
                        initCapacityLimitBracket(heating, nodeProps.current_pre_temp, bracket.first, bracket.second);
                        capacityLimitBracket_[airconKey] = bracket;
                        it = capacityLimitBracket_.find(airconKey);
                    }
                    double& tLow = it->second.first;
                    double& tHigh = it->second.second;
                    const auto result = stepCapacityLimitBracket(heating, maxQ, currentQ,
                                                                nodeProps.current_pre_temp, tLow, tHigh);
                    nodeProps.current_pre_temp = result.newSetpoint;
                    if (result.capacityConverged) {
                        oss << " → 二分探索収束 設定温度=" << result.newSetpoint << "°C（処理熱量≒最大能力）";
                    } else if (result.bracketConverged) {
                        adjustmentMade = true;
                        oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint << "°C（bracket収束・最終解のため再計算1回）";
                    } else {
                        adjustmentMade = true;
                        oss << " → 超過, 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint << "°C（能力=" << maxQ << "Wに合わせて二分探索）";
                        oss << ", 再計算要求";
                    }
                }
            } else if (maxHeatCapacity && current < *maxHeatCapacity && capacityLimitBracket_.count(airconKey)) {
                // 処理熱量が最大能力と同じになる設定温度を探す途中で、下げすぎて処理熱量が 0 等になった場合。
                // bracket を更新して設定温度を上げ、再計算して「処理熱量＝最大能力」に近づける。
                const double previousSetpoint = nodeProps.current_pre_temp;
                oss << " → 不足（能力=" << *maxHeatCapacity << "Wに合わせて二分探索継続）";
                const bool heating = (context.operationMode == "heating");
                const double maxQ = *maxHeatCapacity;
                auto it = capacityLimitBracket_.find(airconKey);
                double& tLow = it->second.first;
                double& tHigh = it->second.second;
                const auto result = stepCapacityLimitBracket(heating, maxQ, current,
                                                             nodeProps.current_pre_temp, tLow, tHigh);
                nodeProps.current_pre_temp = result.newSetpoint;
                if (result.capacityConverged) {
                    oss << ", 二分探索収束 設定温度=" << result.newSetpoint << "°C（処理熱量≒最大能力）";
                } else if (result.bracketConverged) {
                    adjustmentMade = true;
                    oss << ", 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint << "°C（bracket収束・最終解のため再計算1回）";
                } else {
                    adjustmentMade = true;
                    oss << ", 設定温度補正=" << previousSetpoint << "→" << result.newSetpoint << "°C, 再計算要求";
                }
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
                if (tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.key, t)) values[i] = t;
            } else if (dataType == "inTemp") {
                if (!nodeProps.in_node.empty()) {
                    double t = 0.0;
                    if (tryGetTempFromThermalNetwork(thermalNetwork, nodeProps.in_node, t)) values[i] = t;
                }
            } else if (dataType == "flow") {
                values[i] = std::abs(getFlowRate(flowRates, nodeProps.in_node, nodeProps.key));
            } else if (dataType == "sensibleHeatCapacity") {
                // 処理熱量は「実機出力」として扱うため、OFF時は 0 を返す。
                if (!nodeProps.on) {
                    values[i] = 0.0;
                    continue;
                }
                auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
                values[i] = estimateLatentByBypassFactor(
                    context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps)
                                .sensibleHeatCapacity;
            } else if (dataType == "latentHeatCapacity") {
                if (!nodeProps.on) {
                    values[i] = 0.0;
                    continue;
                }
                auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
                values[i] = estimateLatentByBypassFactor(
                    context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps)
                                .latentHeatCapacity;
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
    const auto loads = estimateLatentByBypassFactor(
        context.validData, context.operationMode, context.heatCapacity, context.airFlowRate, nodeProps);
    // 吹出絶対湿度をエアコンノードへ反映する（humidity_x 出力および次ステップ初期値に利用）。
    // ここでは冷房時のみ有効値が入り、暖房/無効時は入力湿度（実質据え置き）となる。
    thermalNetwork.getNode(airconKey).current_x = loads.supplyX;
    acmodel::InputData input =
        buildAcmodelInput(context.operationMode, context.validData,
                          loads.sensibleHeatCapacity, loads.latentHeatCapacity,
                          context.airFlowRate);

    const bool heating = (context.operationMode == "heating");
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
             << " [" << context.operationMode << "]";
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
    auto result = model->estimateCOP(context.operationMode, input);
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
               << " [" << context.operationMode << "]"
               << " 顕熱=" << std::fixed << std::setprecision(2) << loads.sensibleHeatCapacity << "W"
               << " 潜熱=" << std::fixed << std::setprecision(2) << loads.latentHeatCapacity << "W"
               << " 合計=" << std::fixed << std::setprecision(2) << (loads.sensibleHeatCapacity + loads.latentHeatCapacity) << "W"
               << " 風量=" << context.airFlowRate << "m³/s"
               << " 外気=" << context.validData.outdoorTemp << "°C"
               << " 室内=" << context.validData.indoorTemp << "°C"
               << " COP=" << result.COP
               << " 電力=" << powerW << "W";
        writeLog(logs, detail.str());
    }
    return {powerW, result.COP};
}

std::vector<double> AirconController::calculatePowerValues(ThermalNetwork& thermalNetwork,
                                                           const FlowRateMap& flowRates,
                                                           std::ostream& logs) const {
    const auto& keys = getAirconKeys();
    std::vector<double> power(keys.size(), 0.0);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& airconKey = keys[i];
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) {
            continue;
        }
        try {
            const auto pair = estimatePowerAndCOPForAircon(airconKey, thermalNetwork, nodeProps, flowRates, logs, true);
            power[i] = pair.first;
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: エアコン ") + airconKey + " - " + e.what());
        }
    }
    return power;
}

std::vector<double> AirconController::calculateCOPValues(ThermalNetwork& thermalNetwork,
                                                         const FlowRateMap& flowRates,
                                                         std::ostream& logs) const {
    const auto& keys = getAirconKeys();
    std::vector<double> cop(keys.size(), 0.0);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& airconKey = keys[i];
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) {
            continue;
        }
        try {
            const auto pair = estimatePowerAndCOPForAircon(airconKey, thermalNetwork, nodeProps, flowRates, logs, false);
            cop[i] = pair.second;
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: エアコン ") + airconKey + " - " + e.what());
        }
    }
    return cop;
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
        nodeProps.on = false;
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
