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
        writeLog(logs, std::string("　　[acmodel] ") + message);
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
        } catch (const std::exception& e) {
            writeLog(logs,
                     "　エラー: エアコンモデル初期化に失敗" + node.key + " - " + e.what());
        }
    }
    writeLog(logs, "  エアコンモデル初期化総数: " + std::to_string(initialized) + "台");
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

    double deltaT = inletTemp - outletTemp;
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
    if (!nodeProps.set_node.empty()) {
        data.setTemp = getTemp(nodeProps.set_node, "set_node");
    } else {
        data.setTemp = data.indoorTemp;
    }
    return data;
}

AirconController::RuntimeContext AirconController::prepareRuntimeContext(
    const std::string& airconKey,
    ThermalNetwork& thermalNetwork,
    const VertexProperties& nodeProps,
    const FlowRateMap& flowRates) const {
    RuntimeContext context{};
    context.validData = validateAirconData(airconKey, thermalNetwork, nodeProps);
    context.heatCapacity = calculateHeatCapacity(thermalNetwork, nodeProps.in_node, nodeProps.key, flowRates);
    context.airFlowRate = std::abs(getFlowRate(flowRates, nodeProps.in_node, nodeProps.key));

    auto mode = nodeProps.current_mode;
    if (mode == "AUTO") {
        mode = (context.validData.indoorTemp > context.validData.airconTemp) ? "COOLING" : "HEATING";
    }
    context.operationMode = (mode == "HEATING") ? "heating" : "cooling";
    return context;
}

bool AirconController::controlAllAircons(ThermalNetwork& thermalNetwork,
                                         double tolerance,
                                         std::ostream& logFile) const {
    bool allControlled = true;
    auto& graph = thermalNetwork.getGraph();
    const auto& keyToVertex = thermalNetwork.getKeyToVertex();

    for (const auto& [airconKey, _] : airconModels) {
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
            if (!nodeProps.set_node.empty()) {
                auto nodeIt = keyToVertex.find(nodeProps.set_node);
                if (nodeIt != keyToVertex.end()) {
                    graph[nodeIt->second].calc_t = !result.on;
                }
            }
        }
    }

    return allControlled;
}

namespace {
std::optional<double> resolveMaxHeatCapacity(const VertexProperties& nodeProps,
                                             const std::string& operationMode,
                                             std::string& source) {
    auto modeKey = toLowerCopy(operationMode);
    if (const auto* spec = nodeProps.getAirconSpec()) {
        if (auto value = spec->getCapacity(modeKey, "max")) {
            source = "Q." + modeKey + ".max";
            return *value * 1000.0;
        }
        if (auto value = spec->getCapacity(modeKey, "rtd")) {
            source = "Q." + modeKey + ".rtd";
            return *value * 1000.0;
        }
        double fallback = spec->getMaxHeatCapacity();
        if (fallback > 0.0) {
            source = "max_heat_capacity";
            return fallback;
        }
    }
    if (nodeProps.ac_spec.contains("max_heat_capacity")) {
        source = "max_heat_capacity";
        return nodeProps.ac_spec["max_heat_capacity"].get<double>();
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
    bool adjustmentMade = false;
    for (const auto& [airconKey, _] : airconModels) {
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on) {
            continue;
        }
        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            std::string sourceLabel = "unknown";
            auto maxHeatCapacity = resolveMaxHeatCapacity(nodeProps, context.operationMode, sourceLabel);
            double current = context.heatCapacity;

            std::ostringstream oss;
            oss << "　" << airconKey << " 最大処理熱量=";
            if (maxHeatCapacity) {
                oss << std::fixed << std::setprecision(2) << *maxHeatCapacity << "W";
                oss << " (" << sourceLabel << " 基準)";
            } else {
                oss << "N/A";
            }
            oss << ", 現在処理熱量=" << std::fixed << std::setprecision(2) << current << "W";
            if (maxHeatCapacity && current > *maxHeatCapacity) {
                oss << " → 超過";
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
                values[i] = calculateHeatCapacity(thermalNetwork, nodeProps.in_node, nodeProps.key, flowRates);
            } else if (dataType == "latentHeatCapacity") {
                values[i] = 0.0; // 潜熱は現状モデル化していない
            }
        } catch (...) {
            values[i] = 0.0;
        }
    }
    return values;
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
            power[i] = 0.0;
            continue;
        }
        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            auto* model = getModel(airconKey);
            if (!model) {
                throw std::runtime_error("初期化済みモデルがありません");
            }
            acmodel::InputData input;
            input.T_ex = context.validData.outdoorTemp;
            input.T_in = context.validData.indoorTemp;
            if (context.operationMode == "heating") {
                input.X_ex = archenv::jis::X_H_EX;
                input.X_in = archenv::jis::X_H_IN;
            } else {
                input.X_ex = archenv::jis::X_C_EX;
                input.X_in = archenv::jis::X_C_IN;
            }
            input.Q = context.heatCapacity;
            input.Q_S = context.heatCapacity;
            input.Q_L = 0.0;
            input.V_inner = context.airFlowRate;
            input.V_outer = kDefaultOuterFlowRate;
            auto result = model->estimateCOP(context.operationMode, input);
            if (logVerbosity_ >= 2) {
                for (const auto& msg : result.logMessages) {
                    writeLog(logs, msg);
                }
            }
            if (!result.valid) {
                throw std::runtime_error("COP推定に失敗しました");
            }
            double p = result.power * 1000.0; // kW -> W
            std::ostringstream detail;
            detail << "　　エアコン電力計算: " << airconKey
                   << " [" << context.operationMode << "]"
                   << " 処理=" << std::fixed << std::setprecision(2) << context.heatCapacity << "W"
                   << " 風量=" << context.airFlowRate << "m³/s"
                   << " 外気=" << context.validData.outdoorTemp << "°C"
                   << " 室内=" << context.validData.indoorTemp << "°C"
                   << " COP=" << result.COP
                   << " 電力=" << p << "W";
            writeLog(logs, detail.str());
            power[i] = p;
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: エアコン ") + airconKey + " - " + e.what());
            power[i] = 0.0;
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
            cop[i] = 0.0;
            continue;
        }
        try {
            auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
            auto* model = getModel(airconKey);
            if (!model) {
                throw std::runtime_error("初期化済みモデルがありません");
            }
            acmodel::InputData input;
            input.T_ex = context.validData.outdoorTemp;
            input.T_in = context.validData.indoorTemp;
            if (context.operationMode == "heating") {
                input.X_ex = archenv::jis::X_H_EX;
                input.X_in = archenv::jis::X_H_IN;
            } else {
                input.X_ex = archenv::jis::X_C_EX;
                input.X_in = archenv::jis::X_C_IN;
            }
            input.Q = context.heatCapacity;
            input.Q_S = context.heatCapacity;
            input.Q_L = 0.0;
            input.V_inner = context.airFlowRate;
            input.V_outer = kDefaultOuterFlowRate;
            auto result = model->estimateCOP(context.operationMode, input);
            if (!result.valid) {
                throw std::runtime_error("COP推定に失敗しました");
            }
            cop[i] = result.COP;
        } catch (const std::exception& e) {
            writeLog(logs, std::string("　　エラー: エアコン ") + airconKey + " - " + e.what());
            cop[i] = 0.0;
        }
    }
    return cop;
}

void AirconController::applyPreset(ThermalNetwork& thermalNetwork,
                                   std::ostream& logs) const {
    auto& graph = thermalNetwork.getGraph();
    for (const auto& [airconKey, _] : airconModels) {
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
