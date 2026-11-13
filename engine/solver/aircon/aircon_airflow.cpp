#include "aircon/aircon_airflow.h"

#include "aircon/aircon_capacity.h"
#include "aircon/aircon_operation_mode.h"
#include "types/aircon_control_state.h"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace {

inline std::optional<double> readSpecPositive(const nlohmann::json& spec,
                                              const std::string& key1,
                                              const std::string& key2,
                                              const std::string& key3) {
    if (!spec.is_object()) return std::nullopt;
    auto it1 = spec.find(key1);
    if (it1 == spec.end() || !it1->is_object()) return std::nullopt;
    auto it2 = it1->find(key2);
    if (it2 == it1->end() || !it2->is_object()) return std::nullopt;
    auto it3 = it2->find(key3);
    if (it3 == it2->end() || !it3->is_number()) return std::nullopt;
    const double v = it3->get<double>();
    if (!std::isfinite(v) || !(v > 0.0)) return std::nullopt;
    return v;
}

} // namespace

namespace aircon::airflow {

bool isDuctCentralModel(const VertexProperties& nodeProps) {
    return toLowerCopy(nodeProps.model) == "duct_central";
}

bool updateFixedFlowEdgeByNodePair(VentilationNetwork& ventNetwork,
                                   const std::string& fromNode,
                                   const std::string& toNode,
                                   double targetFlowM3s,
                                   double flowTolM3s) {
    auto& graph = ventNetwork.getGraph();
    bool updated = false;
    const double q = std::max(0.0, targetFlowM3s);
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        auto& edge = graph[e];
        if (edge.type != "fixed_flow") continue;
        const std::string source = graph[boost::source(e, graph)].key;
        const std::string target = graph[boost::target(e, graph)].key;
        const bool sameDirection = (source == fromNode && target == toNode);
        const bool reverseDirection = (source == toNode && target == fromNode);
        if (!sameDirection && !reverseDirection) continue;

        const double desired = sameDirection ? q : -q;
        if (std::abs(edge.current_vol - desired) <= flowTolM3s) continue;
        edge.current_vol = desired;
        edge.flow_rate = desired;
        updated = true;
    }
    return updated;
}

bool updateDuctCentralCircuitFixedFlows(VentilationNetwork& ventNetwork,
                                        const std::string& inNode,
                                        const std::string& airconNode,
                                        double targetFlowM3s,
                                        double flowTolM3s) {
    if (airconNode.empty()) return false;
    auto& graph = ventNetwork.getGraph();
    const double q = std::max(0.0, targetFlowM3s);
    // 目標 0 への極小残差は更新しない（0.00→0.00 の再計算ループ防止）。
    constexpr double kZeroMatch = 1e-3; // [m3/s]

    struct Cand {
        Graph::edge_descriptor edge;
        std::string source;
        std::string target;
        std::string subtype;
    };
    std::vector<Cand> cands;
    bool anyAirconSubtype = false;
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        auto& edge = graph[e];
        if (edge.type != "fixed_flow") continue;
        const std::string source = graph[boost::source(e, graph)].key;
        const std::string target = graph[boost::target(e, graph)].key;
        if (source != airconNode && target != airconNode) continue;
        Cand c;
        c.edge = e;
        c.source = source;
        c.target = target;
        c.subtype = edge.subtype;
        if (c.subtype == "aircon") anyAirconSubtype = true;
        cands.push_back(std::move(c));
    }

    bool hasForwardIntake = false;
    for (const auto& c : cands) {
        if (anyAirconSubtype && c.subtype != "aircon") continue;
        if (c.source == inNode && c.target == airconNode) {
            hasForwardIntake = true;
            break;
        }
    }

    bool updated = false;
    for (const auto& c : cands) {
        if (anyAirconSubtype && c.subtype != "aircon") continue;

        bool apply = false;
        double desired = 0.0;
        if (!inNode.empty() && c.source == inNode && c.target == airconNode) {
            // 還気: in -> 空調
            desired = q;
            apply = true;
        } else if (!inNode.empty() && c.source == airconNode && c.target == inNode && !hasForwardIntake) {
            // 還気が逆向き枝だけのとき: 空調 -> in の負号で吸込にする
            desired = -q;
            apply = true;
        } else if (c.source == airconNode) {
            // 吹出: 空調 -> out（in == out のループも含む）
            desired = q;
            apply = true;
        } else if (c.target == airconNode) {
            // 吹出が逆向きに格納されているとき: out -> 空調 の負号で吹出にする
            desired = -q;
            apply = true;
        }
        if (!apply) continue;

        auto& edge = graph[c.edge];
        if (std::abs(desired) <= 0.0 && std::abs(edge.current_vol) < kZeroMatch) continue;
        if (std::abs(edge.current_vol - desired) <= flowTolM3s) continue;
        edge.current_vol = desired;
        edge.flow_rate = desired;
        updated = true;
    }
    return updated;
}

namespace {

void ensureFanRatedCurve(EdgeProperties& edge) {
    if (edge.fan_curve_rated) return;
    edge.p_max_rated = edge.p_max;
    edge.p1_rated = edge.p1;
    edge.q_max_rated = edge.q_max;
    edge.q1_rated = edge.q1;
    edge.fan_curve_rated = true;
}

void applyFanAffinity(EdgeProperties& edge, double speedRatio) {
    const double lambda = std::clamp(speedRatio, 0.0, 1.0);
    // λ=0 でも piecewise の p_max>0 を保つ。流量側は 0 なので風は出ない。
    constexpr double kShutoffP = 1e-6;
    if (!(lambda > 0.0)) {
        edge.q_max = 0.0;
        edge.q1 = 0.0;
        edge.p_max = kShutoffP;
        edge.p1 = 0.0;
        return;
    }
    const double lambda2 = lambda * lambda;
    edge.q_max = lambda * edge.q_max_rated;
    edge.q1 = lambda * edge.q1_rated;
    edge.p_max = std::max(kShutoffP, lambda2 * edge.p_max_rated);
    edge.p1 = std::min(lambda2 * edge.p1_rated, edge.p_max);
}

} // namespace

bool updateDuctCentralFanAffinity(VentilationNetwork& ventNetwork,
                                  const VertexProperties& nodeProps,
                                  OperationMode operationMode,
                                  const std::string& airconNode,
                                  double targetFlowM3s,
                                  double ratioTol,
                                  bool* fanPresent,
                                  bool increaseOnly,
                                  bool dampUpdate) {
    if (fanPresent) *fanPresent = false;
    if (airconNode.empty()) return false;

    auto& graph = ventNetwork.getGraph();
    std::vector<Graph::edge_descriptor> fans;
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        auto& edge = graph[e];
        if (edge.type != "fan" || edge.subtype != "aircon") continue;
        const std::string source = graph[boost::source(e, graph)].key;
        const std::string target = graph[boost::target(e, graph)].key;
        if (source != airconNode && target != airconNode) continue;
        fans.push_back(e);
    }
    if (fans.empty()) return false;
    if (fanPresent) *fanPresent = true;

    const auto vDsgn = readSpecPositive(nodeProps.ac_spec, "V_inner", modeKey(operationMode), "dsgn");
    double lambda = 0.0;
    if (vDsgn && *vDsgn > 0.0) {
        lambda = std::clamp(std::max(0.0, targetFlowM3s) / *vDsgn, 0.0, 1.0);
    }

    bool updated = false;
    for (auto e : fans) {
        auto& edge = graph[e];
        ensureFanRatedCurve(edge);
        EdgeProperties desired = edge;
        applyFanAffinity(desired, lambda);
        const double qScale = std::max({edge.q_max_rated, edge.q_max, desired.q_max, 1e-6});
        const double pScale = std::max({edge.p_max_rated, edge.p_max, desired.p_max, 1e-6});
        if (increaseOnly && desired.q_max <= edge.q_max + ratioTol * qScale) continue;
        const bool same =
            std::abs(edge.q_max - desired.q_max) <= ratioTol * qScale &&
            std::abs(edge.q1 - desired.q1) <= ratioTol * qScale &&
            std::abs(edge.p_max - desired.p_max) <= ratioTol * pScale &&
            std::abs(edge.p1 - desired.p1) <= ratioTol * pScale;
        if (same) continue;
        const double currentLambda = edge.q_max_rated > 0.0
            ? std::clamp(edge.q_max / edge.q_max_rated, 0.0, 1.0) : lambda;
        // 風量依存の負荷へ一度で追従すると二周期になりうる。
        // 目標そのものへの残差で判定してから反復更新だけを緩和する。
        applyFanAffinity(edge, dampUpdate ? 0.5 * (currentLambda + lambda) : lambda);
        updated = true;
    }
    return updated;
}

bool increaseHeatingFanForCoupledSolve(VentilationNetwork& ventNetwork,
                                       const Graph& thermalGraph,
                                       std::ostream& logs,
                                       bool logEnabled) {
    bool updated = false;
    for (auto v : boost::make_iterator_range(boost::vertices(thermalGraph))) {
        const auto& node = thermalGraph[v];
        if (node.getTypeCode() != VertexProperties::TypeCode::Aircon || !node.on ||
            !isDuctCentralModel(node) || node.current_mode != "HEATING" ||
            node.set_node.empty() ||
            node.aircon_control_state == AirconControlState::CapacityLimited ||
            !std::isfinite(node.required_heat_w) || node.required_heat_w <= 0.0) {
            continue;
        }
        // 設定維持負荷と能力上限は通常の外側風量制御と同じ定義を使う。
        const auto basis = selectFlowHeatBasis(node, OperationMode::Heating,
                                               node.required_heat_w,
                                               node.current_requested_pre_temp);
        double minProcessW = 0.0;
        if (const auto* spec = node.getAirconSpec()) {
            if (const auto q = spec->getMinProcess("heating")) minProcessW = *q * 1000.0;
        }
        const auto target = computeTargetFlowFromProcessedHeat(
            node, OperationMode::Heating, basis.heatW, node.vol_zero, minProcessW);
        if (!target || !std::isfinite(*target)) continue;
        if (!updateDuctCentralFanAffinity(ventNetwork, node, OperationMode::Heating,
                                          node.key, *target, 1e-3, nullptr,
                                          /*increaseOnly=*/true)) {
            continue;
        }
        updated = true;
        if (logEnabled) {
            writeDomainLog(logs, "空調", node.key +
                " 暖房ファン先行補正: Qreq=" + std::to_string(node.required_heat_w) +
                " W, target=" + std::to_string(*target) +
                " m3/s; 温度候補を破棄して圧力・温度を再計算");
        }
    }
    return updated;
}

FlowHeatBasis selectFlowHeatBasis(const VertexProperties& nodeProps,
                                  OperationMode operationMode,
                                  double measuredHeatW,
                                  double controlledRoomTemp,
                                  bool previouslyUnmetBoost) {
    FlowHeatBasis basis;
    basis.heatW = measuredHeatW;
    const bool heatingDemand =
        isHeating(operationMode) &&
        std::isfinite(nodeProps.current_requested_pre_temp) &&
        std::isfinite(controlledRoomTemp);
    const bool coolingDemand =
        !isHeating(operationMode) &&
        std::isfinite(nodeProps.current_requested_pre_temp) &&
        std::isfinite(controlledRoomTemp);
    const double tSet = nodeProps.current_requested_pre_temp;
    const bool heatingEnter =
        heatingDemand && (controlledRoomTemp < tSet - kUnmetEnterK);
    const bool heatingHold = heatingDemand && previouslyUnmetBoost &&
                             (controlledRoomTemp < tSet - kUnmetExitK);
    const bool coolingEnter =
        coolingDemand && (controlledRoomTemp > tSet + kUnmetEnterK);
    const bool coolingHold = coolingDemand && previouslyUnmetBoost &&
                             (controlledRoomTemp > tSet + kUnmetExitK);
    const bool heatingUnmet = heatingEnter || heatingHold;
    const bool coolingUnmet = coolingEnter || coolingHold;
    basis.unmetBoost = heatingUnmet || coolingUnmet;
    const bool capacityLimited =
        nodeProps.aircon_control_state == AirconControlState::CapacityLimited;
    const bool setpointHeld = nodeProps.on && !nodeProps.set_node.empty() &&
                               !capacityLimited && !heatingUnmet && !coolingUnmet;
    basis.exogenous = capacityLimited || heatingUnmet || coolingUnmet || setpointHeld;

    std::string maxSource;
    const auto qMax = aircon::capacity::resolveMaxHeatCapacity(nodeProps, operationMode, maxSource);
    if (capacityLimited || heatingUnmet || coolingUnmet) {
        if (qMax) {
            basis.heatW = *qMax;
            basis.label = capacityLimited ? "能力上限" : "未達→能力上限";
        }
    } else if (setpointHeld) {
        if (std::isfinite(nodeProps.required_heat_w)) {
            basis.heatW = std::abs(nodeProps.required_heat_w);
            if (qMax && basis.heatW > *qMax) basis.heatW = *qMax;
            basis.label = "設定維持負荷";
        } else if (qMax) {
            basis.heatW = *qMax;
            basis.label = "設定固定→能力上限";
        }
    }
    return basis;
}

std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW,
                                                         double zeroLoadFlowM3s,
                                                         double minProcessHeatW,
                                                         bool* heldAtMinimum) {
    if (heldAtMinimum) *heldAtMinimum = false;
    const auto qRtdkW = readSpecPositive(nodeProps.ac_spec, "Q", modeKey(operationMode), "rtd");
    const auto vDsgn = readSpecPositive(nodeProps.ac_spec, "V_inner", modeKey(operationMode), "dsgn");
    if (!qRtdkW || !vDsgn) {
        return std::nullopt;
    }

    const double qRtdW = (*qRtdkW) * 1000.0;
    if (!(qRtdW > 0.0)) {
        return std::nullopt;
    }
    // 負荷 0 は零負荷風量（既定 0）。正の負荷は原則線形。
    // 最低能力は風量の下限。ON/OFF用のホールドで上限まで固定しない。
    // 生の必要負荷を残し、下限の前後で連続な風量則を使う。
    if (!(processedHeatW > 0.0)) {
        const double v0 = std::isfinite(zeroLoadFlowM3s) ? std::max(0.0, zeroLoadFlowM3s) : 0.0;
        return v0;
    }
    double heatForRatioW = processedHeatW;
    const bool holdFloor =
        std::isfinite(minProcessHeatW) && minProcessHeatW > 0.0 &&
        processedHeatW <= minProcessHeatW;
    if (holdFloor) {
        heatForRatioW = minProcessHeatW;
        if (heldAtMinimum) *heldAtMinimum = true;
    }
    const double ratio = std::clamp(heatForRatioW / qRtdW, 0.0, 1.0);
    return (*vDsgn) * ratio;
}

} // namespace aircon::airflow
