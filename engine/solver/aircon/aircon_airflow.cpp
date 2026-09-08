#include "aircon/aircon_airflow.h"

#include "aircon/aircon_operation_mode.h"

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

std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW,
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
    // 負荷 0 は風量 0。正でも Q.min 未満は最低風量に留める。
    // 部分負荷を 0 近くまで落とすと、固定温度の移流が連成を壊す。
    if (!(processedHeatW > 0.0)) {
        return 0.0;
    }
    const double ratio = std::clamp(processedHeatW / qRtdW, 0.0, 1.0);
    double usedRatio = ratio;
    const auto qMinkW = readSpecPositive(nodeProps.ac_spec, "Q", modeKey(operationMode), "min");
    if (qMinkW && *qMinkW < *qRtdkW) {
        const double minRatio = *qMinkW / *qRtdkW;
        if (ratio < minRatio) {
            usedRatio = minRatio;
            if (heldAtMinimum) *heldAtMinimum = true;
        }
    }
    return (*vDsgn) * usedRatio;
}

} // namespace aircon::airflow
