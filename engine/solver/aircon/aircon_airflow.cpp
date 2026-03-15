#include "aircon/aircon_airflow.h"

#include "aircon/aircon_operation_mode.h"

#include <algorithm>
#include <cmath>
#include <optional>

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

std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW) {
    const auto qRtdkW = readSpecPositive(nodeProps.ac_spec, "Q", modeKey(operationMode), "rtd");
    const auto vDsgn = readSpecPositive(nodeProps.ac_spec, "V_inner", modeKey(operationMode), "dsgn");
    if (!qRtdkW || !vDsgn) {
        return std::nullopt;
    }

    const double qRtdW = (*qRtdkW) * 1000.0;
    if (!(qRtdW > 0.0)) {
        return std::nullopt;
    }
    const double ratio = std::clamp(processedHeatW / qRtdW, 0.0, 1.0);
    return (*vDsgn) * ratio;
}

} // namespace aircon::airflow
