#pragma once

#include "core/thermal/heat_calculation.h"
#include "core/thermal/thermal_direct_response.h"
#include "../../archenv/include/archenv.h"

#include <vector>

// 熱枝の有効判定・符号・行列寄与・後処理熱流量を一箇所に集約する。
// build / RHS precompute / postprocess が同じ定義を参照する。

namespace thermal_edge_physics {

inline bool isActive(const EdgeProperties& ep) {
    return ep.current_enabled;
}

// 熱収支定義: Σ流入 + heat_source = 0 ⇔ A*T = -heat_source + (枝の温度非依存項)
inline double heatSourceToRhs(double heatSource) {
    return -heatSource;
}

// 温度に比例する係数だけを組み立てる（conductance / advection / response の a0,b0）。
template <typename AddCoeffOrKnown>
inline void assembleTemperatureCoeffsAtNode(const Graph& graph,
                                            Edge edge,
                                            Vertex v,
                                            double f,
                                            AddCoeffOrKnown&& addCoeffOrKnown) {
    const auto& ep = graph[edge];
    if (!isActive(ep)) return;

    const Vertex sv = boost::source(edge, graph);
    const Vertex tv = boost::target(edge, graph);
    const auto tc = ep.getTypeCode();

    if (tc == EdgeProperties::TypeCode::Conductance) {
        const double k = ep.conductance;
        if (sv == v) {
            addCoeffOrKnown(sv, f * (-k));
            addCoeffOrKnown(tv, f * (+k));
        } else {
            addCoeffOrKnown(sv, f * (+k));
            addCoeffOrKnown(tv, f * (-k));
        }
        return;
    }

    if (tc == EdgeProperties::TypeCode::Advection) {
        double flowRate = ep.flow_rate;
        if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) return;
        const double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
        if (flowRate > 0) {
            if (tv == v && !(ep.is_aircon_inflow && graph[tv].on)) {
                addCoeffOrKnown(sv, f * (+mDotCp));
                addCoeffOrKnown(tv, f * (-mDotCp));
            }
        } else {
            if (sv == v && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on)) {
                addCoeffOrKnown(tv, f * (-mDotCp));
                addCoeffOrKnown(sv, f * (+mDotCp));
            }
        }
        return;
    }

    if (tc == EdgeProperties::TypeCode::ResponseConduction) {
        using thermal_direct_response::responseArea;
        const double area = responseArea(ep);
        if (sv == v) {
            const double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0];
            const double b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0];
            addCoeffOrKnown(sv, f * (-area * a0));
            addCoeffOrKnown(tv, f * (-area * b0));
        } else {
            const double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0];
            const double b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0];
            addCoeffOrKnown(tv, f * (-area * a0));
            addCoeffOrKnown(sv, f * (-area * b0));
        }
    }
}

// heat_generation の RHS 符号（A*T = b 側）。sv 側 +q、tv 側 -q。
inline double heatGenerationRhsSignAtNode(Vertex v, Vertex sv) {
    return (sv == v) ? +1.0 : -1.0;
}

inline bool responseHistIsSrcSide(Vertex v, Vertex sv) {
    return sv == v;
}

// フル行列構築向け: 温度係数 + 可変 RHS（発熱・応答履歴）を同時に適用。
template <typename AddCoeffOrKnown, typename AddRhs>
inline void assembleEdgeAtNode(const Graph& graph,
                               Edge edge,
                               Vertex v,
                               double f,
                               AddCoeffOrKnown&& addCoeffOrKnown,
                               AddRhs&& addRhs) {
    const auto& ep = graph[edge];
    if (!isActive(ep)) return;

    assembleTemperatureCoeffsAtNode(graph, edge, v, f, addCoeffOrKnown);

    const Vertex sv = boost::source(edge, graph);
    const auto tc = ep.getTypeCode();
    if (tc == EdgeProperties::TypeCode::HeatGeneration) {
        const double q = ep.current_heat_generation;
        if (q != 0.0) addRhs(f * heatGenerationRhsSignAtNode(v, sv) * q);
        return;
    }
    if (tc == EdgeProperties::TypeCode::ResponseConduction) {
        using thermal_direct_response::evalResponseHistoryWattSrc;
        using thermal_direct_response::evalResponseHistoryWattTgt;
        const double hW = responseHistIsSrcSide(v, sv)
                              ? evalResponseHistoryWattSrc(ep)
                              : evalResponseHistoryWattTgt(ep);
        addRhs(f * (+hW));
    }
}

// 後処理: heat_rate / current_q_* を更新し、熱収支ベクトルへ流入寄与を加算する。
inline void accumulatePostprocess(Graph& graph, Edge e, std::vector<double>& heatBalance) {
    auto& ep = graph[e];
    const Vertex sv = boost::source(e, graph);
    const Vertex tv = boost::target(e, graph);

    if (!isActive(ep)) {
        ep.heat_rate = 0.0;
        if (ep.getTypeCode() == EdgeProperties::TypeCode::ResponseConduction) {
            ep.current_q_src = 0.0;
            ep.current_q_tgt = 0.0;
        }
        return;
    }

    const double Ts = graph[sv].current_t;
    const double Tt = graph[tv].current_t;
    const auto tc = ep.getTypeCode();

    if (tc == EdgeProperties::TypeCode::ResponseConduction) {
        using thermal_direct_response::evalResponseQSrc;
        using thermal_direct_response::evalResponseQTgt;
        const double qs = evalResponseQSrc(ep, Ts, Tt);
        const double qt = evalResponseQTgt(ep, Ts, Tt);
        heatBalance[static_cast<size_t>(sv)] -= qs;
        heatBalance[static_cast<size_t>(tv)] -= qt;
        ep.heat_rate = (qs + qt) / 2.0;
        ep.current_q_src = qs;
        ep.current_q_tgt = qt;
        return;
    }

    if (tc == EdgeProperties::TypeCode::Advection) {
        double Q = HeatCalculation::calcAdvectionHeat(Ts, Tt, ep);
        if (ep.flow_rate > 0) {
            if (ep.is_aircon_inflow && graph[tv].on) Q = 0.0;
            heatBalance[static_cast<size_t>(tv)] += Q;
        } else {
            if (graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on) Q = 0.0;
            heatBalance[static_cast<size_t>(sv)] += Q;
        }
        ep.heat_rate = Q;
        return;
    }

    const double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep);
    ep.heat_rate = Q;
    heatBalance[static_cast<size_t>(sv)] -= Q;
    heatBalance[static_cast<size_t>(tv)] += Q;
}

} // namespace thermal_edge_physics
