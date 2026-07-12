#pragma once

#include "core/thermal/heat_calculation.h"
#include "core/thermal/thermal_direct_response.h"
#include "core/thermal/thermal_moist_air.h"
#include "../../archenv/include/archenv.h"

#include <limits>
#include <vector>

// 熱枝の有効判定・符号・行列寄与・後処理熱流量を一箇所に集約する。
// build / RHS precompute / postprocess が同じ定義を参照する。

namespace thermal_edge_physics {

using MoistAssembleContext = thermal_moist_air::MoistAssembleContext;

inline double humidityXnAt(const MoistAssembleContext& moist, const Graph& graph, Vertex v) {
    if (moist.humidityXnByVertex != nullptr) {
        const size_t i = static_cast<size_t>(v);
        if (i < moist.humidityXnByVertex->size()) return (*moist.humidityXnByVertex)[i];
    }
    return graph[v].current_x;
}

inline bool isActive(const EdgeProperties& ep) {
    return ep.current_enabled;
}

// 熱収支定義: Σ流入 + heat_source = 0 ⇔ A*T = -heat_source + (枝の温度非依存項)
inline double heatSourceToRhs(double heatSource) {
    return -heatSource;
}

inline bool isDedicatedAirCapacityEdge(const EdgeProperties& ep) {
    return ep.getTypeCode() == EdgeProperties::TypeCode::Conductance && ep.subtype == "air_capacity";
}

inline bool isLegacyMoistCapacityEdge(const Graph& graph, Edge edge) {
    const auto& ep = graph[edge];
    if (ep.getTypeCode() != EdgeProperties::TypeCode::Conductance) return false;
    if (ep.subtype != "capacity") return false;
    const Vertex tv = boost::target(edge, graph);
    if (graph[tv].v > 0.0) return true;
    const Vertex sv = boost::source(edge, graph);
    return graph[sv].v > 0.0;
}

inline bool isAirCapacityEdge(const Graph& graph, Edge edge) {
    return isDedicatedAirCapacityEdge(graph[edge]) || isLegacyMoistCapacityEdge(graph, edge);
}

inline Vertex airVertexOfCapacityEdge(const Graph& graph, Edge edge) {
    const Vertex tv = boost::target(edge, graph);
    if (graph[tv].v > 0.0) return tv;
    const Vertex sv = boost::source(edge, graph);
    if (graph[sv].v > 0.0) return sv;
    // air_capacity で v 未設定のときは target を空気とみなす
    return tv;
}

inline Vertex capacityRefVertexOfEdge(const Graph& graph, Edge edge) {
    const Vertex airV = airVertexOfCapacityEdge(graph, edge);
    const Vertex sv = boost::source(edge, graph);
    return (airV == sv) ? boost::target(edge, graph) : sv;
}

inline double airRhoVOverDt(const Graph& graph, Edge edge, const MoistAssembleContext& moist) {
    const Vertex airV = airVertexOfCapacityEdge(graph, edge);
    return thermal_moist_air::rhoVOverDtFromVolume(graph[airV].v, moist.dt);
}

// 温度に比例する係数だけを組み立てる（conductance / advection / response の a0,b0）。
// 湿りエンタルピーの温度非依存項は assembleMoistConstRhsAtNode 側。
template <typename AddCoeffOrKnown>
inline void assembleTemperatureCoeffsAtNode(const Graph& graph,
                                            Edge edge,
                                            Vertex v,
                                            double f,
                                            AddCoeffOrKnown&& addCoeffOrKnown,
                                            const MoistAssembleContext* moist = nullptr) {
    const auto& ep = graph[edge];
    if (!isActive(ep)) return;

    const Vertex sv = boost::source(edge, graph);
    const Vertex tv = boost::target(edge, graph);
    const auto tc = ep.getTypeCode();
    const bool moistOn = moist != nullptr && moist->enabled;

    if (tc == EdgeProperties::TypeCode::Conductance) {
        if (moistOn && isAirCapacityEdge(graph, edge)) {
            const Vertex airV = airVertexOfCapacityEdge(graph, edge);
            const Vertex capV = capacityRefVertexOfEdge(graph, edge);
            const double xNp1 = graph[airV].current_x;
            const double xN = humidityXnAt(*moist, graph, airV);
            const double cpv = archenv::SPECIFIC_HEAT_WATER_VAPOR;
            const double rhoV_dt = airRhoVOverDt(graph, edge, *moist);

            if (isDedicatedAirCapacityEdge(ep)) {
                // 専用 air_capacity: 全湿り蓄積 ρV/Δt * Δh（conductance は使わない）
                const double cpNp1 = thermal_moist_air::moistAirCp(xNp1);
                const double cpN = thermal_moist_air::moistAirCp(xN);
                if (v == airV) {
                    addCoeffOrKnown(airV, f * (-rhoV_dt * cpNp1));
                    addCoeffOrKnown(capV, f * (+rhoV_dt * cpN));
                }
                return;
            }

            // レガシー capacity + v>0: 乾き conductance を維持し、水蒸気分だけ加算
            const double k = ep.conductance;
            if (v == airV) {
                addCoeffOrKnown(airV, f * (-k));
                addCoeffOrKnown(capV, f * (+k));
                addCoeffOrKnown(airV, f * (-rhoV_dt * xNp1 * cpv));
                addCoeffOrKnown(capV, f * (+rhoV_dt * xN * cpv));
            }
            return;
        }
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

        if (moistOn) {
            const double mDot = thermal_moist_air::massFlowKgPerS(flowRate);
            const double cpS = thermal_moist_air::moistAirCp(graph[sv].current_x);
            const double cpT = thermal_moist_air::moistAirCp(graph[tv].current_x);
            if (flowRate > 0) {
                if (tv == v && !(ep.is_aircon_inflow && graph[tv].on)) {
                    addCoeffOrKnown(sv, f * (+mDot * cpS));
                    addCoeffOrKnown(tv, f * (-mDot * cpT));
                }
            } else {
                if (sv == v && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon &&
                                 graph[sv].on)) {
                    addCoeffOrKnown(tv, f * (-mDot * cpT));
                    addCoeffOrKnown(sv, f * (+mDot * cpS));
                }
            }
            return;
        }

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

// 湿りエンタルピーの温度非依存 RHS（潜熱差分など）。A*T = b の b へ加算。
template <typename AddRhs>
inline void assembleMoistConstRhsAtNode(const Graph& graph,
                                        Edge edge,
                                        Vertex v,
                                        double f,
                                        AddRhs&& addRhs,
                                        const MoistAssembleContext& moist) {
    if (!moist.enabled || !isActive(graph[edge])) return;

    const auto& ep = graph[edge];
    const Vertex sv = boost::source(edge, graph);
    const Vertex tv = boost::target(edge, graph);
    const auto tc = ep.getTypeCode();
    const double Lv = archenv::LATENT_HEAT_VAPORIZATION;

    if (tc == EdgeProperties::TypeCode::Advection) {
        double flowRate = ep.flow_rate;
        if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) return;
        const double mDot = thermal_moist_air::massFlowKgPerS(flowRate);
        const double xs = graph[sv].current_x;
        const double xt = graph[tv].current_x;
        // 流入 = mDot*(h_s-h_t) → A*T = -hs - mDot*Lv*(xs-xt) のため b へ -潜熱項
        const double latentW = mDot * Lv * (xs - xt);
        if (flowRate > 0) {
            if (tv == v && !(ep.is_aircon_inflow && graph[tv].on)) {
                addRhs(f * (-latentW));
            }
        } else {
            if (sv == v && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon &&
                             graph[sv].on)) {
                addRhs(f * (-latentW));
            }
        }
        return;
    }

    if (tc == EdgeProperties::TypeCode::Conductance && isAirCapacityEdge(graph, edge)) {
        const Vertex airV = airVertexOfCapacityEdge(graph, edge);
        if (v != airV) return;
        const double rhoV_dt = airRhoVOverDt(graph, edge, moist);
        const double xNp1 = graph[airV].current_x;
        const double xN = humidityXnAt(moist, graph, airV);
        // 水蒸気潜熱差分（専用・レガシー共通）: b += (ρV/Δt)*Lv*(x_np1 - x_n)
        addRhs(f * (rhoV_dt * Lv * (xNp1 - xN)));
    }
}

// heat_generation の RHS 符号（A*T = b 側）。sv 側 +q、tv 側 -q。
inline double heatGenerationRhsSignAtNode(Vertex v, Vertex sv) {
    return (sv == v) ? +1.0 : -1.0;
}

inline bool responseHistIsSrcSide(Vertex v, Vertex sv) {
    return sv == v;
}

// フル行列構築向け: 温度係数 + 可変 RHS（発熱・応答履歴・湿り定数項）を同時に適用。
template <typename AddCoeffOrKnown, typename AddRhs>
inline void assembleEdgeAtNode(const Graph& graph,
                               Edge edge,
                               Vertex v,
                               double f,
                               AddCoeffOrKnown&& addCoeffOrKnown,
                               AddRhs&& addRhs,
                               const MoistAssembleContext* moist = nullptr) {
    const auto& ep = graph[edge];
    if (!isActive(ep)) return;

    assembleTemperatureCoeffsAtNode(graph, edge, v, f, addCoeffOrKnown, moist);
    if (moist != nullptr && moist->enabled) {
        assembleMoistConstRhsAtNode(graph, edge, v, f, addRhs, *moist);
    }

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
inline void accumulatePostprocess(Graph& graph,
                                  Edge e,
                                  std::vector<double>& heatBalance,
                                  const MoistAssembleContext* moist = nullptr) {
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
    const bool moistOn = moist != nullptr && moist->enabled;

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
        double Q = moistOn ? HeatCalculation::calcAdvectionHeatMoist(
                                 Ts, graph[sv].current_x, Tt, graph[tv].current_x, ep)
                           : HeatCalculation::calcAdvectionHeat(Ts, Tt, ep);
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

    if (moistOn && isAirCapacityEdge(graph, e)) {
        const Vertex airV = airVertexOfCapacityEdge(graph, e);
        const Vertex capV = capacityRefVertexOfEdge(graph, e);
        const double Tn = graph[capV].current_t;
        const double Tnp1 = graph[airV].current_t;
        const double xNp1 = graph[airV].current_x;
        const double xN = humidityXnAt(*moist, graph, airV);
        const double rhoV_dt = airRhoVOverDt(graph, e, *moist);
        const double cpv = archenv::SPECIFIC_HEAT_WATER_VAPOR;
        const double Lv = archenv::LATENT_HEAT_VAPORIZATION;
        double Q = 0.0;
        if (isDedicatedAirCapacityEdge(ep)) {
            const double hN = thermal_moist_air::moistAirEnthalpy(Tn, xN);
            const double hNp1 = thermal_moist_air::moistAirEnthalpy(Tnp1, xNp1);
            Q = rhoV_dt * (hN - hNp1);
        } else {
            // 乾き conductance + 水蒸気分
            Q = ep.conductance * (Tn - Tnp1) +
                rhoV_dt * (xN * cpv * Tn - xNp1 * cpv * Tnp1 + Lv * (xN - xNp1));
        }
        ep.heat_rate = Q;
        heatBalance[static_cast<size_t>(sv)] -= Q;
        heatBalance[static_cast<size_t>(tv)] += Q;
        return;
    }

    const double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep);
    ep.heat_rate = Q;
    heatBalance[static_cast<size_t>(sv)] -= Q;
    heatBalance[static_cast<size_t>(tv)] += Q;
}

} // namespace thermal_edge_physics
