#include "core/thermal/thermal_direct_internal.h"
#include "core/thermal/thermal_edge_physics.h"

namespace ThermalSolverLinearDirect::detail {

static inline bool isUnknown(const TopologyCache& topo, Vertex v) {
    const size_t idx = static_cast<size_t>(v);
    if (idx >= topo.vertexToParameterIndex.size()) return false;
    return topo.vertexToParameterIndex[idx] >= 0;
}

CoeffSignatureBreakdown computeCoeffSignatureBreakdown(const Graph& graph, const TopologyCache& topo) {
    using thermal_linear_utils::fnv1a64_update;
    using thermal_linear_utils::hashDoubleBits;

    CoeffSignatureBreakdown s{};
    for (auto e : topo.advectionEdges) {
        const auto& ep = graph[e];
        Vertex sv = boost::source(e, graph);
        Vertex tv = boost::target(e, graph);
        s.flowSig = fnv1a64_update(
            s.flowSig,
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sv)) << 32) ^
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(tv)));
        double flowRate = ep.flow_rate;
        if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) flowRate = 0.0;
        s.flowSig = hashDoubleBits(s.flowSig, flowRate);
        s.flowSig = fnv1a64_update(s.flowSig, ep.is_aircon_inflow ? 1u : 0u);
        // 風量不変でも enable 切替で係数が消えるため、署名に含める。
        s.flowSig = fnv1a64_update(s.flowSig, ep.current_enabled ? 1u : 0u);
    }
    for (auto v : topo.coeffRelevantAirconVertices) {
        const auto& nd = graph[v];
        s.airconOnSig = fnv1a64_update(s.airconOnSig, static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
        s.airconOnSig = fnv1a64_update(s.airconOnSig, nd.on ? 1u : 0u);
    }
    for (Vertex setVertex : topo.coeffRelevantSetVertices) {
        const size_t setV = static_cast<size_t>(setVertex);
        bool anyOn = false;
        for (Vertex v_ac : topo.airconBySetVertex[setV]) {
            if (graph[v_ac].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[v_ac].on) {
                anyOn = true;
                break;
            }
        }
        if (anyOn) {
            s.setNodeActiveSig = fnv1a64_update(s.setNodeActiveSig, static_cast<std::uint64_t>(setV));
            s.setNodeActiveSig = fnv1a64_update(s.setNodeActiveSig, 1u);
        }
    }
    // enable 切替は係数パターンを変える（特に conductance / heat_generation / response）
    for (size_t vi = 0; vi < topo.incidentEdges.size(); ++vi) {
        for (auto e : topo.incidentEdges[vi]) {
            if (boost::source(e, graph) != static_cast<Vertex>(vi)) continue;
            const auto& ep = graph[e];
            if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) continue;
            s.enableSig = fnv1a64_update(
                s.enableSig,
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vi)) << 32) ^
                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(boost::target(e, graph))));
            s.enableSig = fnv1a64_update(s.enableSig, ep.current_enabled ? 1u : 0u);
            s.enableSig = fnv1a64_update(s.enableSig, static_cast<std::uint64_t>(ep.getTypeCode()));
        }
    }

    // 湿りエンタルピー: 移流端点・空気 capacity の current_x / x_n を署名に含める
    if (topo.moist.enabled) {
        s.humidityXSig = fnv1a64_update(s.humidityXSig, 1u);
        for (auto e : topo.advectionEdges) {
            const Vertex sv = boost::source(e, graph);
            const Vertex tv = boost::target(e, graph);
            s.humidityXSig = hashDoubleBits(s.humidityXSig, graph[sv].current_x);
            s.humidityXSig = hashDoubleBits(s.humidityXSig, graph[tv].current_x);
        }
        for (size_t vi = 0; vi < topo.incidentEdges.size(); ++vi) {
            for (auto e : topo.incidentEdges[vi]) {
                if (boost::source(e, graph) != static_cast<Vertex>(vi)) continue;
                if (!thermal_edge_physics::isAirCapacityEdge(graph, e)) continue;
                const Vertex airV = thermal_edge_physics::airVertexOfCapacityEdge(graph, e);
                s.humidityXSig = hashDoubleBits(s.humidityXSig, graph[airV].current_x);
                s.humidityXSig = hashDoubleBits(
                    s.humidityXSig, thermal_edge_physics::humidityXnAt(topo.moist, graph, airV));
            }
        }
    }
    return s;
}

std::uint64_t computeCoeffSignature(const Graph& graph, const TopologyCache& topo) {
    return computeCoeffSignatureBreakdown(graph, topo).combined();
}

void rebuildRhsPrecomputeForCoeffSig(const Graph& graph, TopologyCache& topo, std::uint64_t coeffSig) {
    const size_t n = topo.parameterIndexToVertex.size();
    topo.knownTermsByRow.assign(n, {});
    topo.heatGenByRow.assign(n, {});
    topo.responseHistByRow.assign(n, {});
    topo.moistConstRhsByRow.assign(n, {});
    topo.fixedRowAirconVertex.assign(n, std::numeric_limits<Vertex>::max());

    auto addKnown = [&](size_t row, Vertex v, double coeff) {
        if (std::abs(coeff) < 1e-15) return;
        if (isUnknown(topo, v)) return;
        topo.knownTermsByRow[row].push_back(TopologyCache::KnownTerm{v, coeff});
    };
    auto addHeatGen = [&](size_t row, Edge e, double sign) {
        if (std::abs(sign) < 1e-15) return;
        topo.heatGenByRow[row].push_back(TopologyCache::HeatGenTerm{e, sign});
    };
    auto addRespHist = [&](size_t row, Edge e, bool isSrc, double factor) {
        topo.responseHistByRow[row].push_back(TopologyCache::ResponseHistTerm{e, isSrc, factor});
    };
    auto addMoistConst = [&](size_t row, double value) {
        if (std::abs(value) < 1e-15) return;
        topo.moistConstRhsByRow[row].push_back(TopologyCache::ConstRhsTerm{value});
    };

    for (size_t i = 0; i < n; ++i) {
        Vertex rowV = topo.parameterIndexToVertex[i];

        // 固定温度行（set_node の aircon が ON）
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(rowV)]) {
            if (graph[v_ac].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[v_ac].on) {
                topo.fixedRowAirconVertex[i] = v_ac;
                break;
            }
        }
        if (topo.fixedRowAirconVertex[i] != std::numeric_limits<Vertex>::max()) {
            continue;
        }

        // 行のネットワーク参照頂点（aircon on の場合は set_node 側を見る）
        Vertex procV = rowV;
        if (graph[rowV].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[rowV].on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(rowV)];
            if (setV != std::numeric_limits<Vertex>::max()) procV = setV;
        }

        for (auto edge : topo.incidentEdges[static_cast<size_t>(procV)]) {
            const auto& ep = graph[edge];
            if (!thermal_edge_physics::isActive(ep)) continue;

            const Vertex sv = boost::source(edge, graph);
            thermal_edge_physics::assembleTemperatureCoeffsAtNode(
                graph,
                edge,
                procV,
                1.0,
                [&](Vertex col, double aCoeff) { addKnown(i, col, aCoeff); },
                &topo.moist);

            if (topo.moist.enabled) {
                thermal_edge_physics::assembleMoistConstRhsAtNode(
                    graph, edge, procV, 1.0, [&](double delta) { addMoistConst(i, delta); }, topo.moist);
            }

            const auto tc = ep.getTypeCode();
            if (tc == EdgeProperties::TypeCode::HeatGeneration) {
                addHeatGen(i, edge, thermal_edge_physics::heatGenerationRhsSignAtNode(procV, sv));
            } else if (tc == EdgeProperties::TypeCode::ResponseConduction) {
                addRespHist(i, edge, thermal_edge_physics::responseHistIsSrcSide(procV, sv), 1.0);
            }
        }

        // known term をまとめる（行長が小さい前提）
        auto& terms = topo.knownTermsByRow[i];
        std::sort(terms.begin(), terms.end(),
                  [](const auto& a, const auto& b) { return a.v < b.v; });
        size_t w = 0;
        for (size_t r = 0; r < terms.size(); ++r) {
            if (w == 0 || terms[r].v != terms[w - 1].v) {
                terms[w++] = terms[r];
            } else {
                terms[w - 1].coeff += terms[r].coeff;
            }
        }
        terms.resize(w);
    }

    topo.rhsCoeffSig = coeffSig;
}

void buildRhsOnlyAbsoluteFast(const Graph& graph, const TopologyCache& topo, std::vector<double>& bOut) {
    using thermal_direct_response::evalResponseHistoryWattSrc;
    using thermal_direct_response::evalResponseHistoryWattTgt;

    const size_t n = topo.parameterIndexToVertex.size();
    bOut.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const Vertex v_ac = (i < topo.fixedRowAirconVertex.size())
                                ? topo.fixedRowAirconVertex[i]
                                : std::numeric_limits<Vertex>::max();
        if (v_ac != std::numeric_limits<Vertex>::max()) {
            bOut[i] = graph[v_ac].current_pre_temp;
            continue;
        }

        const Vertex rowV = topo.parameterIndexToVertex[i];
        bOut[i] += thermal_edge_physics::heatSourceToRhs(graph[rowV].heat_source);

        if (i < topo.knownTermsByRow.size()) {
            for (const auto& t : topo.knownTermsByRow[i]) {
                if (t.v == std::numeric_limits<Vertex>::max()) continue;
                bOut[i] -= t.coeff * graph[t.v].current_t;
            }
        }
        if (i < topo.moistConstRhsByRow.size()) {
            for (const auto& t : topo.moistConstRhsByRow[i]) {
                bOut[i] += t.value;
            }
        }
        if (i < topo.heatGenByRow.size()) {
            for (const auto& t : topo.heatGenByRow[i]) {
                bOut[i] += t.sign * graph[t.e].current_heat_generation;
            }
        }
        if (i < topo.responseHistByRow.size()) {
            for (const auto& t : topo.responseHistByRow[i]) {
                const double hW = t.isSrc ? evalResponseHistoryWattSrc(graph[t.e])
                                          : evalResponseHistoryWattTgt(graph[t.e]);
                bOut[i] += t.factor * hW;
            }
        }
    }
}

} // namespace ThermalSolverLinearDirect::detail
