#include "core/thermal/thermal_direct_internal.h"

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
    return s;
}

std::uint64_t computeCoeffSignature(const Graph& graph, const TopologyCache& topo) {
    return computeCoeffSignatureBreakdown(graph, topo).combined();
}

void rebuildRhsPrecomputeForCoeffSig(const Graph& graph, TopologyCache& topo, std::uint64_t coeffSig) {
    using thermal_direct_response::responseArea;
    using thermal_direct_response::evalResponseHistoryWattSrc;
    using thermal_direct_response::evalResponseHistoryWattTgt;

    const size_t n = topo.parameterIndexToVertex.size();
    topo.knownTermsByRow.assign(n, {});
    topo.heatGenByRow.assign(n, {});
    topo.responseHistByRow.assign(n, {});
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
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& ep = graph[edge];
            const auto tc = ep.getTypeCode();

            if (tc == EdgeProperties::TypeCode::Conductance) {
                const double k = ep.conductance;
                if (sv == procV) {
                    addKnown(i, sv, -k);
                    addKnown(i, tv, +k);
                } else {
                    addKnown(i, sv, +k);
                    addKnown(i, tv, -k);
                }
            } else if (tc == EdgeProperties::TypeCode::Advection) {
                double flowRate = ep.flow_rate;
                if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) continue;
                const double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
                if (flowRate > 0) {
                    if (tv == procV && !(ep.is_aircon_inflow && graph[tv].on)) {
                        addKnown(i, sv, +mDotCp);
                        addKnown(i, tv, -mDotCp);
                    }
                } else {
                    if (sv == procV && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on)) {
                        addKnown(i, tv, -mDotCp);
                        addKnown(i, sv, +mDotCp);
                    }
                }
            } else if (tc == EdgeProperties::TypeCode::HeatGeneration) {
                if (sv == procV) addHeatGen(i, edge, +1.0);
                else addHeatGen(i, edge, -1.0);
            } else if (tc == EdgeProperties::TypeCode::ResponseConduction) {
                const double area = responseArea(ep);
                if (sv == procV) {
                    const double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0];
                    const double b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0];
                    addKnown(i, sv, -area * a0);
                    addKnown(i, tv, -area * b0);
                    addRespHist(i, edge, true, 1.0);
                } else {
                    const double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0];
                    const double b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0];
                    addKnown(i, tv, -area * a0);
                    addKnown(i, sv, -area * b0);
                    addRespHist(i, edge, false, 1.0);
                }
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
        bOut[i] += graph[rowV].heat_source;

        if (i < topo.knownTermsByRow.size()) {
            for (const auto& t : topo.knownTermsByRow[i]) {
                if (t.v == std::numeric_limits<Vertex>::max()) continue;
                if (std::abs(t.coeff) < 1e-15) continue;
                bOut[i] -= t.coeff * graph[t.v].current_t;
            }
        }
        if (i < topo.heatGenByRow.size()) {
            for (const auto& tg : topo.heatGenByRow[i]) {
                bOut[i] += tg.sign * graph[tg.e].current_heat_generation;
            }
        }
        if (i < topo.responseHistByRow.size()) {
            for (const auto& rh : topo.responseHistByRow[i]) {
                const auto& ep = graph[rh.e];
                const double hW = rh.isSrc ? evalResponseHistoryWattSrc(ep) : evalResponseHistoryWattTgt(ep);
                bOut[i] += rh.factor * (+hW);
            }
        }
    }
}

} // namespace ThermalSolverLinearDirect::detail


