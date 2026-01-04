#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect::detail {

void buildLinearSystemAbsoluteFast(const Graph& graph, const TopologyCache& topo, LinearSystem& system) {
    using thermal_direct_response::responseArea;
    using thermal_direct_response::evalResponseHistoryWattSrc;
    using thermal_direct_response::evalResponseHistoryWattTgt;

    const size_t n = topo.parameterIndexToVertex.size();
    system.resetValuesKeepPattern();

    std::vector<uint8_t> isFixedRow(n, 0);
    std::vector<double> fixedTemp(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Vertex v = topo.parameterIndexToVertex[i];
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(v)]) {
            if (graph[v_ac].on) {
                isFixedRow[i] = 1;
                fixedTemp[i] = graph[v_ac].current_pre_temp;
                break;
            }
        }
    }

    auto addCoeffOrKnownToB = [&](size_t row, const TopologyCache::RowIndexMap& rowMap, Vertex colVertex, double aCoeff) {
        int colIdx = topo.vertexToParameterIndex[static_cast<size_t>(colVertex)];
        if (colIdx >= 0) {
            int local = rowMap.get(colIdx);
            if (local >= 0) system.addCoefficientLocal(row, local, aCoeff);
        } else {
            system.b[row] -= aCoeff * graph[colVertex].current_t;
        }
    };

    auto processNodeNet = [&](size_t row, const TopologyCache::RowIndexMap& rowMap, Vertex v, double f) {
        for (auto edge : topo.incidentEdges[static_cast<size_t>(v)]) {
            Vertex sv = boost::source(edge, graph), tv = boost::target(edge, graph);
            const auto& ep = graph[edge];
            auto tc = ep.getTypeCode();

            if (tc == EdgeProperties::TypeCode::Conductance) {
                double k = ep.conductance;
                if (sv == v) {
                    addCoeffOrKnownToB(row, rowMap, sv, f * (-k));
                    addCoeffOrKnownToB(row, rowMap, tv, f * (+k));
                } else {
                    addCoeffOrKnownToB(row, rowMap, sv, f * (+k));
                    addCoeffOrKnownToB(row, rowMap, tv, f * (-k));
                }
            } else if (tc == EdgeProperties::TypeCode::Advection) {
                double flowRate = ep.flow_rate;
                if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) continue;
                double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
                if (flowRate > 0) {
                    if (tv == v && !(ep.is_aircon_inflow && graph[tv].on)) {
                        addCoeffOrKnownToB(row, rowMap, sv, f * (+mDotCp));
                        addCoeffOrKnownToB(row, rowMap, tv, f * (-mDotCp));
                    }
                } else {
                    if (sv == v && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on)) {
                        addCoeffOrKnownToB(row, rowMap, tv, f * (-mDotCp));
                        addCoeffOrKnownToB(row, rowMap, sv, f * (+mDotCp));
                    }
                }
            } else if (tc == EdgeProperties::TypeCode::HeatGeneration) {
                double q = ep.current_heat_generation;
                if (q != 0.0) system.b[row] += (sv == v ? f * (+q) : f * (-q));
            } else if (tc == EdgeProperties::TypeCode::ResponseConduction) {
                if (sv == v) {
                    double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0];
                    double b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0];
                    double area = responseArea(ep);
                    addCoeffOrKnownToB(row, rowMap, sv, f * (-area * a0));
                    addCoeffOrKnownToB(row, rowMap, tv, f * (-area * b0));
                    system.b[row] += f * (+evalResponseHistoryWattSrc(ep));
                } else {
                    double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0];
                    double b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0];
                    double area = responseArea(ep);
                    addCoeffOrKnownToB(row, rowMap, tv, f * (-area * a0));
                    addCoeffOrKnownToB(row, rowMap, sv, f * (-area * b0));
                    system.b[row] += f * (+evalResponseHistoryWattTgt(ep));
                }
            }
        }
    };

    for (size_t i = 0; i < n; ++i) {
        if (isFixedRow[i]) {
            system.b[i] = fixedTemp[i];
            int local = topo.rowIndexMaps[i].get(static_cast<int>(i));
            if (local >= 0) system.A[i][static_cast<size_t>(local)] = 1.0;
            continue;
        }

        Vertex v = topo.parameterIndexToVertex[i];
        system.b[i] += graph[v].heat_source;

        if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[v].on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(v)];
            if (setV != std::numeric_limits<Vertex>::max())
                processNodeNet(i, topo.rowIndexMaps[i], setV, 1.0);
            else
                processNodeNet(i, topo.rowIndexMaps[i], v, 1.0);
        } else {
            processNodeNet(i, topo.rowIndexMaps[i], v, 1.0);
        }
    }
}

} // namespace ThermalSolverLinearDirect::detail


