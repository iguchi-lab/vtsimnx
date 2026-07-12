#include "core/thermal/thermal_direct_internal.h"
#include "core/thermal/thermal_edge_physics.h"

namespace ThermalSolverLinearDirect::detail {

void buildLinearSystemAbsoluteFast(const Graph& graph, const TopologyCache& topo, LinearSystem& system) {
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
            thermal_edge_physics::assembleEdgeAtNode(
                graph,
                edge,
                v,
                f,
                [&](Vertex col, double aCoeff) { addCoeffOrKnownToB(row, rowMap, col, aCoeff); },
                [&](double delta) { system.b[row] += delta; },
                &topo.moist);
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
        system.b[i] += thermal_edge_physics::heatSourceToRhs(graph[v].heat_source);

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
