#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect::detail {

void rebuildTopologyCache(ThermalNetwork& network,
                          const Graph& graph,
                          size_t curV,
                          size_t curE,
                          TopologyCache& topo) {
    topo = TopologyCache{};
    topo.graphPtr = &graph;
    topo.numVertices = curV;
    topo.numEdges = curE;

    topo.incidentEdges.assign(curV, {});
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        topo.incidentEdges[static_cast<size_t>(boost::source(e, graph))].push_back(e);
        topo.incidentEdges[static_cast<size_t>(boost::target(e, graph))].push_back(e);
    }

    topo.airconBySetVertex.assign(curV, {});
    topo.airconSetVertex.assign(curV, std::numeric_limits<Vertex>::max());
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon && !graph[v].set_node.empty()) {
            auto it = network.getKeyToVertex().find(graph[v].set_node);
            if (it != network.getKeyToVertex().end()) {
                topo.airconBySetVertex[static_cast<size_t>(it->second)].push_back(v);
                topo.airconSetVertex[static_cast<size_t>(v)] = it->second;
            }
        }
    }

    topo.advectionEdges.clear();
    topo.responseEdges.clear();
    topo.airconVertices.clear();
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        auto tc = graph[e].getTypeCode();
        if (tc == EdgeProperties::TypeCode::Advection) topo.advectionEdges.push_back(e);
        else if (tc == EdgeProperties::TypeCode::ResponseConduction) topo.responseEdges.push_back(e);
    }
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon) topo.airconVertices.push_back(v);
    }

    topo.nodeNames.clear();
    topo.vertexToParameterIndex.assign(curV, -1);
    topo.parameterIndexToVertex.clear();
    size_t pIdx = 0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        if (graph[v].calc_t) {
            topo.nodeNames.push_back(graph[v].key);
            topo.vertexToParameterIndex[static_cast<size_t>(v)] = static_cast<int>(pIdx++);
            topo.parameterIndexToVertex.push_back(v);
        }
    }

    topo.rowColsPattern.assign(topo.nodeNames.size(), {});
    for (size_t r = 0; r < topo.nodeNames.size(); ++r) {
        Vertex v = topo.parameterIndexToVertex[r];
        std::vector<int> cols = {static_cast<int>(r)};
        auto addV = [&](Vertex vv) {
            int idx = topo.vertexToParameterIndex[static_cast<size_t>(vv)];
            if (idx >= 0) cols.push_back(idx);
        };
        for (auto e : topo.incidentEdges[static_cast<size_t>(v)]) {
            addV(boost::source(e, graph));
            addV(boost::target(e, graph));
        }
        Vertex setV = topo.airconSetVertex[static_cast<size_t>(v)];
        if (setV != std::numeric_limits<Vertex>::max()) {
            for (auto e : topo.incidentEdges[static_cast<size_t>(setV)]) {
                addV(boost::source(e, graph));
                addV(boost::target(e, graph));
            }
        }
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        topo.rowColsPattern[r] = std::move(cols);
    }
    topo.rowIndexMaps.assign(topo.rowColsPattern.size(), {});
    for (size_t r = 0; r < topo.rowColsPattern.size(); ++r)
        topo.rowIndexMaps[r].buildFromCols(topo.rowColsPattern[r]);

    topo.rhsCoeffSig = 0;
    topo.initialized = true;
}

} // namespace ThermalSolverLinearDirect::detail


