#include "core/thermal/thermal_direct_internal.h"
#include "core/thermal/thermal_edge_physics.h"
#include "core/thermal/thermal_moist_air.h"

namespace ThermalSolverLinearDirect::detail {
namespace {

// 既に accumulatePostprocess 済みの heat_rate から、edge が vertex へ与える流入 [W] を返す
double edgeHeatIntoVertex(const Graph& graph, Edge e, Vertex v) {
    const auto& ep = graph[e];
    const Vertex sv = boost::source(e, graph);
    const Vertex tv = boost::target(e, graph);
    if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) {
        if (ep.flow_rate > 0.0) {
            return (tv == v) ? ep.heat_rate : 0.0;
        }
        if (ep.flow_rate < 0.0) {
            return (sv == v) ? ep.heat_rate : 0.0;
        }
        return 0.0;
    }
    // conductance / capacity / response など: sv -= Q, tv += Q（Q=heat_rate）
    if (tv == v) return ep.heat_rate;
    if (sv == v) return -ep.heat_rate;
    return 0.0;
}

bool edgeTouchesVertex(const Graph& graph, Edge e, Vertex v) {
    return boost::source(e, graph) == v || boost::target(e, graph) == v;
}

// set と AC が直接移流結合していないとき用: 還気→吹出の符号付きコイル処理熱量 [W]
double coilSignedProcessedHeatW(const Graph& graph,
                                const ThermalNetwork& network,
                                Vertex acV,
                                bool moistEnthalpy) {
    const auto& ac = graph[acV];
    if (ac.in_node.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto& keyToVertex = network.getKeyToVertex();
    auto it = keyToVertex.find(ac.in_node);
    if (it == keyToVertex.end()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Vertex inV = it->second;

    // Graph は directedS のため in_edges は使えない。AC↔in の双方の out_edges を見る。
    double flowAbs = 0.0;
    for (auto e : boost::make_iterator_range(boost::out_edges(acV, graph))) {
        const auto& ep = graph[e];
        if (ep.getTypeCode() != EdgeProperties::TypeCode::Advection) continue;
        if (boost::target(e, graph) == inV) {
            flowAbs = std::max(flowAbs, std::abs(ep.flow_rate));
        }
    }
    for (auto e : boost::make_iterator_range(boost::out_edges(inV, graph))) {
        const auto& ep = graph[e];
        if (ep.getTypeCode() != EdgeProperties::TypeCode::Advection) continue;
        if (boost::target(e, graph) == acV) {
            flowAbs = std::max(flowAbs, std::abs(ep.flow_rate));
        }
    }

    return thermal_moist_air::signedProcessedHeatW(
        graph[inV].current_t,
        graph[inV].current_x,
        graph[acV].current_t,
        graph[acV].current_x,
        flowAbs,
        moistEnthalpy);
}

} // namespace

void postprocessAndReport(ThermalNetwork& network,
                          Graph& graph,
                          const TopologyCache& topo,
                          size_t curV,
                          size_t n,
                          const SimulationConstants& constants,
                          const std::string& method,
                          std::ostream& logFile,
                          std::chrono::high_resolution_clock::time_point startTime,
                          DirectTStats& stats) {
    std::vector<double> heatBalance(curV, 0.0);
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        thermal_edge_physics::accumulatePostprocess(graph, e, heatBalance, &topo.moist);
    }

    // 1パス目: heat_source 加算と required_heat_w の初期化
    for (size_t i = 0; i < curV; ++i) {
        heatBalance[i] += graph[i].heat_source;
        if (graph[i].getTypeCode() == VertexProperties::TypeCode::Aircon) {
            graph[i].required_heat_w = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // 2パス目: 設定温度維持に必要な符号付き負荷
    // - 還気/吹出が set_node に直結: set 熱収支から AC 寄与を除いた符号反転（正本）
    // - set と AC が非直結（例: set=LDK, in/out=階間）: dual-row 後の set 収支は ≈0 になり
    //   常に Qreq≈0 となるため、コイル処理熱量で代替する
    for (Vertex acV : topo.airconVertices) {
        const size_t i = static_cast<size_t>(acV);
        if (!graph[i].on) continue;

        Vertex setV = topo.airconSetVertex[i];
        if (setV == std::numeric_limits<Vertex>::max()) continue;

        const size_t setIdx = static_cast<size_t>(setV);
        double qAcIntoSet = 0.0;
        bool hasDirectAcSetEdge = false;
        for (auto e : topo.incidentEdges[setIdx]) {
            if (!edgeTouchesVertex(graph, e, acV)) continue;
            hasDirectAcSetEdge = true;
            qAcIntoSet += edgeHeatIntoVertex(graph, e, setV);
        }

        if (hasDirectAcSetEdge) {
            const double qOther = heatBalance[setIdx] - qAcIntoSet;
            // qOther>0 = AC以外が室を加熱 → 設定維持には除熱が必要 → Qrequired<0
            graph[i].required_heat_w = -qOther;
        } else {
            graph[i].required_heat_w = coilSignedProcessedHeatW(
                graph, network, acV, topo.moist.enabled);
        }

        // RMSE 用: 固定温度行の残差は空調側へ移す（必要負荷とは別）
        heatBalance[i] = heatBalance[setIdx];
        heatBalance[setIdx] = 0.0;
    }

    double maxB = 0.0, rmseB = 0.0;
    for (auto v : topo.parameterIndexToVertex) {
        double b = heatBalance[static_cast<size_t>(v)];
        maxB = std::max(maxB, std::abs(b));
        rmseB += b * b;
    }
    rmseB = std::sqrt(rmseB / n);
    auto durUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - startTime);
    const double durMs = static_cast<double>(durUs.count()) / 1000.0;

    const double balanceTolW = effectiveThermalBalanceToleranceW(constants);
    std::ostringstream oss;
    oss << "--------熱計算(線形): "
        << (rmseB <= balanceTolW ? "収束" : "未収束")
        << " (method=" << method
        << ", RMSE=" << std::scientific << std::setprecision(6) << rmseB
        << ", maxBalance=" << maxB
        << ", time=" << std::fixed << std::setprecision(3) << durMs << "ms)";
    writeLog(logFile, oss.str());
    network.setLastThermalConvergence(rmseB <= balanceTolW, rmseB, maxB, method);

    constexpr std::uint64_t kStatsLogInterval = 500;
    // VTSIMNX_TIMINGS 有効時は毎呼び出しで cache stats を出し、性能ベンチで
    // luFactorize / topoRebuild を短ケースでも拾えるようにする。
    const bool timingsEnv = (std::getenv("VTSIMNX_TIMINGS") != nullptr);
    const std::uint64_t statsInterval = timingsEnv ? 1 : kStatsLogInterval;
    if (stats.calls > 0 && (stats.calls % statsInterval) == 0) {
        std::ostringstream ss;
        ss << "--------DirectT cache stats: calls=" << stats.calls
           << ", n=" << n
           << ", coeffSigChanged=" << stats.coeffSigChanged
           << ", coeffSigFlowChanged=" << stats.coeffSigFlowChanged
           << ", coeffSigAirconOnChanged=" << stats.coeffSigAirconOnChanged
           << ", coeffSigSetNodeChanged=" << stats.coeffSigSetNodeChanged
           << ", missNotAnalyzed=" << stats.reuseMissNotAnalyzed
           << ", missNoFactorized=" << stats.reuseMissNoFactorized
           << ", missSizeMismatch=" << stats.reuseMissSizeMismatch
           << ", missCoeffSigMismatch=" << stats.reuseMissCoeffSigMismatch
           << ", topoRebuild=" << stats.topoRebuild
           << ", rhsPrecomputeRebuild=" << stats.rhsPrecomputeRebuild
           << ", rhsOnlyBuild=" << stats.rhsOnlyBuild
           << ", fullBuild=" << stats.fullBuild
           << ", patternRebuild=" << stats.patternRebuild
           << ", solveCached=" << stats.solveCached
           << ", solveFull=" << stats.solveFull
           << ", rhsSolutionReuse=" << stats.rhsSolutionReuse
           << ", postprocessReuse=" << stats.postprocessReuse
           << ", cholFactorize=" << stats.cholFactorize
           << ", luFactorize=" << stats.luFactorize;
        writeLog(logFile, ss.str());
    }
}

} // namespace ThermalSolverLinearDirect::detail
