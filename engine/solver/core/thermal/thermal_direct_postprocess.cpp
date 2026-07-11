#include "core/thermal/thermal_direct_internal.h"
#include "core/thermal/thermal_edge_physics.h"

namespace ThermalSolverLinearDirect::detail {

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
        thermal_edge_physics::accumulatePostprocess(graph, e, heatBalance);
    }
    for (size_t i = 0; i < curV; ++i) {
        heatBalance[i] += graph[i].heat_source;
        if (graph[i].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[i].on) {
            Vertex setV = topo.airconSetVertex[i];
            if (setV != std::numeric_limits<Vertex>::max()) {
                heatBalance[i] = heatBalance[static_cast<size_t>(setV)];
                heatBalance[static_cast<size_t>(setV)] = 0.0;
            }
        }
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
