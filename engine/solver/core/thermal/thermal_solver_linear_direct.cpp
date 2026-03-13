#include "core/thermal/thermal_solver_linear_direct.h"
#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect {

using detail::g_cholCache;
using detail::g_directTStats;
using detail::g_sparseLuCache;
using detail::g_topologyCache;
using detail::s_lastCoeffSig;

DirectTCacheStats getDirectTCacheStats() {
    DirectTCacheStats s;
    s.calls = g_directTStats.calls;
    s.coeffSigChanged = g_directTStats.coeffSigChanged;
    s.rhsOnlyBuild = g_directTStats.rhsOnlyBuild;
    s.fullBuild = g_directTStats.fullBuild;
    s.solveCached = g_directTStats.solveCached;
    s.solveFull = g_directTStats.solveFull;
    return s;
}

void resetDirectTCacheStats() {
    g_directTStats = detail::DirectTStats{};
    s_lastCoeffSig = 0;
    // テストの再現性のため、キャッシュも無効化
    g_topologyCache = detail::TopologyCache{};
    g_sparseLuCache.analyzed = false;
    g_sparseLuCache.factorized = false;
    g_sparseLuCache.n = 0;
    g_sparseLuCache.nnz = 0;
    g_sparseLuCache.patternHash = 0;
    g_sparseLuCache.valueHash = 0;
    g_sparseLuCache.coeffSig = 0;
    g_sparseLuCache.valuePtrIndexByRow.clear();
    g_sparseLuCache.A.resize(0, 0);
    g_sparseLuCache.solver.~SparseLU();
    new (&g_sparseLuCache.solver) Eigen::SparseLU<Eigen::SparseMatrix<double>>();

    // SparseCholeskyCache は noncopyable（Eigen solver を含む）なので、代入で初期化しない
    g_cholCache.analyzed = false;
    g_cholCache.factorized = false;
    g_cholCache.n = 0;
    g_cholCache.nnz = 0;
    g_cholCache.patternHash = 0;
    g_cholCache.valueHash = 0;
    g_cholCache.patternSymmetric = false;
    g_cholCache.llt.~SimplicialLLT();
    new (&g_cholCache.llt) Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>();
    g_cholCache.ldlt.~SimplicialLDLT();
    new (&g_cholCache.ldlt) Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>();
}

void solveTemperaturesLinearDirect(ThermalNetwork& network, const SimulationConstants& constants, std::ostream& logFile) {
    ++g_directTStats.calls;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto& graph = network.getGraph();
    const size_t curV = boost::num_vertices(graph);
    const size_t curE = boost::num_edges(graph);

    const bool needRebuildTopo =
        (!g_topologyCache.initialized) ||
        (g_topologyCache.graphPtr != &graph) ||
        (g_topologyCache.numVertices != curV) ||
        (g_topologyCache.numEdges != curE);

    if (needRebuildTopo) {
        ++g_directTStats.topoRebuild;
        detail::rebuildTopologyCache(network, graph, curV, curE, g_topologyCache);
    }

    const size_t n = g_topologyCache.nodeNames.size();
    if (n == 0) return;

    static detail::LinearSystem system;
    static size_t systemN = 0;
    static const Graph* systemGraphPtr = nullptr;
    if (needRebuildTopo || systemGraphPtr != &graph || systemN != n || system.colIndices.size() != n) {
        system.initWithPattern(g_topologyCache.rowColsPattern);
        systemN = n;
        systemGraphPtr = &graph;
    }

    const std::uint64_t coeffSig = detail::computeCoeffSignature(graph, g_topologyCache);
    if (g_directTStats.calls > 1 && s_lastCoeffSig != 0 && coeffSig != s_lastCoeffSig) {
        ++g_directTStats.coeffSigChanged;
    }
    s_lastCoeffSig = coeffSig;
    if (g_topologyCache.rhsCoeffSig != coeffSig ||
        g_topologyCache.fixedRowAirconVertex.size() != n ||
        g_topologyCache.knownTermsByRow.size() != n ||
        g_topologyCache.responseHistByRow.size() != n) {
        ++g_directTStats.rhsPrecomputeRebuild;
        detail::rebuildRhsPrecomputeForCoeffSig(graph, g_topologyCache, coeffSig);
    }

    bool canReuseFactorization = true;
    if (!g_sparseLuCache.analyzed) { canReuseFactorization = false; ++g_directTStats.reuseMissNotAnalyzed; }
    if (!(g_sparseLuCache.factorized || (g_cholCache.analyzed && g_cholCache.factorized))) {
        canReuseFactorization = false;
        ++g_directTStats.reuseMissNoFactorized;
    }
    if (g_sparseLuCache.n != static_cast<int>(n)) { canReuseFactorization = false; ++g_directTStats.reuseMissSizeMismatch; }
    if (g_sparseLuCache.coeffSig != coeffSig) { canReuseFactorization = false; ++g_directTStats.reuseMissCoeffSigMismatch; }

    if (canReuseFactorization) {
        ++g_directTStats.rhsOnlyBuild;
        detail::buildRhsOnlyAbsoluteFast(graph, g_topologyCache, system.b);
    } else {
        ++g_directTStats.fullBuild;
        detail::buildLinearSystemAbsoluteFast(graph, g_topologyCache, system);
    }

    std::vector<double> temperatures(n, 0.0);
    bool solved = false;
    std::string method = "LLT";
    if (canReuseFactorization) {
        ++g_directTStats.solveCached;
        Eigen::VectorXd eb(static_cast<int>(n));
        for (size_t i = 0; i < n; ++i) eb[static_cast<int>(i)] = system.b[i];
        solved = detail::solveWithCachedFactorization(eb, temperatures, constants.thermalTolerance, logFile, method);
    } else {
        ++g_directTStats.solveFull;
        solved = detail::solveSparseDirect(system, temperatures, constants.thermalTolerance, logFile, method);
        if (solved) g_sparseLuCache.coeffSig = coeffSig;
    }

    if (!solved) {
        throw std::runtime_error("thermal solve failed (direct absolute T solver)");
    }

    for (size_t i = 0; i < n; ++i) graph[g_topologyCache.parameterIndexToVertex[i]].current_t = temperatures[i];

    detail::postprocessAndReport(network, graph, g_topologyCache, curV, n, constants, method, logFile, startTime, g_directTStats);
}

} // namespace ThermalSolverLinearDirect
