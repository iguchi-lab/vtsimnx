#include "core/thermal/thermal_solver_linear_direct.h"
#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect {

using detail::g_cholCache;
using detail::g_directTStats;
using detail::g_sparseLuCache;
using detail::g_topologyCache;
using detail::s_lastCoeffSig;

namespace {

struct CachedSolutionReuse {
    bool valid = false;
    const Graph* graphPtr = nullptr;
    size_t n = 0;
    std::uint64_t coeffSig = 0;
    std::uint64_t rhsHash = 0;
    std::vector<double> temperatures;
    std::string method;
};

CachedSolutionReuse g_cachedSolutionReuse;

struct CachedPostprocessReuse {
    bool valid = false;
    const Graph* graphPtr = nullptr;
    size_t n = 0;
    std::uint64_t coeffSig = 0;
    std::uint64_t rhsHash = 0;
    bool converged = false;
    double rmse = 0.0;
    double maxBalance = 0.0;
    std::string method;
};

CachedPostprocessReuse g_cachedPostprocessReuse;
detail::CoeffSignatureBreakdown g_lastCoeffSigBreakdown;

std::uint64_t hashRhsValues(const std::vector<double>& rhs) {
    std::uint64_t h = 0;
    for (double v : rhs) h = thermal_linear_utils::hashDoubleBits(h, v);
    return h;
}

std::string stripRhsCachedSuffixes(const std::string& method) {
    static const std::string kSuffix = "(rhs-cached)";
    std::string base = method;
    while (base.size() >= kSuffix.size() &&
           base.compare(base.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0) {
        base.resize(base.size() - kSuffix.size());
    }
    return base;
}

std::string makeRhsCachedLabel(const std::string& method) {
    return stripRhsCachedSuffixes(method) + "(rhs-cached)";
}

} // namespace

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

    g_cachedSolutionReuse = CachedSolutionReuse{};
    g_cachedPostprocessReuse = CachedPostprocessReuse{};
    g_lastCoeffSigBreakdown = detail::CoeffSignatureBreakdown{};
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

    const detail::CoeffSignatureBreakdown coeffSigBreakdown =
        detail::computeCoeffSignatureBreakdown(graph, g_topologyCache);
    const std::uint64_t coeffSig = coeffSigBreakdown.combined();
    if (g_directTStats.calls > 1 && s_lastCoeffSig != 0 && coeffSig != s_lastCoeffSig) {
        ++g_directTStats.coeffSigChanged;
        if (g_lastCoeffSigBreakdown.flowSig != coeffSigBreakdown.flowSig) {
            ++g_directTStats.coeffSigFlowChanged;
        }
        if (g_lastCoeffSigBreakdown.airconOnSig != coeffSigBreakdown.airconOnSig) {
            ++g_directTStats.coeffSigAirconOnChanged;
        }
        if (g_lastCoeffSigBreakdown.setNodeActiveSig != coeffSigBreakdown.setNodeActiveSig) {
            ++g_directTStats.coeffSigSetNodeChanged;
        }
    }
    s_lastCoeffSig = coeffSig;
    g_lastCoeffSigBreakdown = coeffSigBreakdown;
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

    const std::uint64_t rhsHash = hashRhsValues(system.b);

    static std::vector<double> temperaturesBuffer;
    if (temperaturesBuffer.size() != n) temperaturesBuffer.assign(n, 0.0);
    else std::fill(temperaturesBuffer.begin(), temperaturesBuffer.end(), 0.0);
    std::vector<double>& temperatures = temperaturesBuffer;
    bool solved = false;
    std::string method = "LLT";
    const bool canReusePreviousSolution =
        canReuseFactorization &&
        g_cachedSolutionReuse.valid &&
        g_cachedSolutionReuse.graphPtr == &graph &&
        g_cachedSolutionReuse.n == n &&
        g_cachedSolutionReuse.coeffSig == coeffSig &&
        g_cachedSolutionReuse.rhsHash == rhsHash &&
        g_cachedSolutionReuse.temperatures.size() == n;
    if (canReuseFactorization) {
        ++g_directTStats.solveCached;
        if (canReusePreviousSolution) {
            ++g_directTStats.rhsSolutionReuse;
            temperatures = g_cachedSolutionReuse.temperatures;
            method = makeRhsCachedLabel(g_cachedSolutionReuse.method);
            solved = true;
        } else {
            static Eigen::VectorXd eb;
            if (eb.size() != static_cast<int>(n)) eb.resize(static_cast<int>(n));
            for (size_t i = 0; i < n; ++i) eb[static_cast<int>(i)] = system.b[i];
            solved = detail::solveWithCachedFactorization(eb, temperatures, constants.thermalTolerance, logFile, method);
        }
    } else {
        ++g_directTStats.solveFull;
        solved = detail::solveSparseDirect(system, temperatures, constants.thermalTolerance, logFile, method);
        if (solved) g_sparseLuCache.coeffSig = coeffSig;
    }

    if (!solved) {
        throw std::runtime_error("thermal solve failed (direct absolute T solver)");
    }

    g_cachedSolutionReuse.valid = true;
    g_cachedSolutionReuse.graphPtr = &graph;
    g_cachedSolutionReuse.n = n;
    g_cachedSolutionReuse.coeffSig = coeffSig;
    g_cachedSolutionReuse.rhsHash = rhsHash;
    g_cachedSolutionReuse.temperatures = temperatures;
    g_cachedSolutionReuse.method = stripRhsCachedSuffixes(method);

    for (size_t i = 0; i < n; ++i) graph[g_topologyCache.parameterIndexToVertex[i]].current_t = temperatures[i];
    const bool canReusePostprocess =
        canReusePreviousSolution &&
        g_cachedPostprocessReuse.valid &&
        g_cachedPostprocessReuse.graphPtr == &graph &&
        g_cachedPostprocessReuse.n == n &&
        g_cachedPostprocessReuse.coeffSig == coeffSig &&
        g_cachedPostprocessReuse.rhsHash == rhsHash;

    if (canReusePostprocess) {
        ++g_directTStats.postprocessReuse;
        network.setLastThermalConvergence(
            g_cachedPostprocessReuse.converged,
            g_cachedPostprocessReuse.rmse,
            g_cachedPostprocessReuse.maxBalance,
            makeRhsCachedLabel(g_cachedPostprocessReuse.method));
        auto durUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - startTime);
        const double durMs = static_cast<double>(durUs.count()) / 1000.0;
        std::ostringstream oss;
        oss << "--------熱計算(線形): "
            << (g_cachedPostprocessReuse.converged ? "収束" : "未収束")
            << " (method=" << makeRhsCachedLabel(g_cachedPostprocessReuse.method)
            << ", RMSE=" << std::scientific << std::setprecision(6) << g_cachedPostprocessReuse.rmse
            << ", maxBalance=" << g_cachedPostprocessReuse.maxBalance
            << ", time=" << std::fixed << std::setprecision(3) << durMs << "ms"
            << ", post=cached)";
        writeLog(logFile, oss.str());
    } else {
        detail::postprocessAndReport(network, graph, g_topologyCache, curV, n, constants, method, logFile, startTime, g_directTStats);
    }

    g_cachedPostprocessReuse.valid = true;
    g_cachedPostprocessReuse.graphPtr = &graph;
    g_cachedPostprocessReuse.n = n;
    g_cachedPostprocessReuse.coeffSig = coeffSig;
    g_cachedPostprocessReuse.rhsHash = rhsHash;
    g_cachedPostprocessReuse.converged = network.getLastThermalConverged();
    g_cachedPostprocessReuse.rmse = network.getLastThermalRmseBalance();
    g_cachedPostprocessReuse.maxBalance = network.getLastThermalMaxBalance();
    g_cachedPostprocessReuse.method = stripRhsCachedSuffixes(network.getLastThermalMethod());
}

} // namespace ThermalSolverLinearDirect
