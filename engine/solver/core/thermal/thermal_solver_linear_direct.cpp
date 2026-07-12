#include "core/thermal/thermal_solver_linear_direct.h"
#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect {

namespace {

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
    return getDirectTCacheStats(detail::defaultDirectTContext());
}

DirectTCacheStats getDirectTCacheStats(detail::DirectTSolverContext& ctx) {
    const auto& stats = ctx.stats;
    DirectTCacheStats s;
    s.calls = stats.calls;
    s.coeffSigChanged = stats.coeffSigChanged;
    s.rhsOnlyBuild = stats.rhsOnlyBuild;
    s.fullBuild = stats.fullBuild;
    s.solveCached = stats.solveCached;
    s.solveFull = stats.solveFull;
    return s;
}

void resetDirectTSolverContext() {
    detail::defaultDirectTContext().reset();
}

void resetDirectTCacheStats() {
    // テストの再現性のため、統計とキャッシュをまとめて無効化
    resetDirectTSolverContext();
}

void solveTemperaturesLinearDirect(ThermalNetwork& network,
                                   const SimulationConstants& constants,
                                   std::ostream& logFile) {
    solveTemperaturesLinearDirect(network, constants, logFile, detail::defaultDirectTContext());
}

void solveTemperaturesLinearDirect(ThermalNetwork& network,
                                   const SimulationConstants& constants,
                                   std::ostream& logFile,
                                   detail::DirectTSolverContext& ctx) {
    ++ctx.stats.calls;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto& graph = network.getGraph();
    const size_t curV = boost::num_vertices(graph);
    const size_t curE = boost::num_edges(graph);

    const bool needRebuildTopo =
        (!ctx.topology.initialized) ||
        (ctx.topology.graphPtr != &graph) ||
        (ctx.topology.numVertices != curV) ||
        (ctx.topology.numEdges != curE) ||
        (ctx.topology.topologyRevision != network.getTopologyRevision());

    if (needRebuildTopo) {
        ++ctx.stats.topoRebuild;
        detail::rebuildTopologyCache(network, graph, curV, curE, ctx.topology);
    }

    const size_t n = ctx.topology.nodeNames.size();
    if (n == 0) {
        network.setLastThermalConvergence(true, 0.0, 0.0, "DirectT(no-active-node)");
        return;
    }

    if (needRebuildTopo || ctx.systemGraphPtr != &graph || ctx.systemN != n || ctx.system.colIndices.size() != n) {
        ctx.system.initWithPattern(ctx.topology.rowColsPattern);
        ctx.systemN = n;
        ctx.systemGraphPtr = &graph;
    }

    // 湿りエンタルピー組立コンテキスト（係数署名より前に同期）
    ctx.topology.moist.enabled = constants.moistEnthalpyEnabled;
    ctx.topology.moist.dt = static_cast<double>(constants.timestep);
    ctx.topology.moist.humidityXnByVertex = network.moistEnthalpyHumidityXn();

    const detail::CoeffSignatureBreakdown coeffSigBreakdown =
        detail::computeCoeffSignatureBreakdown(graph, ctx.topology);
    const std::uint64_t coeffSig = coeffSigBreakdown.combined();
    if (ctx.stats.calls > 1 && ctx.lastCoeffSig != 0 && coeffSig != ctx.lastCoeffSig) {
        ++ctx.stats.coeffSigChanged;
        if (ctx.lastCoeffSigBreakdown.flowSig != coeffSigBreakdown.flowSig) {
            ++ctx.stats.coeffSigFlowChanged;
        }
        if (ctx.lastCoeffSigBreakdown.airconOnSig != coeffSigBreakdown.airconOnSig) {
            ++ctx.stats.coeffSigAirconOnChanged;
        }
        if (ctx.lastCoeffSigBreakdown.setNodeActiveSig != coeffSigBreakdown.setNodeActiveSig) {
            ++ctx.stats.coeffSigSetNodeChanged;
        }
    }
    ctx.lastCoeffSig = coeffSig;
    ctx.lastCoeffSigBreakdown = coeffSigBreakdown;
    if (ctx.topology.rhsCoeffSig != coeffSig ||
        ctx.topology.fixedRowAirconVertex.size() != n ||
        ctx.topology.knownTermsByRow.size() != n ||
        ctx.topology.responseHistByRow.size() != n ||
        ctx.topology.moistConstRhsByRow.size() != n) {
        ++ctx.stats.rhsPrecomputeRebuild;
        detail::rebuildRhsPrecomputeForCoeffSig(graph, ctx.topology, coeffSig);
    }

    bool canReuseFactorization = true;
    if (!ctx.sparseLu.analyzed) { canReuseFactorization = false; ++ctx.stats.reuseMissNotAnalyzed; }
    if (!(ctx.sparseLu.factorized || (ctx.chol.analyzed && ctx.chol.factorized))) {
        canReuseFactorization = false;
        ++ctx.stats.reuseMissNoFactorized;
    }
    if (ctx.sparseLu.n != static_cast<int>(n)) { canReuseFactorization = false; ++ctx.stats.reuseMissSizeMismatch; }
    if (ctx.sparseLu.coeffSig != coeffSig) { canReuseFactorization = false; ++ctx.stats.reuseMissCoeffSigMismatch; }

    if (canReuseFactorization) {
        ++ctx.stats.rhsOnlyBuild;
        detail::buildRhsOnlyAbsoluteFast(graph, ctx.topology, ctx.system.b);
    } else {
        ++ctx.stats.fullBuild;
        detail::buildLinearSystemAbsoluteFast(graph, ctx.topology, ctx.system);
    }

    const std::uint64_t rhsHash = hashRhsValues(ctx.system.b);

    if (ctx.temperaturesBuffer.size() != n) ctx.temperaturesBuffer.assign(n, 0.0);
    else std::fill(ctx.temperaturesBuffer.begin(), ctx.temperaturesBuffer.end(), 0.0);
    std::vector<double>& temperatures = ctx.temperaturesBuffer;
    bool solved = false;
    std::string method = "LLT";
    const bool canReusePreviousSolution =
        canReuseFactorization &&
        ctx.solutionReuse.valid &&
        ctx.solutionReuse.graphPtr == &graph &&
        ctx.solutionReuse.n == n &&
        ctx.solutionReuse.coeffSig == coeffSig &&
        ctx.solutionReuse.rhsHash == rhsHash &&
        ctx.solutionReuse.temperatures.size() == n;
    if (canReuseFactorization) {
        ++ctx.stats.solveCached;
        if (canReusePreviousSolution) {
            ++ctx.stats.rhsSolutionReuse;
            temperatures = ctx.solutionReuse.temperatures;
            method = makeRhsCachedLabel(ctx.solutionReuse.method);
            solved = true;
        } else {
            if (ctx.rhsBuffer.size() != static_cast<int>(n)) ctx.rhsBuffer.resize(static_cast<int>(n));
            for (size_t i = 0; i < n; ++i) ctx.rhsBuffer[static_cast<int>(i)] = ctx.system.b[i];
            solved = detail::solveWithCachedFactorization(
                ctx,
                ctx.rhsBuffer,
                temperatures,
                effectiveThermalLinearResidualRelativeTolerance(constants),
                logFile,
                method);
        }
    } else {
        ++ctx.stats.solveFull;
        solved = detail::solveSparseDirect(
            ctx,
            ctx.system,
            temperatures,
            effectiveThermalLinearResidualRelativeTolerance(constants),
            logFile,
            method);
        if (solved) ctx.sparseLu.coeffSig = coeffSig;
    }

    if (!solved) {
        throw std::runtime_error("thermal solve failed (direct absolute T solver)");
    }

    ctx.solutionReuse.valid = true;
    ctx.solutionReuse.graphPtr = &graph;
    ctx.solutionReuse.n = n;
    ctx.solutionReuse.coeffSig = coeffSig;
    ctx.solutionReuse.rhsHash = rhsHash;
    ctx.solutionReuse.temperatures = temperatures;
    ctx.solutionReuse.method = stripRhsCachedSuffixes(method);

    for (size_t i = 0; i < n; ++i) graph[ctx.topology.parameterIndexToVertex[i]].current_t = temperatures[i];
    const bool canReusePostprocess =
        canReusePreviousSolution &&
        ctx.postprocessReuse.valid &&
        ctx.postprocessReuse.graphPtr == &graph &&
        ctx.postprocessReuse.n == n &&
        ctx.postprocessReuse.coeffSig == coeffSig &&
        ctx.postprocessReuse.rhsHash == rhsHash;

    if (canReusePostprocess) {
        ++ctx.stats.postprocessReuse;
        network.setLastThermalConvergence(
            ctx.postprocessReuse.converged,
            ctx.postprocessReuse.rmse,
            ctx.postprocessReuse.maxBalance,
            makeRhsCachedLabel(ctx.postprocessReuse.method));
        auto durUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - startTime);
        const double durMs = static_cast<double>(durUs.count()) / 1000.0;
        std::ostringstream oss;
        oss << "--------熱計算(線形): "
            << (ctx.postprocessReuse.converged ? "収束" : "未収束")
            << " (method=" << makeRhsCachedLabel(ctx.postprocessReuse.method)
            << ", RMSE=" << std::scientific << std::setprecision(6) << ctx.postprocessReuse.rmse
            << ", maxBalance=" << ctx.postprocessReuse.maxBalance
            << ", time=" << std::fixed << std::setprecision(3) << durMs << "ms"
            << ", post=cached)";
        writeLog(logFile, oss.str());
    } else {
        detail::postprocessAndReport(network, graph, ctx.topology, curV, n, constants, method, logFile, startTime, ctx.stats);
    }

    ctx.postprocessReuse.valid = true;
    ctx.postprocessReuse.graphPtr = &graph;
    ctx.postprocessReuse.n = n;
    ctx.postprocessReuse.coeffSig = coeffSig;
    ctx.postprocessReuse.rhsHash = rhsHash;
    ctx.postprocessReuse.converged = network.getLastThermalConverged();
    ctx.postprocessReuse.rmse = network.getLastThermalRmseBalance();
    ctx.postprocessReuse.maxBalance = network.getLastThermalMaxBalance();
    ctx.postprocessReuse.method = stripRhsCachedSuffixes(network.getLastThermalMethod());
}

} // namespace ThermalSolverLinearDirect
