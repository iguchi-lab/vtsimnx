#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect::detail {

namespace {

void resetSparseLuCache(SparseLUCache& cache) {
    cache.analyzed = false;
    cache.factorized = false;
    cache.n = 0;
    cache.nnz = 0;
    cache.patternHash = 0;
    cache.valueHash = 0;
    cache.coeffSig = 0;
    cache.valuePtrIndexByRow.clear();
    cache.A.resize(0, 0);
    cache.solver.~SparseLU();
    new (&cache.solver) Eigen::SparseLU<Eigen::SparseMatrix<double>>();
}

void resetCholCache(SparseCholeskyCache& cache) {
    cache.analyzed = false;
    cache.factorized = false;
    cache.n = 0;
    cache.nnz = 0;
    cache.patternHash = 0;
    cache.valueHash = 0;
    cache.patternSymmetric = false;
    cache.llt.~SimplicialLLT();
    new (&cache.llt) Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>();
    cache.ldlt.~SimplicialLDLT();
    new (&cache.ldlt) Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>();
}

} // namespace

DirectTSolverContext::~DirectTSolverContext() {
    resetOptionalDirectSolverCaches(*this);
}

void DirectTSolverContext::reset() {
    topology = TopologyCache{};
    resetSparseLuCache(sparseLu);
    resetCholCache(chol);
    resetOptionalDirectSolverCaches(*this);
    stats = DirectTStats{};
    lastCoeffSig = 0;
    lastCoeffSigBreakdown = CoeffSignatureBreakdown{};
    solutionReuse = CachedSolutionReuse{};
    postprocessReuse = CachedPostprocessReuse{};
    system = LinearSystem{};
    systemN = 0;
    systemGraphPtr = nullptr;
    temperaturesBuffer.clear();
    rhsBuffer.resize(0);
    cachedResidualCheckCounter = 0;
    luBackendState = -1;
}

DirectTSolverContext& defaultDirectTContext() {
    static DirectTSolverContext ctx;
    return ctx;
}

} // namespace ThermalSolverLinearDirect::detail
