#pragma once

#include "../../vtsim_solver.h"
#include "core/thermal/thermal_direct_response.h"
#include "core/thermal/thermal_linear_utils.h"
#include "core/thermal/heat_calculation.h"
#include "utils/utils.h"
#include "../../network/thermal_network.h"

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <new>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThermalSolverLinearDirect::detail {

struct TopologyCache {
    const Graph* graphPtr = nullptr;
    size_t numVertices = 0;
    size_t numEdges = 0;

    std::vector<std::vector<Edge>> incidentEdges;
    std::vector<std::vector<Vertex>> airconBySetVertex;
    std::vector<Vertex> airconSetVertex;

    std::vector<Edge> advectionEdges;
    std::vector<Edge> responseEdges;
    std::vector<Vertex> airconVertices;

    std::vector<std::string> nodeNames;
    std::vector<int> vertexToParameterIndex;
    std::vector<Vertex> parameterIndexToVertex;
    std::vector<std::vector<int>> rowColsPattern;

    using RowIndexMap = thermal_linear_utils::RowIndexMap;
    std::vector<RowIndexMap> rowIndexMaps;

    // --- b(右辺)だけ更新するための前計算（係数状態が変わった時だけ作り直す） ---
    struct KnownTerm {
        Vertex v = std::numeric_limits<Vertex>::max(); // known vertex (non-calc_t)
        double coeff = 0.0;                            // b -= coeff * T_known
    };
    struct HeatGenTerm {
        Edge e{};
        double sign = 0.0; // b += sign * current_heat_generation
    };
    struct ResponseHistTerm {
        Edge e{};
        bool isSrc = true;  // true: src-side history, false: tgt-side history
        double factor = 1.0;
    };
    std::vector<std::vector<KnownTerm>> knownTermsByRow;
    std::vector<std::vector<HeatGenTerm>> heatGenByRow;
    std::vector<std::vector<ResponseHistTerm>> responseHistByRow;
    std::vector<Vertex> fixedRowAirconVertex; // [row] -> aircon vertex providing set temp (or max if not fixed)
    std::uint64_t rhsCoeffSig = 0;

    bool initialized = false;
};

struct LinearSystem {
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    std::vector<std::vector<int>> colIndices;

    void initWithPattern(const std::vector<std::vector<int>>& rowColsPattern) {
        const size_t n = rowColsPattern.size();
        A.resize(n);
        b.assign(n, 0.0);
        colIndices = rowColsPattern;
        for (size_t i = 0; i < n; ++i) A[i].assign(colIndices[i].size(), 0.0);
    }

    void resetValuesKeepPattern() {
        std::fill(b.begin(), b.end(), 0.0);
        for (auto& rowA : A) std::fill(rowA.begin(), rowA.end(), 0.0);
    }

    inline void addCoefficientLocal(size_t row, int localIdx, double value) {
        if (std::abs(value) >= 1e-15 && localIdx >= 0) A[row][static_cast<size_t>(localIdx)] += value;
    }
};

struct SparseLUCache {
    bool analyzed = false;
    int n = 0;
    size_t nnz = 0;
    std::uint64_t patternHash = 0;
    bool factorized = false;
    std::uint64_t valueHash = 0;
    std::uint64_t coeffSig = 0;
    Eigen::SparseMatrix<double> A;
    std::vector<std::vector<int>> valuePtrIndexByRow; // system.colIndices と同型
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
};

struct SparseCholeskyCache {
    bool analyzed = false;
    int n = 0;
    size_t nnz = 0;
    std::uint64_t patternHash = 0;
    bool factorized = false;
    std::uint64_t valueHash = 0;
    bool patternSymmetric = false;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
};

struct DirectTStats {
    std::uint64_t calls = 0;
    std::uint64_t coeffSigChanged = 0;
    std::uint64_t reuseMissNotAnalyzed = 0;
    std::uint64_t reuseMissNoFactorized = 0;
    std::uint64_t reuseMissSizeMismatch = 0;
    std::uint64_t reuseMissCoeffSigMismatch = 0;
    std::uint64_t topoRebuild = 0;
    std::uint64_t rhsPrecomputeRebuild = 0;
    std::uint64_t rhsOnlyBuild = 0;
    std::uint64_t fullBuild = 0;
    std::uint64_t patternRebuild = 0;
    std::uint64_t luFactorize = 0;
    std::uint64_t cholFactorize = 0;
    std::uint64_t solveCached = 0;
    std::uint64_t solveFull = 0;
};

extern TopologyCache g_topologyCache;
extern SparseLUCache g_sparseLuCache;
extern SparseCholeskyCache g_cholCache;
extern DirectTStats g_directTStats;
extern std::uint64_t s_lastCoeffSig;

// --- split implementation functions ---
void rebuildTopologyCache(ThermalNetwork& network, const Graph& graph, size_t curV, size_t curE, TopologyCache& topo);

std::uint64_t computeCoeffSignature(const Graph& graph, const TopologyCache& topo);
void rebuildRhsPrecomputeForCoeffSig(const Graph& graph, TopologyCache& topo, std::uint64_t coeffSig);
void buildRhsOnlyAbsoluteFast(const Graph& graph, const TopologyCache& topo, std::vector<double>& bOut);

void buildLinearSystemAbsoluteFast(const Graph& graph, const TopologyCache& topo, LinearSystem& system);

bool solveSparseDirect(const LinearSystem& system,
                      std::vector<double>& x,
                      double tolerance,
                      std::ostream& logFile,
                      std::string& methodLabel);

bool solveWithCachedFactorization(const Eigen::VectorXd& b,
                                 std::vector<double>& x,
                                 double tolerance,
                                 std::ostream& logFile,
                                 std::string& methodLabel);

void postprocessAndReport(ThermalNetwork& network,
                          Graph& graph,
                          const TopologyCache& topo,
                          size_t curV,
                          size_t n,
                          const SimulationConstants& constants,
                          const std::string& method,
                          std::ostream& logFile,
                          std::chrono::high_resolution_clock::time_point startTime,
                          DirectTStats& stats);

} // namespace ThermalSolverLinearDirect::detail


