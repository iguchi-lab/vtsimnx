#pragma once

#include "types/common_types.h"
#include "types/graph_types.h"
#include "network/humidity_network.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace core::humidity {

// 湿度疎直接法の 3 段キャッシュ（pattern / factorize / RHS）と x_n / x_k 作業バッファ。
struct HumiditySolverContext {
    std::vector<Vertex> updateVertices;
    std::unordered_map<Vertex, int> rowByVertex;
    Eigen::SparseMatrix<double> matrix;
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> solver;
    Eigen::VectorXd rhs;
    Eigen::VectorXd solution;
    std::vector<double> xN;       // previous timestep
    std::vector<double> xIterate; // coupling iterate (x_k)
    std::vector<double> xSolved;
    std::uint64_t patternSignature = 0;
    std::uint64_t coefficientSignature = 0;
    std::uint64_t rhsSignature = 0;
    bool analyzed = false;
    bool factorized = false;
    std::size_t patternAnalyzes = 0;
    std::size_t factorizes = 0;
    std::size_t rhsOnlySolves = 0;
    std::size_t solutionReuse = 0;
    double lastRelativeResidual = 0.0;

    void invalidate();
};

struct SolveStats {
    int iterations = 0;
    // ||Ax-b|| / ||b|| （連成の湿度変化量とは別物）
    double finalRelativeResidual = 0.0;
    bool converged = true;
    std::size_t patternAnalyzes = 0;
    std::size_t factorizes = 0;
    std::size_t rhsOnlySolves = 0;
    std::size_t solutionReuse = 0;
};

void initializeHumidityState(const Graph& tGraph,
                             std::vector<double>& xOld,
                             std::vector<double>& xNew);

SolveStats solveHumidityImplicitStep(const Graph& tGraph,
                                     const HumidityNetworkTerms& terms,
                                     double dt,
                                     double tolerance,
                                     HumiditySolverContext& ctx);

void applyHumidityStateToGraphs(Graph& tGraph,
                                Graph& vGraph,
                                const std::unordered_map<std::string, Vertex>& vKeyToV,
                                const std::vector<Vertex>& updateVertices,
                                const std::vector<double>& xNew);

// 求解後の x と terms / xN から水分収支内訳を評価（数値ソルバには影響しない）
void evaluateMoistureBalanceTerms(const Graph& tGraph,
                                  const HumidityNetworkTerms& terms,
                                  const std::vector<double>& xN,
                                  double dt,
                                  MoistureBalanceTerms& out);

} // namespace core::humidity
