#include "core/humidity/humidity_coupling.h"

#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <boost/range/iterator_range.hpp>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace core::humidity {

namespace {
inline size_t idxOf(Vertex v) { return static_cast<size_t>(v); }

// 絶対湿度の許容下限。これ未満は非物理として不合格、[-eps,0) は 0 に丸める。
constexpr double kHumidityNegEpsilon = 1e-12;
} // namespace

void initializeHumidityState(const Graph& tGraph,
                             std::vector<double>& xOld,
                             std::vector<double>& xNew) {
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    xOld.assign(nV, 0.0);
    xNew.assign(nV, 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        const size_t i = idxOf(v);
        xOld[i] = tGraph[v].current_x;
        xNew[i] = tGraph[v].current_x;
    }
}

SolveStats solveHumidityImplicitStep(const Graph& tGraph,
                                     const HumidityNetworkTerms& terms,
                                     double dt,
                                     double tolerance,
                                     std::vector<double>& xNew,
                                     const std::vector<double>& xOld) {
    constexpr double rho = PhysicalConstants::DENSITY_DRY_AIR; // [kg/m3]
    const double tol = (tolerance > 0.0) ? tolerance : 1e-9;
    SolveStats stats{};
    const int n = static_cast<int>(terms.updateVertices.size());
    if (n <= 0) return stats;

    std::unordered_map<Vertex, int> rowByVertex;
    rowByVertex.reserve(static_cast<size_t>(n) * 2);
    for (int r = 0; r < n; ++r) {
        rowByVertex[terms.updateVertices[static_cast<size_t>(r)]] = r;
    }

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> trips;
    trips.reserve(static_cast<size_t>(n) * 8);
    Eigen::VectorXd b(n);
    b.setZero();

    auto addCoeff = [&](int row, Vertex colV, double coeff, double& rhsKnown) {
        if (std::abs(coeff) <= 0.0) return;
        auto itRow = rowByVertex.find(colV);
        if (itRow != rowByVertex.end()) {
            trips.emplace_back(row, itRow->second, coeff);
        } else {
            rhsKnown -= coeff * xOld[idxOf(colV)];
        }
    };

    for (int r = 0; r < n; ++r) {
        const Vertex v = terms.updateVertices[static_cast<size_t>(r)];
        const size_t i = idxOf(v);
        const double V = tGraph[v].v;
        const double cap = (tGraph[v].moisture_capacity > 0.0)
                               ? tGraph[v].moisture_capacity
                               : (rho * V);
        const auto itG = terms.genByVertex.find(v);
        const double g = (itG == terms.genByVertex.end()) ? 0.0 : itG->second;

        double rhs = 0.0;
        if (cap > 0.0) {
            // (1 + dt*(out+sum(k))/cap) * x_i
            // - dt*(md/cap)*x_src - dt*(k/cap)*x_nb = x_old + dt*g/cap
            double diag = 1.0 + dt * terms.outSum[i] / cap;
            rhs = xOld[i] + dt * (g / cap);

            for (const auto& in : terms.inflow[i]) {
                const Vertex sv = in.first;
                const double md = in.second;
                addCoeff(r, sv, -dt * (md / cap), rhs);
            }
            for (const auto& lk : terms.moistureLinks[i]) {
                const Vertex ov = lk.first;
                const double k = lk.second;
                diag += dt * (k / cap);
                addCoeff(r, ov, -dt * (k / cap), rhs);
            }
            trips.emplace_back(r, r, diag);
            b[r] = rhs;
        } else {
            // 容量なしノードの定常収支:
            // (outSum + Σk) x_i - Σ(m_in x_src) - Σ(k x_neighbor) = generation
            const bool hasFlow = (terms.outSum[i] > 0.0) || !terms.inflow[i].empty();
            const bool hasLinks = !terms.moistureLinks[i].empty();
            if (!hasFlow && !hasLinks) {
                // 移流も湿気リンクもなければ現状態を保持
                trips.emplace_back(r, r, 1.0);
                b[r] = xOld[i];
            } else {
                double diag = terms.outSum[i];
                rhs = g;
                for (const auto& in : terms.inflow[i]) {
                    addCoeff(r, in.first, -in.second, rhs);
                }
                for (const auto& lk : terms.moistureLinks[i]) {
                    const double k = lk.second;
                    diag += k;
                    addCoeff(r, lk.first, -k, rhs);
                }
                // diag==0 は特異（流入のみ・流出なし等）。解不能として失敗側へ回すため 0 のまま置く。
                trips.emplace_back(r, r, diag);
                b[r] = rhs;
            }
        }
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(trips.begin(), trips.end());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        stats.converged = false;
        stats.iterations = 0;
        stats.finalRelativeResidual = std::numeric_limits<double>::infinity();
        return stats;
    }

    Eigen::VectorXd x = solver.solve(b);
    stats.iterations = 1;
    stats.converged = (solver.info() == Eigen::Success);

    if (!stats.converged || x.size() != n) {
        stats.converged = false;
        stats.finalRelativeResidual = std::numeric_limits<double>::infinity();
        return stats;
    }

    // 直接法でも診断値として相対残差 ||Ax-b||/||b|| を保持
    const Eigen::VectorXd residual = A * x - b;
    const double bNorm = b.norm();
    const double relResidual = (bNorm > 0.0) ? (residual.norm() / bNorm) : residual.norm();
    stats.finalRelativeResidual = relResidual;
    if (!(relResidual <= tol) || !std::isfinite(relResidual)) {
        stats.converged = false;
        return stats;
    }

    for (int r = 0; r < n; ++r) {
        const double xi = x[r];
        if (!std::isfinite(xi) || xi < -kHumidityNegEpsilon) {
            stats.converged = false;
            stats.finalRelativeResidual = std::numeric_limits<double>::infinity();
            return stats;
        }
        const Vertex v = terms.updateVertices[static_cast<size_t>(r)];
        // 微小な負値のみ 0 に丸める
        xNew[idxOf(v)] = (xi < 0.0) ? 0.0 : xi;
    }
    return stats;
}

void applyHumidityStateToGraphs(Graph& tGraph,
                                Graph& vGraph,
                                const std::unordered_map<std::string, Vertex>& vKeyToV,
                                const std::vector<Vertex>& updateVertices,
                                const std::vector<double>& xNew) {
    for (Vertex v : updateVertices) {
        const size_t i = idxOf(v);
        tGraph[v].current_x = xNew[i];
        tGraph[v].current_w = xNew[i];
        auto itV = vKeyToV.find(tGraph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_x = xNew[i];
        }
    }
}

} // namespace core::humidity
