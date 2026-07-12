#include "core/humidity/humidity_coupling.h"

#include "core/thermal/thermal_linear_utils.h"

#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/range/iterator_range.hpp>

namespace core::humidity {

namespace {
inline size_t idxOf(Vertex v) { return static_cast<size_t>(v); }

// 絶対湿度の許容下限。これ未満は非物理として不合格、[-eps,0) は 0 に丸める。
constexpr double kHumidityNegEpsilon = 1e-12;

using thermal_linear_utils::fnv1a64_update;
using thermal_linear_utils::hashDoubleBits;

void failSolve(SolveStats& stats) {
    stats.converged = false;
    stats.iterations = 0;
    stats.finalRelativeResidual = std::numeric_limits<double>::infinity();
}

std::uint64_t hashRhsVector(const Eigen::VectorXd& b) {
    std::uint64_t h = 0;
    for (Eigen::Index i = 0; i < b.size(); ++i) {
        h = hashDoubleBits(h, b[i]);
    }
    return h;
}
} // namespace

void HumiditySolverContext::invalidate() {
    updateVertices.clear();
    rowByVertex.clear();
    matrix.resize(0, 0);
    solver.reset();
    rhs.resize(0);
    solution.resize(0);
    patternSignature = 0;
    coefficientSignature = 0;
    rhsSignature = 0;
    analyzed = false;
    factorized = false;
    lastRelativeResidual = 0.0;
    // 累積メトリクスは診断用に保持する
}

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
                                     HumiditySolverContext& ctx) {
    constexpr double rho = PhysicalConstants::DENSITY_DRY_AIR; // [kg/m3]
    const double tol = (tolerance > 0.0) ? tolerance : 1e-9;
    SolveStats stats{};
    const int n = static_cast<int>(terms.updateVertices.size());
    if (n <= 0) return stats;

    if (ctx.xN.size() < static_cast<size_t>(boost::num_vertices(tGraph)) ||
        ctx.xIterate.size() < static_cast<size_t>(boost::num_vertices(tGraph))) {
        failSolve(stats);
        return stats;
    }

    // --- row map / signatures -------------------------------------------------
    std::unordered_map<Vertex, int> rowByVertex;
    rowByVertex.reserve(static_cast<size_t>(n) * 2);
    for (int r = 0; r < n; ++r) {
        rowByVertex[terms.updateVertices[static_cast<size_t>(r)]] = r;
    }

    std::uint64_t patternSig = 0;
    std::uint64_t coeffSig = 0;
    patternSig = fnv1a64_update(patternSig, static_cast<std::uint64_t>(n));
    coeffSig = hashDoubleBits(coeffSig, dt);

    for (int r = 0; r < n; ++r) {
        const Vertex v = terms.updateVertices[static_cast<size_t>(r)];
        const size_t i = idxOf(v);
        patternSig = fnv1a64_update(patternSig, static_cast<std::uint64_t>(v));
        // 対角は常に存在
        patternSig = fnv1a64_update(
            patternSig,
            (static_cast<std::uint64_t>(r) << 32) ^ static_cast<std::uint64_t>(r));

        const double V = tGraph[v].v;
        const double cap = (tGraph[v].moisture_capacity > 0.0)
                               ? tGraph[v].moisture_capacity
                               : (rho * V);
        const auto itG = terms.genByVertex.find(v);
        const double g = (itG == terms.genByVertex.end()) ? 0.0 : itG->second;

        coeffSig = hashDoubleBits(coeffSig, cap);
        coeffSig = hashDoubleBits(coeffSig, terms.outSum[i]);
        coeffSig = fnv1a64_update(coeffSig, (cap > 0.0) ? 1u : 0u);

        // パターンは流量非依存の換気隣接＋湿気リンク（固定上位パターン）
        if (i < terms.ventNeighbors.size()) {
            for (const Vertex nv : terms.ventNeighbors[i]) {
                auto itCol = rowByVertex.find(nv);
                if (itCol != rowByVertex.end()) {
                    patternSig = fnv1a64_update(
                        patternSig,
                        (static_cast<std::uint64_t>(r) << 32) ^
                            static_cast<std::uint64_t>(itCol->second));
                }
            }
        }
        for (const auto& in : terms.inflow[i]) {
            coeffSig = fnv1a64_update(coeffSig, static_cast<std::uint64_t>(in.first));
            coeffSig = hashDoubleBits(coeffSig, in.second);
        }
        for (const auto& lk : terms.moistureLinks[i]) {
            const Vertex ov = lk.first;
            const double k = lk.second;
            coeffSig = fnv1a64_update(coeffSig, static_cast<std::uint64_t>(ov));
            coeffSig = hashDoubleBits(coeffSig, k);
            auto itCol = rowByVertex.find(ov);
            if (itCol != rowByVertex.end()) {
                patternSig = fnv1a64_update(
                    patternSig,
                    (static_cast<std::uint64_t>(r) << 32) ^
                        static_cast<std::uint64_t>(itCol->second));
            }
        }

        if (!(cap > 0.0)) {
            const bool hasFlow = (terms.outSum[i] > 0.0) || !terms.inflow[i].empty();
            const bool hasLinks = !terms.moistureLinks[i].empty();
            const bool holdIdentity = !hasFlow && !hasLinks && g == 0.0;
            coeffSig = fnv1a64_update(coeffSig, holdIdentity ? 1u : 0u);
            coeffSig = hashDoubleBits(coeffSig, g);
        }
    }

    const bool patternSame =
        ctx.analyzed &&
        ctx.solver &&
        ctx.patternSignature == patternSig &&
        static_cast<int>(ctx.updateVertices.size()) == n;
    const bool coeffsSame =
        patternSame &&
        ctx.factorized &&
        ctx.coefficientSignature == coeffSig;

    using Triplet = Eigen::Triplet<double>;
    auto inflowFrom = [&](size_t i, Vertex sv) -> double {
        for (const auto& in : terms.inflow[i]) {
            if (in.first == sv) return in.second;
        }
        return 0.0;
    };

    auto buildMatrixAndRhs = [&](std::vector<Triplet>* tripsOut, Eigen::VectorXd& bOut) {
        if (tripsOut) {
            tripsOut->clear();
            tripsOut->reserve(static_cast<size_t>(n) * 8);
        }
        bOut.resize(n);
        bOut.setZero();

        // 構造スロットは係数0でも残す（SparseLU の pattern 固定用）
        auto addStructural = [&](int row, Vertex colV, double coeff, double& rhsKnown) {
            auto itRow = rowByVertex.find(colV);
            if (itRow != rowByVertex.end()) {
                if (tripsOut) tripsOut->emplace_back(row, itRow->second, coeff);
            } else if (std::abs(coeff) > 0.0) {
                rhsKnown -= coeff * ctx.xIterate[idxOf(colV)];
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
                double diag = 1.0 + dt * terms.outSum[i] / cap;
                rhs = ctx.xN[i] + dt * (g / cap);

                if (i < terms.ventNeighbors.size()) {
                    for (const Vertex nv : terms.ventNeighbors[i]) {
                        const double md = inflowFrom(i, nv);
                        addStructural(r, nv, -dt * md / cap, rhs);
                    }
                }
                for (const auto& lk : terms.moistureLinks[i]) {
                    const double k = lk.second;
                    diag += dt * (k / cap);
                    addStructural(r, lk.first, -dt * (k / cap), rhs);
                }
                if (tripsOut) tripsOut->emplace_back(r, r, diag);
                bOut[r] = rhs;
            } else {
                const bool hasFlow = (terms.outSum[i] > 0.0) || !terms.inflow[i].empty();
                const bool hasLinks = !terms.moistureLinks[i].empty();
                if (!hasFlow && !hasLinks && g == 0.0) {
                    // identity でも換気隣接スロットを0で埋め pattern を固定
                    if (tripsOut) tripsOut->emplace_back(r, r, 1.0);
                    if (i < terms.ventNeighbors.size()) {
                        for (const Vertex nv : terms.ventNeighbors[i]) {
                            addStructural(r, nv, 0.0, rhs);
                        }
                    }
                    bOut[r] = ctx.xIterate[i];
                } else {
                    double diag = terms.outSum[i];
                    rhs = g;
                    if (i < terms.ventNeighbors.size()) {
                        for (const Vertex nv : terms.ventNeighbors[i]) {
                            addStructural(r, nv, -inflowFrom(i, nv), rhs);
                        }
                    }
                    for (const auto& lk : terms.moistureLinks[i]) {
                        const double k = lk.second;
                        diag += k;
                        addStructural(r, lk.first, -k, rhs);
                    }
                    if (tripsOut) tripsOut->emplace_back(r, r, diag);
                    bOut[r] = rhs;
                }
            }
        }
    };

    auto applySolutionToXSolved = [&](const Eigen::VectorXd& x) -> bool {
        if (ctx.xSolved.size() < static_cast<size_t>(boost::num_vertices(tGraph))) {
            ctx.xSolved.resize(static_cast<size_t>(boost::num_vertices(tGraph)));
        }
        for (int r = 0; r < n; ++r) {
            const double xi = x[r];
            if (!std::isfinite(xi) || xi < -kHumidityNegEpsilon) {
                return false;
            }
            const Vertex v = terms.updateVertices[static_cast<size_t>(r)];
            ctx.xSolved[idxOf(v)] = (xi < 0.0) ? 0.0 : xi;
        }
        return true;
    };

    auto finalizeResidual = [&](const Eigen::VectorXd& x) -> bool {
        const Eigen::VectorXd residual = ctx.matrix * x - ctx.rhs;
        const double bNorm = ctx.rhs.norm();
        const double relResidual = (bNorm > 0.0) ? (residual.norm() / bNorm) : residual.norm();
        stats.finalRelativeResidual = relResidual;
        ctx.lastRelativeResidual = relResidual;
        if (!(relResidual <= tol) || !std::isfinite(relResidual)) {
            stats.converged = false;
            return false;
        }
        return true;
    };

    // --- Level 3: coeffs unchanged → RHS only (or solution reuse) ------------
    if (coeffsSame) {
        buildMatrixAndRhs(nullptr, ctx.rhs);
        const std::uint64_t rhsSig = hashRhsVector(ctx.rhs);
        if (ctx.rhsSignature == rhsSig &&
            ctx.solution.size() == n) {
            ++ctx.solutionReuse;
            stats.solutionReuse = 1;
            stats.iterations = 1;
            stats.converged = true;
            stats.finalRelativeResidual = ctx.lastRelativeResidual;
            if (!applySolutionToXSolved(ctx.solution)) {
                failSolve(stats);
                ctx.factorized = false;
                return stats;
            }
            return stats;
        }

        ++ctx.rhsOnlySolves;
        stats.rhsOnlySolves = 1;
        ctx.solution = ctx.solver->solve(ctx.rhs);
        stats.iterations = 1;
        stats.converged = (ctx.solver->info() == Eigen::Success);
        if (!stats.converged || ctx.solution.size() != n) {
            failSolve(stats);
            return stats;
        }
        if (!finalizeResidual(ctx.solution)) return stats;
        if (!applySolutionToXSolved(ctx.solution)) {
            failSolve(stats);
            return stats;
        }
        ctx.rhsSignature = rhsSig;
        return stats;
    }

    // --- Rebuild matrix: pattern same → factorize only; else analyze+factorize ---
    std::vector<Triplet> trips;
    buildMatrixAndRhs(&trips, ctx.rhs);

    ctx.matrix.resize(n, n);
    ctx.matrix.setFromTriplets(trips.begin(), trips.end());
    ctx.matrix.makeCompressed();

    ctx.updateVertices = terms.updateVertices;
    ctx.rowByVertex = std::move(rowByVertex);
    ctx.patternSignature = patternSig;
    ctx.coefficientSignature = coeffSig;
    ctx.rhsSignature = 0; // invalidate until successful solve

    if (patternSame) {
        // 同一 sparsity: symbolic を再利用し数値分解のみ
        ctx.solver->factorize(ctx.matrix);
        if (ctx.solver->info() != Eigen::Success) {
            ctx.analyzed = false;
            ctx.factorized = false;
            failSolve(stats);
            return stats;
        }
        ctx.factorized = true;
        ++ctx.factorizes;
        stats.factorizes = 1;
    } else {
        ctx.solver = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
        ctx.solver->analyzePattern(ctx.matrix);
        ++ctx.patternAnalyzes;
        stats.patternAnalyzes = 1;

        ctx.solver->factorize(ctx.matrix);
        if (ctx.solver->info() != Eigen::Success) {
            ctx.analyzed = false;
            ctx.factorized = false;
            failSolve(stats);
            return stats;
        }
        ctx.analyzed = true;
        ctx.factorized = true;
        ++ctx.factorizes;
        stats.factorizes = 1;
    }

    ctx.solution = ctx.solver->solve(ctx.rhs);
    stats.iterations = 1;
    stats.converged = (ctx.solver->info() == Eigen::Success);
    if (!stats.converged || ctx.solution.size() != n) {
        failSolve(stats);
        return stats;
    }
    if (!finalizeResidual(ctx.solution)) return stats;
    if (!applySolutionToXSolved(ctx.solution)) {
        failSolve(stats);
        return stats;
    }
    ctx.rhsSignature = hashRhsVector(ctx.rhs);
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

void evaluateMoistureBalanceTerms(const Graph& tGraph,
                                  const HumidityNetworkTerms& terms,
                                  const std::vector<double>& xN,
                                  double dt,
                                  MoistureBalanceTerms& out) {
    constexpr double rho = PhysicalConstants::DENSITY_DRY_AIR;
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    out.ventilationTransport.assign(nV, 0.0);
    out.vaporGeneration.assign(nV, 0.0);
    out.materialPhaseChange.assign(nV, 0.0);
    out.airconCondensation.assign(nV, 0.0);
    out.storage.assign(nV, 0.0);
    out.residual.assign(nV, 0.0);
    out.maxAbsResidual = 0.0;
    if (!(dt > 0.0) || nV == 0) {
        return;
    }

    auto xOf = [&](Vertex v) -> double {
        return tGraph[v].current_x;
    };

    std::unordered_set<Vertex> active;
    active.reserve(terms.updateVertices.size() * 2 + 1);
    for (Vertex v : terms.updateVertices) {
        active.insert(v);
    }

    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        const size_t i = idxOf(v);
        const double xi = xOf(v);

        double vent = 0.0;
        if (i < terms.outSum.size()) {
            vent -= terms.outSum[i] * xi;
        }
        if (i < terms.inflow.size()) {
            for (const auto& in : terms.inflow[i]) {
                vent += in.second * xOf(in.first);
            }
        }
        out.ventilationTransport[i] = vent;

        const auto itG = terms.genByVertex.find(v);
        out.vaporGeneration[i] = (itG == terms.genByVertex.end()) ? 0.0 : itG->second;

        double mat = 0.0;
        if (i < terms.moistureLinks.size()) {
            for (const auto& lk : terms.moistureLinks[i]) {
                mat += lk.second * (xOf(lk.first) - xi);
            }
        }
        out.materialPhaseChange[i] = mat;

        // 空調除湿診断: 吹出境界 x=supplyX で移流に織り込み済み。残差には載せないよう
        // active ノードでは通常 0。空調ノードへ除去量（空気系から見た負の生成）を記録。
        if (tGraph[v].getTypeCode() == VertexProperties::TypeCode::Aircon) {
            out.airconCondensation[i] = -std::max(0.0, tGraph[v].aircon_moisture_removal_kg_s);
        }

        const double cap = (tGraph[v].moisture_capacity > 0.0)
                               ? tGraph[v].moisture_capacity
                               : (rho * tGraph[v].v);
        if (cap > 0.0 && i < xN.size()) {
            out.storage[i] = cap * (xi - xN[i]) / dt;
        } else {
            out.storage[i] = 0.0;
        }

        out.residual[i] = out.storage[i]
                          - (out.ventilationTransport[i] + out.vaporGeneration[i]
                             + out.materialPhaseChange[i] + out.airconCondensation[i]);
        // 方程式を解いた calc_x ノードのみ residual を検算する
        if (active.count(v) != 0) {
            out.maxAbsResidual = std::max(out.maxAbsResidual, std::abs(out.residual[i]));
        }
    }
}

} // namespace core::humidity
