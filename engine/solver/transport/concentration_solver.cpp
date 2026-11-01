#include "transport/concentration_solver.h"

#include "network/contaminant_network.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <Eigen/SparseLU>
#include <limits>
#include <vector>

#include <boost/range/iterator_range.hpp>

namespace transport {
namespace {
constexpr double kConcentrationNegEpsilon = 1e-12;
} // namespace

ConcentrationSolveStats updateConcentrationIfEnabled(const SimulationConstants& constants,
                                                     VentilationNetwork& ventNetwork,
                                                     Graph& nodeGraph,
                                                     ConstNodeStateView nodeState,
                                                     ContaminantNetwork& contaminantNetwork,
                                                     std::ostream& logs,
                                                     TimingList& timings,
                                                     const std::string& meta) {
    ConcentrationSolveStats stats{};
    if (!constants.concentrationCalc) return stats;
    const bool logEnabled = constants.logVerbosity >= 1;

    ScopedTimer timer(timings, "concentration_update", meta);

    auto& tGraph = nodeGraph;
    auto& vGraph = ventNetwork.getGraph();
    const auto& vKeyToV = ventNetwork.getKeyToVertex();

    const double dt = static_cast<double>(constants.timestep);
    if (!(dt > 0.0)) return stats;

    auto idxOf = [](Vertex v) -> size_t { return static_cast<size_t>(v); };
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));

    // c(t) の前値（全ノード）
    std::vector<double> cOld(nV, 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        cOld[idxOf(v)] = tGraph[v].current_c;
    }

    ContaminantNetworkTerms terms;
    contaminantNetwork.buildTerms(nodeState, ventNetwork, terms);
    stats.activeVertices = static_cast<int>(terms.updateVertices.size());
    if (terms.updateVertices.empty()) {
        stats.converged = true;
        return stats;
    }

    // 有限体積の同時輸送: V dc/dt = g + Σ(q_eff c_src) - (Σq_out + Vβ)c。
    // 上流を旧値に固定せず連立陰解法で解き、閉鎖系のΣV cを保存する。
    // 他の未知節点と結合しない行は従来どおり局所解析解を使う。
    const int n = static_cast<int>(terms.updateVertices.size());
    std::vector<int> rowByVertex(nV, -1);
    for (int r = 0; r < n; ++r) rowByVertex[idxOf(terms.updateVertices[r])] = r;
    std::vector<bool> coupled(n, false);
    for (int r = 0; r < n; ++r) {
        for (const auto& in : terms.inflow[idxOf(terms.updateVertices[r])]) {
            const int col = rowByVertex[idxOf(in.first)];
            if (col >= 0 && in.second > 0.0) coupled[r] = coupled[col] = true;
        }
    }

    using SparseMatrix = Eigen::SparseMatrix<double>;
    std::vector<Eigen::Triplet<double>> entries;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
    for (int r = 0; r < n; ++r) {
        const Vertex v = terms.updateVertices[r];
        const size_t i = idxOf(v);
        const double volume = std::max(0.0, tGraph[v].v);
        const auto gen = terms.genByVertex.find(v);
        double source = gen == terms.genByVertex.end() ? 0.0 : gen->second;
        const double loss = terms.outSum[i] + volume * tGraph[v].current_beta;
        for (const auto& in : terms.inflow[i]) {
            const int col = rowByVertex[idxOf(in.first)];
            if (col < 0) source += in.second * cOld[idxOf(in.first)];
            else if (in.second != 0.0) entries.emplace_back(r, col, -in.second);
        }
        if (volume > 0.0 && !coupled[r]) {
            // expm1で微小な減衰率でも発生・流入項の桁落ちを避ける。
            const double rate = loss / volume;
            const double gain = rate == 0.0 ? dt : -std::expm1(-rate * dt) / rate;
            rhs[r] = cOld[i] * std::exp(-rate * dt) + (source / volume) * gain;
            entries.emplace_back(r, r, 1.0);
        } else {
            double diagonal = volume / dt + loss;
            rhs[r] = (volume / dt) * cOld[i] + source;
            // 出口のないゼロ容量混合節点は、除去前流入量で混合する。
            if (volume == 0.0 && diagonal == 0.0) diagonal = terms.inSum[i];
            if (diagonal == 0.0 && terms.inflow[i].empty() && source == 0.0) {
                diagonal = 1.0;
                rhs[r] = cOld[i];
            }
            entries.emplace_back(r, r, diagonal);
        }
    }

    SparseMatrix matrix(n, n);
    matrix.setFromTriplets(entries.begin(), entries.end());
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
        stats.converged = false;
        return stats;
    }
    const Eigen::VectorXd solved = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !solved.allFinite() ||
        (matrix * solved - rhs).lpNorm<Eigen::Infinity>() >
            1e-10 * std::max(1e-30, rhs.lpNorm<Eigen::Infinity>())) {
        stats.converged = false;
        return stats;
    }
    std::vector<double> cNew = cOld;
    for (int r = 0; r < n; ++r) cNew[idxOf(terms.updateVertices[r])] = solved[r];

    // 検証してから一括反映（失敗時は旧状態を維持）
    for (Vertex v : terms.updateVertices) {
        const double ci = cNew[idxOf(v)];
        if (!std::isfinite(ci) || ci < -kConcentrationNegEpsilon) {
            stats.converged = false;
            stats.updated = false;
            if (logEnabled) {
                writeDomainLog(logs, "濃度",
                    "[ERROR] 更新失敗: 非有限または負の濃度 node=" + tGraph[v].key);
            }
            return stats;
        }
    }

    for (Vertex v : terms.updateVertices) {
        const size_t i = idxOf(v);
        const double ci = (cNew[i] < 0.0) ? 0.0 : cNew[i];
        tGraph[v].current_c = ci;
        auto itV = vKeyToV.find(tGraph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_c = ci;
        }
    }
    stats.converged = true;
    stats.updated = true;
    if (logEnabled) {
        writeDomainLog(logs, "濃度",
            "更新完了: nodes=" + std::to_string(terms.updateVertices.size()));
    }
    return stats;
}

} // namespace transport
