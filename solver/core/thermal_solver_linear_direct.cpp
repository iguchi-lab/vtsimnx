#include "core/thermal_solver_linear_direct.h"
#include "core/heat_calculation.h"
#include "utils/utils.h"
#include "../network/thermal_network.h"

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <limits>
#include <new>
#include <stdexcept>

namespace ThermalSolverLinearDirect {

namespace {

struct TopologyCache {
    const Graph* graphPtr = nullptr;
    size_t numVertices = 0;
    size_t numEdges = 0;

    // Graph は vecS なので Vertex は 0..V-1（vector 化で高速化）
    std::vector<std::vector<Edge>> incidentEdges;       // [vertex] -> incident edges
    std::vector<std::vector<Vertex>> airconBySetVertex; // [setVertex] -> aircon vertices
    std::vector<Vertex> airconSetVertex;                // [airconVertex] -> setVertex（無ければ max）

    // 係数シグネチャ計算の高速化用（トポロジ不変の間は一定）
    std::vector<Edge> advectionEdges;                   // advection 辺の一覧
    std::vector<Vertex> airconVertices;                 // aircon 頂点の一覧

    std::vector<std::string> nodeNames;
    std::vector<int> vertexToParameterIndex;            // [vertex] -> param idx（無ければ -1）
    std::vector<Vertex> parameterIndexToVertex;         // [param idx] -> vertex
    std::vector<std::vector<int>> rowColsPattern;       // [row(param idx)] -> sorted unique cols (param idx)

    // rowColsPattern と同型の高速 lookup（col -> local index）
    struct RowIndexMap {
        int mask = 0;                 // tableSize-1（2冪）
        std::vector<int> keys;        // col (param idx) / empty=-1
        std::vector<int> values;      // local index in row (0..rowLen-1) / empty=-1

        void clear() {
            mask = 0;
            keys.clear();
            values.clear();
        }

        static int nextPow2(int x) {
            int p = 1;
            while (p < x) p <<= 1;
            return p;
        }

        void buildFromCols(const std::vector<int>& cols) {
            const int need = std::max(2, static_cast<int>(cols.size()) * 2);
            const int tableSize = nextPow2(need);
            mask = tableSize - 1;
            keys.assign(static_cast<size_t>(tableSize), -1);
            values.assign(static_cast<size_t>(tableSize), -1);
            for (int local = 0; local < static_cast<int>(cols.size()); ++local) {
                const int col = cols[static_cast<size_t>(local)];
                uint32_t h = static_cast<uint32_t>(col) * 2654435761u;
                int idx = static_cast<int>(h) & mask;
                for (int probe = 0; probe < tableSize; ++probe) {
                    if (keys[static_cast<size_t>(idx)] == -1 || keys[static_cast<size_t>(idx)] == col) {
                        keys[static_cast<size_t>(idx)] = col;
                        values[static_cast<size_t>(idx)] = local;
                        break;
                    }
                    idx = (idx + 1) & mask;
                }
            }
        }

        inline int get(int col) const {
            const int tableSize = mask + 1;
            if (tableSize <= 0) return -1;
            uint32_t h = static_cast<uint32_t>(col) * 2654435761u;
            int idx = static_cast<int>(h) & mask;
            for (int probe = 0; probe < tableSize; ++probe) {
                const int k = keys[static_cast<size_t>(idx)];
                if (k == col) return values[static_cast<size_t>(idx)];
                if (k == -1) return -1;
                idx = (idx + 1) & mask;
            }
            return -1;
        }
    };

    std::vector<RowIndexMap> rowIndexMaps; // [row] -> lookup table

    // --- b 更新高速化用（係数不変の間だけ有効） ---
    struct KnownTerm {
        Vertex v = std::numeric_limits<Vertex>::max(); // known vertex
        double coeff = 0.0;                            // A(row, v) 係数
    };
    struct HeatGenTerm {
        Edge e{};
        double sign = 0.0; // b += sign * current_heat_generation
    };
    std::vector<std::vector<KnownTerm>> knownTermsByRow;   // [row] -> known terms
    std::vector<std::vector<HeatGenTerm>> heatGenByRow;    // [row] -> heat generation terms
    std::vector<Vertex> fixedRowAirconVertex;              // [row] -> aircon vertex providing set temp / max if not fixed
    uint64_t rhsCoeffSig = 0;                               // この前計算が対応する coeffSig

    bool initialized = false;
};

static TopologyCache g_topologyCache;

struct SparseLUCache {
    bool analyzed = false;
    int n = 0;
    size_t nnz = 0;
    uint64_t patternHash = 0;
    bool factorized = false;
    uint64_t valueHash = 0; // A の係数値ハッシュ（同一なら factorize を再利用可能）
    uint64_t coeffSig = 0;  // A が「変わったか」判定用の軽量シグネチャ（flow/固定行など）
    Eigen::SparseMatrix<double> A;
    std::vector<std::vector<int>> valuePtrIndexByRow; // system.colIndices と同型
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
};

static SparseLUCache g_sparseLuCache;

struct SparseCholeskyCache {
    bool analyzed = false;
    int n = 0;
    size_t nnz = 0;
    uint64_t patternHash = 0;
    bool factorized = false;
    uint64_t valueHash = 0;
    bool patternSymmetric = false;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
};

static SparseCholeskyCache g_cholCache;

static uint64_t fnv1a64_update(uint64_t h, uint64_t v) {
    constexpr uint64_t kFnvOffset = 14695981039346656037ull;
    constexpr uint64_t kFnvPrime  = 1099511628211ull;
    if (h == 0) h = kFnvOffset;
    h ^= v;
    h *= kFnvPrime;
    return h;
}

static inline uint64_t hashDoubleBits(uint64_t h, double x) {
    uint64_t bits = 0;
    static_assert(sizeof(double) == sizeof(uint64_t), "double size mismatch");
    std::memcpy(&bits, &x, sizeof(bits));
    return fnv1a64_update(h, bits);
}

static inline bool isSymmetricPatternByCols(const std::vector<std::vector<int>>& colIndices) {
    const int n = static_cast<int>(colIndices.size());
    for (int r = 0; r < n; ++r) {
        const auto& colsR = colIndices[static_cast<size_t>(r)];
        for (int c : colsR) {
            if (c == r) continue;
            if (c < 0 || c >= n) return false;
            const auto& colsC = colIndices[static_cast<size_t>(c)];
            auto it = std::lower_bound(colsC.begin(), colsC.end(), r);
            if (it == colsC.end() || *it != r) return false;
        }
    }
    return true;
}

// A が変わったかを判定するための軽量シグネチャを作る。
// 温度に依存しない係数（conductance と |flow_rate|）と、固定行（set_node の aircon on）だけを見る。
static uint64_t computeCoeffSignature(const Graph& graph, const TopologyCache& topo) {
    uint64_t h = 0;

    for (auto e : topo.advectionEdges) {
        const auto& ep = graph[e];
        Vertex sv = boost::source(e, graph);
        Vertex tv = boost::target(e, graph);
        h = fnv1a64_update(h, (static_cast<uint64_t>(static_cast<uint32_t>(sv)) << 32) ^
                               static_cast<uint64_t>(static_cast<uint32_t>(tv)));
        double flowRate = ep.flow_rate;
        double coeff = 0.0;
        if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
            coeff = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
        }
        h = hashDoubleBits(h, coeff);
    }

    // 固定温度行（set_node に紐づく aircon が on）の有無
    for (size_t setV = 0; setV < topo.airconBySetVertex.size(); ++setV) {
        if (topo.airconBySetVertex[setV].empty()) continue;
        bool anyOn = false;
        for (Vertex v_ac : topo.airconBySetVertex[setV]) {
            const auto& nd = graph[v_ac];
            if (nd.getTypeCode() == VertexProperties::TypeCode::Aircon && nd.on) {
                anyOn = true;
                break;
            }
        }
        if (anyOn) {
            h = fnv1a64_update(h, static_cast<uint64_t>(setV));
            h = fnv1a64_update(h, 1u);
        }
    }

    // aircon shoulder は on/off で行が変わるので、on/off を入れる
    for (auto v : topo.airconVertices) {
        const auto& nd = graph[v];
        h = fnv1a64_update(h, static_cast<uint64_t>(static_cast<uint32_t>(v)));
        h = fnv1a64_update(h, nd.on ? 1u : 0u);
    }

    return h;
}

// 線形システム Ax=b（絶対温度 T を未知数）を構築
struct LinearSystem {
    std::vector<std::vector<double>> A;             // 各行の値（疎形式）
    std::vector<double> b;                          // 右辺
    std::vector<std::vector<int>> colIndices;       // 各行の非ゼロ列

    void initWithPattern(const std::vector<std::vector<int>>& rowColsPattern) {
        const size_t n = rowColsPattern.size();
        A.resize(n);
        b.assign(n, 0.0);
        colIndices = rowColsPattern;
        for (size_t i = 0; i < n; ++i) {
            A[i].assign(colIndices[i].size(), 0.0);
        }
    }

    void resetValuesKeepPattern() {
        std::fill(b.begin(), b.end(), 0.0);
        for (auto& rowA : A) std::fill(rowA.begin(), rowA.end(), 0.0);
    }

    inline void addCoefficientLocal(size_t row, int localIdx, double value) {
        if (std::abs(value) < 1e-15) return;
        if (localIdx < 0) return;
        auto& rowA = A[row];
        const size_t k = static_cast<size_t>(localIdx);
        if (k >= rowA.size()) return;
        rowA[k] += value;
    }
};

// （完全線形）b（右辺）だけを構築（A は作らない）。
// A は温度に依存しないので、既知温度（非calc_t）だけ RHS に吸収して b を作る。
// （未知=calc_t の係数は A 側にあるので b には入れない）
static void buildRhsOnlyAbsoluteFast(const Graph& graph,
                                     const TopologyCache& topo,
                                     std::vector<double>& bOut) {
    const size_t n = topo.nodeNames.size();
    bOut.assign(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        // 固定温度行（set_node の aircon が ON）
        const Vertex v_ac = topo.fixedRowAirconVertex.empty() ? std::numeric_limits<Vertex>::max()
                                                              : topo.fixedRowAirconVertex[i];
        if (v_ac != std::numeric_limits<Vertex>::max()) {
            bOut[i] = graph[v_ac].current_pre_temp;
            continue;
        }
        // 既知温度寄与（b -= Aik * T_known）
        if (i < topo.knownTermsByRow.size()) {
            for (const auto& term : topo.knownTermsByRow[i]) {
                if (term.v == std::numeric_limits<Vertex>::max() || term.coeff == 0.0) continue;
                bOut[i] -= term.coeff * graph[term.v].current_t;
            }
        }
        // 発熱寄与（b += sign * q0）
        if (i < topo.heatGenByRow.size()) {
            for (const auto& tg : topo.heatGenByRow[i]) {
                bOut[i] += tg.sign * graph[tg.e].current_heat_generation;
            }
        }
    }
}

static void rebuildRhsPrecomputeForCoeffSig(const Graph& graph,
                                            TopologyCache& topo,
                                            uint64_t coeffSig) {
    const size_t n = topo.nodeNames.size();
    topo.knownTermsByRow.assign(n, {});
    topo.heatGenByRow.assign(n, {});
    topo.fixedRowAirconVertex.assign(n, std::numeric_limits<Vertex>::max());

    auto isUnknown = [&](Vertex v) -> bool {
        return topo.vertexToParameterIndex[static_cast<size_t>(v)] >= 0;
    };

    auto addKnown = [&](size_t row, Vertex v, double coeff) {
        if (std::abs(coeff) < 1e-15) return;
        if (isUnknown(v)) return;
        topo.knownTermsByRow[row].push_back(TopologyCache::KnownTerm{v, coeff});
    };

    auto addHeatGen = [&](size_t row, Edge e, double sign) {
        if (std::abs(sign) < 1e-15) return;
        topo.heatGenByRow[row].push_back(TopologyCache::HeatGenTerm{e, sign});
    };

    auto processNodeNet = [&](size_t row, Vertex nodeVertex, double factor) {
        for (auto edge : topo.incidentEdges[static_cast<size_t>(nodeVertex)]) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& ep = graph[edge];

            double k = 0.0;
            bool isGen = false;
            if (ep.getTypeCode() == EdgeProperties::TypeCode::Conductance) {
                k = ep.conductance;
            } else if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) {
                double flowRate = ep.flow_rate;
                if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
                    k = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
                }
            } else if (ep.getTypeCode() == EdgeProperties::TypeCode::HeatGeneration) {
                isGen = true;
            } else {
                continue;
            }

            if (sv == nodeVertex) {
                // net(source) += -Q, Q = k(Ts-Tt)+q0 -> (-k)*Ts + (+k)*Tt = +q0
                if (k != 0.0) {
                    addKnown(row, sv, factor * (-k));
                    addKnown(row, tv, factor * (+k));
                }
                if (isGen) {
                    addHeatGen(row, edge, factor * (+1.0));
                }
            } else if (tv == nodeVertex) {
                // net(target) += +Q -> (+k)*Ts + (-k)*Tt = -q0
                if (k != 0.0) {
                    addKnown(row, sv, factor * (+k));
                    addKnown(row, tv, factor * (-k));
                }
                if (isGen) {
                    addHeatGen(row, edge, factor * (-1.0));
                }
            }
        }
    };

    for (size_t i = 0; i < n; ++i) {
        Vertex nodeVertex = topo.parameterIndexToVertex[i];

        // 固定温度行（set_node の aircon が ON）なら、どの aircon を使うか確定して保持
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(nodeVertex)]) {
            const auto& nd = graph[v_ac];
            if (nd.getTypeCode() == VertexProperties::TypeCode::Aircon && nd.on) {
                topo.fixedRowAirconVertex[i] = v_ac;
                break;
            }
        }
        if (topo.fixedRowAirconVertex[i] != std::numeric_limits<Vertex>::max()) {
            continue; // b は setTemp だけ
        }

        processNodeNet(i, nodeVertex, +1.0);

        const auto& nodeData = graph[nodeVertex];
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(nodeVertex)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                processNodeNet(i, setV, -1.0);
            }
        }

        // known term をまとめておく（rowLenは小さい想定）
        auto& terms = topo.knownTermsByRow[i];
        std::sort(terms.begin(), terms.end(),
                  [](const auto& a, const auto& b) { return a.v < b.v; });
        size_t w = 0;
        for (size_t r = 0; r < terms.size(); ++r) {
            if (w == 0 || terms[r].v != terms[w - 1].v) {
                terms[w++] = terms[r];
            } else {
                terms[w - 1].coeff += terms[r].coeff;
            }
        }
        terms.resize(w);
    }

    topo.rhsCoeffSig = coeffSig;
}

// （完全線形）A と b を構築（絶対温度 AT=b）
static void buildLinearSystemAbsoluteFast(const Graph& graph,
                                          const TopologyCache& topo,
                                          LinearSystem& system) {
    const size_t n = topo.parameterIndexToVertex.size();
    if (system.colIndices.size() == n && system.A.size() == n && system.b.size() == n &&
        (n == 0 || !system.colIndices[0].empty())) {
        system.resetValuesKeepPattern();
    } else {
        system.initWithPattern(topo.rowColsPattern);
    }

    // 固定温度行（set_node の aircon が ON）：T = setTemp
    std::vector<uint8_t> isFixedRow(n, 0);
    std::vector<double> fixedTemp(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Vertex nodeVertex = topo.parameterIndexToVertex[i];
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(nodeVertex)]) {
            const auto& nd = graph[v_ac];
            if (nd.getTypeCode() == VertexProperties::TypeCode::Aircon && nd.on) {
                isFixedRow[i] = 1;
                fixedTemp[i] = nd.current_pre_temp;
                break;
            }
        }
    }

    auto addCoeffOrKnownToB = [&](size_t row, const TopologyCache::RowIndexMap& rowMap, Vertex colVertex, double aCoeff) {
        const int colIdx = topo.vertexToParameterIndex[static_cast<size_t>(colVertex)];
        if (colIdx >= 0) {
            const int local = rowMap.get(colIdx);
            if (local >= 0) system.addCoefficientLocal(row, local, aCoeff);
        } else {
            // known: move to RHS
            system.b[row] -= aCoeff * graph[colVertex].current_t;
        }
    };

    auto processNodeNetIntoRow = [&](size_t row, const TopologyCache::RowIndexMap& rowMap, Vertex nodeVertex, double factor) {
        for (auto edge : topo.incidentEdges[static_cast<size_t>(nodeVertex)]) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& ep = graph[edge];

            double k = 0.0;
            double q0 = 0.0;
            if (ep.getTypeCode() == EdgeProperties::TypeCode::Conductance) {
                k = ep.conductance;
            } else if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) {
                double flowRate = ep.flow_rate;
                if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
                    k = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
                }
            } else if (ep.getTypeCode() == EdgeProperties::TypeCode::HeatGeneration) {
                q0 = ep.current_heat_generation;
            } else {
                continue;
            }

            if (sv == nodeVertex) {
                // (-k)*Ts + (+k)*Tt = +q0
                if (k != 0.0) {
                    addCoeffOrKnownToB(row, rowMap, sv, factor * (-k));
                    addCoeffOrKnownToB(row, rowMap, tv, factor * (+k));
                }
                if (q0 != 0.0) system.b[row] += factor * (+q0);
            } else if (tv == nodeVertex) {
                // (+k)*Ts + (-k)*Tt = -q0
                if (k != 0.0) {
                    addCoeffOrKnownToB(row, rowMap, sv, factor * (+k));
                    addCoeffOrKnownToB(row, rowMap, tv, factor * (-k));
                }
                if (q0 != 0.0) system.b[row] += factor * (-q0);
            }
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const auto& rowMap = topo.rowIndexMaps[i];

        if (isFixedRow[i]) {
            // resetValuesKeepPattern でゼロ化済みなので diag と b だけ入れる
            system.b[i] = fixedTemp[i];
            const int localDiag = rowMap.get(static_cast<int>(i));
            if (localDiag >= 0) system.addCoefficientLocal(i, localDiag, 1.0);
            continue;
        }

        Vertex nodeVertex = topo.parameterIndexToVertex[i];
        const auto& nodeData = graph[nodeVertex];

        processNodeNetIntoRow(i, rowMap, nodeVertex, +1.0);
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(nodeVertex)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                processNodeNetIntoRow(i, rowMap, setV, -1.0);
            }
        }
    }
}

static bool solveSparseDirect(const LinearSystem& system,
                             std::vector<double>& x,
                             double tolerance,
                             std::ostream& logFile,
                             std::string& methodLabel) {
    const size_t n = x.size();
    if (n == 0) return true;

    size_t nnz = 0;
    for (size_t i = 0; i < n; ++i) nnz += system.colIndices[i].size();

    uint64_t patternHash = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto& cols = system.colIndices[i];
        for (size_t k = 0; k < cols.size(); ++k) {
            patternHash = fnv1a64_update(patternHash, (static_cast<uint64_t>(i) << 32) ^ static_cast<uint64_t>(cols[k]));
        }
    }

    Eigen::VectorXd b(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) b[static_cast<int>(i)] = system.b[i];

    const bool needRebuildPattern = (!g_sparseLuCache.analyzed) ||
                                   (g_sparseLuCache.n != static_cast<int>(n)) ||
                                   (g_sparseLuCache.nnz != nnz) ||
                                   (g_sparseLuCache.patternHash != patternHash);
    if (needRebuildPattern) {
        g_sparseLuCache.analyzed = false;
        g_sparseLuCache.n = static_cast<int>(n);
        g_sparseLuCache.nnz = nnz;
        g_sparseLuCache.patternHash = patternHash;
        g_sparseLuCache.factorized = false;
        g_sparseLuCache.valueHash = 0;
        g_sparseLuCache.valuePtrIndexByRow.clear();
        g_sparseLuCache.A.resize(0, 0);

        g_sparseLuCache.solver.~SparseLU();
        new (&g_sparseLuCache.solver) Eigen::SparseLU<Eigen::SparseMatrix<double>>();

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nnz);
        uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& cols = system.colIndices[i];
            const auto& vals = system.A[i];
            for (size_t k = 0; k < cols.size(); ++k) {
                triplets.emplace_back(static_cast<int>(i), cols[k], vals[k]);
                valueHash = hashDoubleBits(valueHash, vals[k]);
            }
        }

        g_sparseLuCache.A = Eigen::SparseMatrix<double>(static_cast<int>(n), static_cast<int>(n));
        g_sparseLuCache.A.setFromTriplets(triplets.begin(), triplets.end());
        g_sparseLuCache.A.makeCompressed();

        // valuePtr インデックス表を構築
        g_sparseLuCache.valuePtrIndexByRow.assign(n, {});
        std::vector<std::vector<std::pair<int, int>>> rowEntries(n);
        for (size_t r = 0; r < n; ++r) rowEntries[r].reserve(system.colIndices[r].size());
        double* base = g_sparseLuCache.A.valuePtr();
        for (int outer = 0; outer < g_sparseLuCache.A.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(g_sparseLuCache.A, outer); it; ++it) {
                const int r = it.row();
                const int c = it.col();
                const int p = static_cast<int>(&it.valueRef() - base);
                if (r >= 0 && r < static_cast<int>(n)) {
                    rowEntries[static_cast<size_t>(r)].emplace_back(c, p);
                }
            }
        }
        for (size_t r = 0; r < n; ++r) {
            auto& entries = rowEntries[r];
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            const auto& cols = system.colIndices[r];
            g_sparseLuCache.valuePtrIndexByRow[r].assign(cols.size(), -1);
            size_t j = 0;
            for (size_t k = 0; k < cols.size(); ++k) {
                const int col = cols[k];
                while (j < entries.size() && entries[j].first < col) ++j;
                if (j < entries.size() && entries[j].first == col) {
                    g_sparseLuCache.valuePtrIndexByRow[r][k] = entries[j].second;
                }
            }
        }
        bool mappingOk = true;
        for (size_t r = 0; r < n && mappingOk; ++r) {
            for (int p : g_sparseLuCache.valuePtrIndexByRow[r]) {
                if (p < 0) {
                    mappingOk = false;
                    break;
                }
            }
        }
        if (!mappingOk) {
            writeLog(logFile, "--------疎直接法(DirectT): valuePtrIndexByRow の構築に失敗（パターン不一致）。停止します。");
            g_sparseLuCache.analyzed = false;
            g_sparseLuCache.factorized = false;
            g_sparseLuCache.valueHash = 0;
            g_sparseLuCache.valuePtrIndexByRow.clear();
            g_cholCache.analyzed = false;
            g_cholCache.factorized = false;
            return false;
        }

        g_sparseLuCache.solver.analyzePattern(g_sparseLuCache.A);
        g_sparseLuCache.analyzed = true;
        g_sparseLuCache.valueHash = valueHash;

        g_cholCache.analyzed = false;
        g_cholCache.factorized = false;
        g_cholCache.patternSymmetric = isSymmetricPatternByCols(system.colIndices);
    } else {
        uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& vals = system.A[i];
            for (size_t k = 0; k < vals.size(); ++k) {
                const int p = g_sparseLuCache.valuePtrIndexByRow[i][k];
                if (p < 0) {
                    writeLog(logFile, "--------疎直接法(DirectT): valuePtrIndexByRow が不正（-1）。停止します。");
                    g_sparseLuCache.factorized = false;
                    g_cholCache.factorized = false;
                    return false;
                }
                g_sparseLuCache.A.valuePtr()[p] = vals[k];
                valueHash = hashDoubleBits(valueHash, vals[k]);
            }
        }
        if (g_sparseLuCache.valueHash != valueHash) {
            g_sparseLuCache.factorized = false;
            g_sparseLuCache.valueHash = valueHash;
            g_cholCache.factorized = false;
        }
    }

    const bool symmetricCandidate = g_cholCache.patternSymmetric;
    Eigen::VectorXd sol;
    bool solved = false;
    if (symmetricCandidate) {
        const bool needAnalyze = (!g_cholCache.analyzed) ||
                                 (g_cholCache.n != static_cast<int>(n)) ||
                                 (g_cholCache.nnz != nnz) ||
                                 (g_cholCache.patternHash != patternHash);
        if (needAnalyze) {
            g_cholCache.analyzed = false;
            g_cholCache.n = static_cast<int>(n);
            g_cholCache.nnz = nnz;
            g_cholCache.patternHash = patternHash;
            g_cholCache.factorized = false;
            g_cholCache.valueHash = 0;
            g_cholCache.llt.analyzePattern(g_sparseLuCache.A);
            g_cholCache.ldlt.analyzePattern(g_sparseLuCache.A);
            g_cholCache.analyzed = true;
        }

        if (!g_cholCache.factorized || g_cholCache.valueHash != g_sparseLuCache.valueHash) {
            g_cholCache.llt.factorize(g_sparseLuCache.A);
            if (g_cholCache.llt.info() == Eigen::Success) {
                g_cholCache.factorized = true;
                g_cholCache.valueHash = g_sparseLuCache.valueHash;
                sol = g_cholCache.llt.solve(b);
                if (g_cholCache.llt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "SimplicialLLT";
                }
            }
            if (!solved) {
                g_cholCache.ldlt.factorize(g_sparseLuCache.A);
                if (g_cholCache.ldlt.info() == Eigen::Success) {
                    g_cholCache.factorized = true;
                    g_cholCache.valueHash = g_sparseLuCache.valueHash;
                    sol = g_cholCache.ldlt.solve(b);
                    if (g_cholCache.ldlt.info() == Eigen::Success) {
                        solved = true;
                        methodLabel = "SimplicialLDLT";
                    }
                }
            }
            if (!solved) {
                g_cholCache.factorized = false;
                g_cholCache.patternSymmetric = false;
            }
        } else {
            sol = g_cholCache.llt.solve(b);
            if (g_cholCache.llt.info() == Eigen::Success) {
                solved = true;
                methodLabel = "SimplicialLLT(cached)";
            } else {
                sol = g_cholCache.ldlt.solve(b);
                if (g_cholCache.ldlt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "SimplicialLDLT(cached)";
                }
            }
        }
    }

    if (!solved) {
        if (!g_sparseLuCache.factorized) {
            g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
            if (g_sparseLuCache.solver.info() != Eigen::Success) {
                writeLog(logFile, "--------疎直接法(DirectT:SparseLU)のfactorizeに失敗。停止します。");
                return false;
            }
            g_sparseLuCache.factorized = true;
        }
        sol = g_sparseLuCache.solver.solve(b);
        if (g_sparseLuCache.solver.info() != Eigen::Success) {
            writeLog(logFile, "--------疎直接法(DirectT:SparseLU)のsolveに失敗。停止します。");
            return false;
        }
        methodLabel = "SparseLU";
    }

    Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
    double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
    if (!std::isfinite(maxResidual)) {
        writeLog(logFile, "--------疎直接法(DirectT)で非有限の残差が発生。停止します。");
        return false;
    }
    if (maxResidual > tolerance * 10.0) {
        std::ostringstream oss;
        oss << "--------疎直接法(DirectT)の残差が大きいため停止します: maxResidual="
            << std::scientific << std::setprecision(6) << maxResidual
            << " tol=" << tolerance;
        writeLog(logFile, oss.str());
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) {
            writeLog(logFile, "--------疎直接法(DirectT)で非有限の解が発生。停止します。");
            return false;
        }
        x[i] = v;
    }
    return true;
}

static bool solveWithCachedFactorization(const Eigen::VectorXd& b,
                                        std::vector<double>& x,
                                        double tolerance,
                                        std::ostream& logFile,
                                        std::string& methodLabel) {
    const size_t n = x.size();
    if (n == 0) return true;

    Eigen::VectorXd sol;
    bool ok = false;
    static uint64_t s_cachedResidualCheckCounter = 0;

    if (g_cholCache.analyzed && g_cholCache.factorized && g_cholCache.patternSymmetric) {
        sol = g_cholCache.llt.solve(b);
        if (g_cholCache.llt.info() == Eigen::Success) {
            ok = true;
            methodLabel = "SimplicialLLT(cached)";
        } else {
            sol = g_cholCache.ldlt.solve(b);
            if (g_cholCache.ldlt.info() == Eigen::Success) {
                ok = true;
                methodLabel = "SimplicialLDLT(cached)";
            }
        }
    }
    if (!ok && g_sparseLuCache.analyzed && g_sparseLuCache.factorized) {
        sol = g_sparseLuCache.solver.solve(b);
        if (g_sparseLuCache.solver.info() == Eigen::Success) {
            ok = true;
            methodLabel = "SparseLU(cached)";
        }
    }
    if (!ok) return false;

    constexpr uint64_t kCachedResidualCheckInterval = 200;
    if ((s_cachedResidualCheckCounter++ % kCachedResidualCheckInterval) == 0) {
        Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
        double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
        if (!std::isfinite(maxResidual)) {
            writeLog(logFile, "--------疎直接法(DirectT:cached)で非有限の残差が発生。停止します。");
            return false;
        }
        if (maxResidual > tolerance * 10.0) {
            std::ostringstream oss;
            oss << "--------疎直接法(DirectT:cached)の残差が大きいため停止します: maxResidual="
                << std::scientific << std::setprecision(6) << maxResidual
                << " tol=" << tolerance;
            writeLog(logFile, oss.str());
            return false;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) return false;
        x[i] = v;
    }
    return true;
}

} // namespace

void solveTemperaturesLinearDirect(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    std::ostream& logFile) {

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
        g_topologyCache = TopologyCache{};
        g_topologyCache.graphPtr = &graph;
        g_topologyCache.numVertices = curV;
        g_topologyCache.numEdges = curE;
        g_topologyCache.incidentEdges.assign(curV, {});

        for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            g_topologyCache.incidentEdges[static_cast<size_t>(sv)].push_back(edge);
            g_topologyCache.incidentEdges[static_cast<size_t>(tv)].push_back(edge);
        }

        g_topologyCache.airconBySetVertex.assign(curV, {});
        g_topologyCache.airconSetVertex.assign(curV, std::numeric_limits<Vertex>::max());
        for (auto vertex : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& properties = graph[vertex];
            if (properties.getTypeCode() == VertexProperties::TypeCode::Aircon && !properties.set_node.empty()) {
                auto itSet = network.getKeyToVertex().find(properties.set_node);
                if (itSet != network.getKeyToVertex().end()) {
                    Vertex setV = itSet->second;
                    g_topologyCache.airconBySetVertex[static_cast<size_t>(setV)].push_back(vertex);
                    g_topologyCache.airconSetVertex[static_cast<size_t>(vertex)] = setV;
                }
            }
        }

        g_topologyCache.advectionEdges.clear();
        g_topologyCache.advectionEdges.reserve(curE / 4 + 1);
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            if (graph[e].getTypeCode() == EdgeProperties::TypeCode::Advection) {
                g_topologyCache.advectionEdges.push_back(e);
            }
        }
        g_topologyCache.airconVertices.clear();
        g_topologyCache.airconVertices.reserve(curV / 16 + 1);
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon) {
                g_topologyCache.airconVertices.push_back(v);
            }
        }

        g_topologyCache.nodeNames.clear();
        g_topologyCache.vertexToParameterIndex.assign(curV, -1);
        g_topologyCache.parameterIndexToVertex.clear();
        size_t parameterIndex = 0;
        for (auto vertex : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& properties = graph[vertex];
            if (properties.calc_t) {
                g_topologyCache.nodeNames.push_back(properties.key);
                g_topologyCache.vertexToParameterIndex[static_cast<size_t>(vertex)] = static_cast<int>(parameterIndex++);
                g_topologyCache.parameterIndexToVertex.push_back(vertex);
            }
        }

        g_topologyCache.rowColsPattern.assign(g_topologyCache.nodeNames.size(), {});
        for (size_t row = 0; row < g_topologyCache.nodeNames.size(); ++row) {
            Vertex v = g_topologyCache.parameterIndexToVertex[row];
            std::vector<int> cols;
            cols.reserve(16);
            cols.push_back(static_cast<int>(row));

            auto addVar = [&](Vertex vv) {
                int idx = g_topologyCache.vertexToParameterIndex[static_cast<size_t>(vv)];
                if (idx >= 0) cols.push_back(idx);
            };

            for (auto e : g_topologyCache.incidentEdges[static_cast<size_t>(v)]) {
                addVar(boost::source(e, graph));
                addVar(boost::target(e, graph));
            }

            Vertex setV = g_topologyCache.airconSetVertex[static_cast<size_t>(v)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                for (auto e2 : g_topologyCache.incidentEdges[static_cast<size_t>(setV)]) {
                    addVar(boost::source(e2, graph));
                    addVar(boost::target(e2, graph));
                }
            }

            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            g_topologyCache.rowColsPattern[row] = std::move(cols);
        }

        g_topologyCache.rowIndexMaps.assign(g_topologyCache.rowColsPattern.size(), {});
        for (size_t row = 0; row < g_topologyCache.rowColsPattern.size(); ++row) {
            g_topologyCache.rowIndexMaps[row].buildFromCols(g_topologyCache.rowColsPattern[row]);
        }

        g_topologyCache.initialized = true;
    }

    const auto& nodeNames = g_topologyCache.nodeNames;
    if (nodeNames.empty()) {
        writeLog(logFile, "--警告: 温度計算対象のノードがありません");
        return;
    }

    static LinearSystem system;
    static size_t systemN = 0;
    static const Graph* systemGraphPtr = nullptr;
    if (needRebuildTopo || systemGraphPtr != &graph || systemN != nodeNames.size() ||
        system.colIndices.size() != nodeNames.size()) {
        system.initWithPattern(g_topologyCache.rowColsPattern);
        systemN = nodeNames.size();
        systemGraphPtr = &graph;
    }

    const uint64_t coeffSig = computeCoeffSignature(graph, g_topologyCache);

    // b 更新高速化用の前計算（係数状態が変わったときだけ）
    if (g_topologyCache.rhsCoeffSig != coeffSig ||
        g_topologyCache.knownTermsByRow.size() != g_topologyCache.nodeNames.size() ||
        g_topologyCache.fixedRowAirconVertex.size() != g_topologyCache.nodeNames.size()) {
        rebuildRhsPrecomputeForCoeffSig(graph, g_topologyCache, coeffSig);
    }

    const bool canReuseFactorization =
        g_sparseLuCache.analyzed &&
        (g_sparseLuCache.factorized || (g_cholCache.analyzed && g_cholCache.factorized)) &&
        g_sparseLuCache.n == static_cast<int>(nodeNames.size()) &&
        g_sparseLuCache.coeffSig == coeffSig;

    if (canReuseFactorization) {
        // A は不変なので b だけ更新（温度評価なし）
        buildRhsOnlyAbsoluteFast(graph, g_topologyCache, system.b);
    } else {
        // A,b を係数だけで構築（温度評価なし）
        buildLinearSystemAbsoluteFast(graph, g_topologyCache, system);
    }

    std::vector<double> temperatures(nodeNames.size(), 0.0); // solve結果（絶対温度）
    bool solved = false;
    std::string methodLabel = "directT";
    {
        bool ok = false;
        if (canReuseFactorization) {
            Eigen::VectorXd b(static_cast<int>(temperatures.size()));
            for (size_t i = 0; i < temperatures.size(); ++i) b[static_cast<int>(i)] = system.b[i];
            ok = solveWithCachedFactorization(b, temperatures, constants.thermalTolerance, logFile, methodLabel);
            if (!ok) writeLog(logFile, "--------疎直接法(DirectT:cached)で失敗。停止します。");
        } else {
            ok = solveSparseDirect(system, temperatures, constants.thermalTolerance, logFile, methodLabel);
            if (ok) g_sparseLuCache.coeffSig = coeffSig;
        }
        solved = ok;
    }

    if (!solved) {
        throw std::runtime_error("thermal solve failed (direct absolute T solver)");
    }

    // 計算温度を graph に反映（calc_t ノードのみ）
    for (size_t i = 0; i < temperatures.size(); ++i) {
        const Vertex v = g_topologyCache.parameterIndexToVertex[i];
        graph[v].current_t = temperatures[i];
    }

    // vertex index ベース温度（後続のheatRates/収支計算を高速化）
    std::vector<double> tempsByVertex(curV, 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        tempsByVertex[static_cast<size_t>(v)] = graph[v].current_t;
    }

    std::vector<double> heatBalanceByVertex(curV, 0.0);
    for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
        Vertex sv = boost::source(edge, graph);
        Vertex tv = boost::target(edge, graph);
        const auto& eprop = graph[edge];

        const double Ts = tempsByVertex[static_cast<size_t>(sv)];
        const double Tt = tempsByVertex[static_cast<size_t>(tv)];
        const double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);

        graph[edge].heat_rate = Q;

        heatBalanceByVertex[static_cast<size_t>(sv)] -= Q;
        heatBalanceByVertex[static_cast<size_t>(tv)] += Q;
    }

    // --- エアコンによるバランス付け替え（vertex で処理） ---
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const auto& nodeData = graph[v];
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = g_topologyCache.airconSetVertex[static_cast<size_t>(v)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                heatBalanceByVertex[static_cast<size_t>(v)] = -heatBalanceByVertex[static_cast<size_t>(setV)];
                heatBalanceByVertex[static_cast<size_t>(setV)] = 0.0;
            }
        }
    }

    double maxBalance = 0.0;
    double rmseBalance = 0.0;
    for (size_t i = 0; i < g_topologyCache.parameterIndexToVertex.size(); ++i) {
        const Vertex v = g_topologyCache.parameterIndexToVertex[i];
        double balance = heatBalanceByVertex[static_cast<size_t>(v)];
        maxBalance = std::max(maxBalance, std::abs(balance));
        rmseBalance += balance * balance;
    }
    rmseBalance = std::sqrt(rmseBalance / static_cast<double>(nodeNames.size()));

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double seconds = duration.count() / 1000.0;
    {
        const bool ok = rmseBalance <= constants.thermalTolerance;
        std::ostringstream oss;
        // 既存ログフォーマット（熱計算(線形)）に合わせつつ、method に DirectT を明示する
        oss << "--------熱計算(線形): "
            << (ok ? "収束" : "未収束/バランス超過")
            << " (method=DirectT:" << methodLabel
            << ", RMSE=" << std::scientific << std::setprecision(6) << rmseBalance
            << ", maxBalance=" << std::scientific << std::setprecision(6) << maxBalance
            << ", tol=" << std::scientific << std::setprecision(6) << constants.thermalTolerance
            << ", time=" << std::fixed << std::setprecision(3) << seconds << "s)";
        writeLog(logFile, oss.str());
    }
}

} // namespace ThermalSolverLinearDirect


