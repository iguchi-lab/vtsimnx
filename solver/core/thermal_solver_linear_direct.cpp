#include "core/thermal_solver_linear_direct.h"
#include "core/heat_calculation.h"
#include "utils/utils.h"
#include "../network/thermal_network.h"

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

    struct RowIndexMap {
        int mask = 0;
        std::vector<int> keys;
        std::vector<int> values;

        void clear() { mask = 0; keys.clear(); values.clear(); }
        static int nextPow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }
        void buildFromCols(const std::vector<int>& cols) {
            const int tableSize = nextPow2(std::max(2, static_cast<int>(cols.size()) * 2));
            mask = tableSize - 1;
            keys.assign(static_cast<size_t>(tableSize), -1);
            values.assign(static_cast<size_t>(tableSize), -1);
            for (int local = 0; local < static_cast<int>(cols.size()); ++local) {
                const int col = cols[static_cast<size_t>(local)];
                uint32_t h = static_cast<uint32_t>(col) * 2654435761u;
                int idx = static_cast<int>(h) & mask;
                while (keys[static_cast<size_t>(idx)] != -1) idx = (idx + 1) & mask;
                keys[static_cast<size_t>(idx)] = col; values[static_cast<size_t>(idx)] = local;
            }
        }
        inline int get(int col) const {
            if (mask == 0) return -1;
            uint32_t h = static_cast<uint32_t>(col) * 2654435761u;
            int idx = static_cast<int>(h) & mask;
            while (keys[static_cast<size_t>(idx)] != -1) {
                if (keys[static_cast<size_t>(idx)] == col) return values[static_cast<size_t>(idx)];
                idx = (idx + 1) & mask;
            }
            return -1;
        }
    };

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

static TopologyCache g_topologyCache;

struct LinearSystem {
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    std::vector<std::vector<int>> colIndices;
    void initWithPattern(const std::vector<std::vector<int>>& rowColsPattern) {
        const size_t n = rowColsPattern.size();
        A.resize(n); b.assign(n, 0.0); colIndices = rowColsPattern;
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

static inline double responseArea(const EdgeProperties& ep) { return (ep.area > 0.0) ? ep.area : 1.0; }

static std::uint64_t fnv1a64_update(std::uint64_t h, std::uint64_t v) {
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    constexpr std::uint64_t kFnvPrime  = 1099511628211ull;
    if (h == 0) h = kFnvOffset;
    h ^= v;
    h *= kFnvPrime;
    return h;
}

static inline std::uint64_t hashDoubleBits(std::uint64_t h, double x) {
    std::uint64_t bits = 0;
    static_assert(sizeof(double) == sizeof(std::uint64_t), "double size mismatch");
    std::memcpy(&bits, &x, sizeof(bits));
    return fnv1a64_update(h, bits);
}

static inline bool isUnknown(const TopologyCache& topo, Vertex v) {
    const size_t idx = static_cast<size_t>(v);
    if (idx >= topo.vertexToParameterIndex.size()) return false;
    return topo.vertexToParameterIndex[idx] >= 0;
}

// A が変わったかを判定する軽量シグネチャ
// - advection の flow_rate（符号込み、閾値以下は 0 扱い）
// - aircon の on/off
// - set_node に紐づく fixed row の有無（airconBySetVertex の anyOn）
static std::uint64_t computeCoeffSignature(const Graph& graph, const TopologyCache& topo) {
    std::uint64_t h = 0;
    for (auto e : topo.advectionEdges) {
        const auto& ep = graph[e];
        Vertex sv = boost::source(e, graph);
        Vertex tv = boost::target(e, graph);
        h = fnv1a64_update(h,
                           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(sv)) << 32) ^
                               static_cast<std::uint64_t>(static_cast<std::uint32_t>(tv)));
        double flowRate = ep.flow_rate;
        if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) flowRate = 0.0;
        h = hashDoubleBits(h, flowRate);
        h = fnv1a64_update(h, ep.is_aircon_inflow ? 1u : 0u);
    }
    for (auto v : topo.airconVertices) {
        const auto& nd = graph[v];
        h = fnv1a64_update(h, static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)));
        h = fnv1a64_update(h, nd.on ? 1u : 0u);
    }
    for (size_t setV = 0; setV < topo.airconBySetVertex.size(); ++setV) {
        if (topo.airconBySetVertex[setV].empty()) continue;
        bool anyOn = false;
        for (Vertex v_ac : topo.airconBySetVertex[setV]) {
            if (graph[v_ac].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[v_ac].on) {
                anyOn = true;
                break;
            }
        }
        if (anyOn) {
            h = fnv1a64_update(h, static_cast<std::uint64_t>(setV));
            h = fnv1a64_update(h, 1u);
        }
    }
    return h;
}

static inline double evalResponseTempHistoryTermSrc(const EdgeProperties& ep) {
    double s = 0.0;
    for (size_t k = 1; k < ep.resp_a_src.size(); ++k) if (k-1 < ep.hist_t_src.size()) s += ep.resp_a_src[k] * ep.hist_t_src[k-1];
    for (size_t k = 1; k < ep.resp_b_src.size(); ++k) if (k-1 < ep.hist_t_tgt.size()) s += ep.resp_b_src[k] * ep.hist_t_tgt[k-1];
    return s;
}

static inline double evalResponseTempHistoryTermTgt(const EdgeProperties& ep) {
    double s = 0.0;
    for (size_t k = 1; k < ep.resp_a_tgt.size(); ++k) if (k-1 < ep.hist_t_tgt.size()) s += ep.resp_a_tgt[k] * ep.hist_t_tgt[k-1];
    for (size_t k = 1; k < ep.resp_b_tgt.size(); ++k) if (k-1 < ep.hist_t_src.size()) s += ep.resp_b_tgt[k] * ep.hist_t_src[k-1];
    return s;
}

static inline double evalResponseHistoryWattSrc(const EdgeProperties& ep) {
    double hW = responseArea(ep) * evalResponseTempHistoryTermSrc(ep);
    for (size_t k = 0; k < ep.resp_c_src.size(); ++k) if (k < ep.hist_q_src.size()) hW += ep.resp_c_src[k] * ep.hist_q_src[k];
    return hW;
}

static inline double evalResponseHistoryWattTgt(const EdgeProperties& ep) {
    double hW = responseArea(ep) * evalResponseTempHistoryTermTgt(ep);
    for (size_t k = 0; k < ep.resp_c_tgt.size(); ++k) if (k < ep.hist_q_tgt.size()) hW += ep.resp_c_tgt[k] * ep.hist_q_tgt[k];
    return hW;
}

static inline double evalResponseQSrc(const EdgeProperties& ep, double Ts, double Tt) {
    double q = responseArea(ep) * (ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0] * Ts + (ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0] * Tt) + evalResponseTempHistoryTermSrc(ep));
    for (size_t k = 0; k < ep.resp_c_src.size(); ++k) if (k < ep.hist_q_src.size()) q += ep.resp_c_src[k] * ep.hist_q_src[k];
    return q;
}

static inline double evalResponseQTgt(const EdgeProperties& ep, double Ts, double Tt) {
    double q = responseArea(ep) * (ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0] * Tt + (ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0] * Ts) + evalResponseTempHistoryTermTgt(ep));
    for (size_t k = 0; k < ep.resp_c_tgt.size(); ++k) if (k < ep.hist_q_tgt.size()) q += ep.resp_c_tgt[k] * ep.hist_q_tgt[k];
    return q;
}

static void buildLinearSystemAbsoluteFast(const Graph& graph, const TopologyCache& topo, LinearSystem& system) {
    const size_t n = topo.parameterIndexToVertex.size();
    system.resetValuesKeepPattern();
    std::vector<uint8_t> isFixedRow(n, 0); std::vector<double> fixedTemp(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Vertex v = topo.parameterIndexToVertex[i];
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(v)]) {
            if (graph[v_ac].on) { isFixedRow[i] = 1; fixedTemp[i] = graph[v_ac].current_pre_temp; break; }
        }
    }
    auto addCoeffOrKnownToB = [&](size_t row, const TopologyCache::RowIndexMap& rowMap, Vertex colVertex, double aCoeff) {
        int colIdx = topo.vertexToParameterIndex[static_cast<size_t>(colVertex)];
        if (colIdx >= 0) { int local = rowMap.get(colIdx); if (local >= 0) system.addCoefficientLocal(row, local, aCoeff); }
        else system.b[row] -= aCoeff * graph[colVertex].current_t;
    };
    auto processNodeNet = [&](size_t row, const TopologyCache::RowIndexMap& rowMap, Vertex v, double f) {
        for (auto edge : topo.incidentEdges[static_cast<size_t>(v)]) {
            Vertex sv = boost::source(edge, graph), tv = boost::target(edge, graph); const auto& ep = graph[edge];
            auto tc = ep.getTypeCode();
            if (tc == EdgeProperties::TypeCode::Conductance) {
                double k = ep.conductance;
                if (sv == v) { addCoeffOrKnownToB(row, rowMap, sv, f * (-k)); addCoeffOrKnownToB(row, rowMap, tv, f * (+k)); }
                else { addCoeffOrKnownToB(row, rowMap, sv, f * (+k)); addCoeffOrKnownToB(row, rowMap, tv, f * (-k)); }
            } else if (tc == EdgeProperties::TypeCode::Advection) {
                double flowRate = ep.flow_rate; if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) continue;
                double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
                if (flowRate > 0) { if (tv == v && !(ep.is_aircon_inflow && graph[tv].on)) { addCoeffOrKnownToB(row, rowMap, sv, f * (+mDotCp)); addCoeffOrKnownToB(row, rowMap, tv, f * (-mDotCp)); } }
                else { if (sv == v && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on)) { addCoeffOrKnownToB(row, rowMap, tv, f * (-mDotCp)); addCoeffOrKnownToB(row, rowMap, sv, f * (+mDotCp)); } }
            } else if (tc == EdgeProperties::TypeCode::HeatGeneration) {
                double q = ep.current_heat_generation; if (q != 0.0) system.b[row] += (sv == v ? f * (+q) : f * (-q));
            } else if (tc == EdgeProperties::TypeCode::ResponseConduction) {
                if (sv == v) {
                    double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0], b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0], area = responseArea(ep);
                    addCoeffOrKnownToB(row, rowMap, sv, f * (-area * a0)); addCoeffOrKnownToB(row, rowMap, tv, f * (-area * b0));
                    system.b[row] += f * (+evalResponseHistoryWattSrc(ep));
                } else {
                    double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0], b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0], area = responseArea(ep);
                    addCoeffOrKnownToB(row, rowMap, tv, f * (-area * a0)); addCoeffOrKnownToB(row, rowMap, sv, f * (-area * b0));
                    system.b[row] += f * (+evalResponseHistoryWattTgt(ep));
                }
            }
        }
    };
    for (size_t i = 0; i < n; ++i) {
        if (isFixedRow[i]) { system.b[i] = fixedTemp[i]; int local = topo.rowIndexMaps[i].get(static_cast<int>(i)); if (local >= 0) system.A[i][static_cast<size_t>(local)] = 1.0; continue; }
        Vertex v = topo.parameterIndexToVertex[i]; system.b[i] += graph[v].heat_source;
        if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[v].on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(v)]; if (setV != std::numeric_limits<Vertex>::max()) processNodeNet(i, topo.rowIndexMaps[i], setV, 1.0);
            else processNodeNet(i, topo.rowIndexMaps[i], v, 1.0);
        } else processNodeNet(i, topo.rowIndexMaps[i], v, 1.0);
    }
}

static void rebuildRhsPrecomputeForCoeffSig(const Graph& graph, TopologyCache& topo, std::uint64_t coeffSig) {
    const size_t n = topo.parameterIndexToVertex.size();
    topo.knownTermsByRow.assign(n, {});
    topo.heatGenByRow.assign(n, {});
    topo.responseHistByRow.assign(n, {});
    topo.fixedRowAirconVertex.assign(n, std::numeric_limits<Vertex>::max());

    auto addKnown = [&](size_t row, Vertex v, double coeff) {
        if (std::abs(coeff) < 1e-15) return;
        if (isUnknown(topo, v)) return;
        topo.knownTermsByRow[row].push_back(TopologyCache::KnownTerm{v, coeff});
    };
    auto addHeatGen = [&](size_t row, Edge e, double sign) {
        if (std::abs(sign) < 1e-15) return;
        topo.heatGenByRow[row].push_back(TopologyCache::HeatGenTerm{e, sign});
    };
    auto addRespHist = [&](size_t row, Edge e, bool isSrc, double factor) {
        topo.responseHistByRow[row].push_back(TopologyCache::ResponseHistTerm{e, isSrc, factor});
    };

    for (size_t i = 0; i < n; ++i) {
        Vertex rowV = topo.parameterIndexToVertex[i];

        // 固定温度行（set_node の aircon が ON）
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(rowV)]) {
            if (graph[v_ac].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[v_ac].on) {
                topo.fixedRowAirconVertex[i] = v_ac;
                break;
            }
        }
        if (topo.fixedRowAirconVertex[i] != std::numeric_limits<Vertex>::max()) {
            continue;
        }

        // 行のネットワーク参照頂点（aircon on の場合は set_node 側を見る）
        Vertex procV = rowV;
        if (graph[rowV].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[rowV].on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(rowV)];
            if (setV != std::numeric_limits<Vertex>::max()) procV = setV;
        }

        for (auto edge : topo.incidentEdges[static_cast<size_t>(procV)]) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& ep = graph[edge];
            const auto tc = ep.getTypeCode();

            if (tc == EdgeProperties::TypeCode::Conductance) {
                const double k = ep.conductance;
                if (sv == procV) {
                    addKnown(i, sv, -k);
                    addKnown(i, tv, +k);
                } else {
                    addKnown(i, sv, +k);
                    addKnown(i, tv, -k);
                }
            } else if (tc == EdgeProperties::TypeCode::Advection) {
                double flowRate = ep.flow_rate;
                if (std::abs(flowRate) < archenv::FLOW_RATE_MIN) continue;
                const double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
                if (flowRate > 0) {
                    if (tv == procV && !(ep.is_aircon_inflow && graph[tv].on)) {
                        addKnown(i, sv, +mDotCp);
                        addKnown(i, tv, -mDotCp);
                    }
                } else {
                    if (sv == procV && !(graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on)) {
                        addKnown(i, tv, -mDotCp);
                        addKnown(i, sv, +mDotCp);
                    }
                }
            } else if (tc == EdgeProperties::TypeCode::HeatGeneration) {
                if (sv == procV) addHeatGen(i, edge, +1.0);
                else addHeatGen(i, edge, -1.0);
            } else if (tc == EdgeProperties::TypeCode::ResponseConduction) {
                const double area = responseArea(ep);
                if (sv == procV) {
                    const double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0];
                    const double b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0];
                    addKnown(i, sv, -area * a0);
                    addKnown(i, tv, -area * b0);
                    addRespHist(i, edge, true, 1.0);
                } else {
                    const double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0];
                    const double b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0];
                    addKnown(i, tv, -area * a0);
                    addKnown(i, sv, -area * b0);
                    addRespHist(i, edge, false, 1.0);
                }
            }
        }

        // known term をまとめる（行長が小さい前提）
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

static void buildRhsOnlyAbsoluteFast(const Graph& graph, const TopologyCache& topo, std::vector<double>& bOut) {
    const size_t n = topo.parameterIndexToVertex.size();
    bOut.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const Vertex v_ac = (i < topo.fixedRowAirconVertex.size())
                                ? topo.fixedRowAirconVertex[i]
                                : std::numeric_limits<Vertex>::max();
        if (v_ac != std::numeric_limits<Vertex>::max()) {
            bOut[i] = graph[v_ac].current_pre_temp;
            continue;
        }

        const Vertex rowV = topo.parameterIndexToVertex[i];
        bOut[i] += graph[rowV].heat_source;

        if (i < topo.knownTermsByRow.size()) {
            for (const auto& t : topo.knownTermsByRow[i]) {
                if (t.v == std::numeric_limits<Vertex>::max()) continue;
                if (std::abs(t.coeff) < 1e-15) continue;
                bOut[i] -= t.coeff * graph[t.v].current_t;
            }
        }
        if (i < topo.heatGenByRow.size()) {
            for (const auto& tg : topo.heatGenByRow[i]) {
                bOut[i] += tg.sign * graph[tg.e].current_heat_generation;
            }
        }
        if (i < topo.responseHistByRow.size()) {
            for (const auto& rh : topo.responseHistByRow[i]) {
                const auto& ep = graph[rh.e];
                const double hW = rh.isSrc ? evalResponseHistoryWattSrc(ep) : evalResponseHistoryWattTgt(ep);
                bOut[i] += rh.factor * (+hW);
            }
        }
    }
}

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

static SparseLUCache g_sparseLuCache;

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

static SparseCholeskyCache g_cholCache;

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

static DirectTStats g_directTStats;
static std::uint64_t s_lastCoeffSig = 0;

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

static bool solveSparseDirect(const LinearSystem& system,
                             std::vector<double>& x,
                             double tolerance,
                             std::ostream& logFile,
                             std::string& methodLabel) {
    const size_t n = x.size();
    if (n == 0) return true;

    size_t nnz = 0;
    for (size_t i = 0; i < n; ++i) nnz += system.colIndices[i].size();

    std::uint64_t patternHash = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto& cols = system.colIndices[i];
        for (size_t k = 0; k < cols.size(); ++k) {
            patternHash = fnv1a64_update(patternHash,
                                         (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(cols[k]));
        }
    }

    Eigen::VectorXd b(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) b[static_cast<int>(i)] = system.b[i];

    const bool needRebuildPattern = (!g_sparseLuCache.analyzed) ||
                                   (g_sparseLuCache.n != static_cast<int>(n)) ||
                                   (g_sparseLuCache.nnz != nnz) ||
                                   (g_sparseLuCache.patternHash != patternHash);

    if (needRebuildPattern) {
        ++g_directTStats.patternRebuild;
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
        std::uint64_t valueHash = 0;
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

        // valuePtr mapping (row-wise)
        g_sparseLuCache.valuePtrIndexByRow.assign(n, {});
        std::vector<std::vector<std::pair<int, int>>> rowEntries(n);
        for (size_t r = 0; r < n; ++r) rowEntries[r].reserve(system.colIndices[r].size());
        double* base = g_sparseLuCache.A.valuePtr();
        for (int outer = 0; outer < g_sparseLuCache.A.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(g_sparseLuCache.A, outer); it; ++it) {
                const int r = it.row();
                const int c = it.col();
                const int p = static_cast<int>(&it.valueRef() - base);
                if (r >= 0 && r < static_cast<int>(n)) rowEntries[static_cast<size_t>(r)].emplace_back(c, p);
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
                if (j < entries.size() && entries[j].first == col) g_sparseLuCache.valuePtrIndexByRow[r][k] = entries[j].second;
            }
        }
        bool mappingOk = true;
        for (size_t r = 0; r < n && mappingOk; ++r) {
            for (int p : g_sparseLuCache.valuePtrIndexByRow[r]) {
                if (p < 0) { mappingOk = false; break; }
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
        std::uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& vals = system.A[i];
            for (size_t k = 0; k < vals.size(); ++k) {
                const int p = g_sparseLuCache.valuePtrIndexByRow[i][k];
                if (p < 0) return false;
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
            ++g_directTStats.cholFactorize;
            g_cholCache.llt.factorize(g_sparseLuCache.A);
            if (g_cholCache.llt.info() == Eigen::Success) {
                g_cholCache.factorized = true;
                g_cholCache.valueHash = g_sparseLuCache.valueHash;
                sol = g_cholCache.llt.solve(b);
                if (g_cholCache.llt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "LLT";
                }
            }
            if (!solved) {
                ++g_directTStats.cholFactorize;
                g_cholCache.ldlt.factorize(g_sparseLuCache.A);
                if (g_cholCache.ldlt.info() == Eigen::Success) {
                    g_cholCache.factorized = true;
                    g_cholCache.valueHash = g_sparseLuCache.valueHash;
                    sol = g_cholCache.ldlt.solve(b);
                    if (g_cholCache.ldlt.info() == Eigen::Success) {
                        solved = true;
                        methodLabel = "LDLT";
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
                methodLabel = "LLT(cached)";
            } else {
                sol = g_cholCache.ldlt.solve(b);
                if (g_cholCache.ldlt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "LDLT(cached)";
                }
            }
        }
    }

    if (!solved) {
        if (!g_sparseLuCache.factorized) {
            ++g_directTStats.luFactorize;
            g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
            if (g_sparseLuCache.solver.info() != Eigen::Success) return false;
            g_sparseLuCache.factorized = true;
        }
        sol = g_sparseLuCache.solver.solve(b);
        if (g_sparseLuCache.solver.info() != Eigen::Success) return false;
        methodLabel = "LU";
    }

    Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
    double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
    if (!std::isfinite(maxResidual)) return false;
    if (maxResidual > tolerance * 10.0) return false;

    for (size_t i = 0; i < n; ++i) {
        const double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) return false;
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
    static std::uint64_t s_cachedResidualCheckCounter = 0;

    if (g_cholCache.analyzed && g_cholCache.factorized && g_cholCache.patternSymmetric) {
        sol = g_cholCache.llt.solve(b);
        if (g_cholCache.llt.info() == Eigen::Success) {
            ok = true;
            methodLabel = "LLT(cached)";
        } else {
            sol = g_cholCache.ldlt.solve(b);
            if (g_cholCache.ldlt.info() == Eigen::Success) {
                ok = true;
                methodLabel = "LDLT(cached)";
            }
        }
    }
    if (!ok && g_sparseLuCache.analyzed && g_sparseLuCache.factorized) {
        sol = g_sparseLuCache.solver.solve(b);
        if (g_sparseLuCache.solver.info() == Eigen::Success) {
            ok = true;
            methodLabel = "LU(cached)";
        }
    }
    if (!ok) return false;

    constexpr std::uint64_t kCachedResidualCheckInterval = 200;
    if ((s_cachedResidualCheckCounter++ % kCachedResidualCheckInterval) == 0) {
        Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
        double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
        if (!std::isfinite(maxResidual)) return false;
        if (maxResidual > tolerance * 10.0) return false;
    }

    for (size_t i = 0; i < n; ++i) {
        const double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) return false;
        x[i] = v;
    }
    (void)logFile; // keep signature; log throttled by caller
    return true;
}

} // namespace

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
        g_topologyCache = TopologyCache{};
        g_topologyCache.graphPtr = &graph;
        g_topologyCache.numVertices = curV;
        g_topologyCache.numEdges = curE;

        g_topologyCache.incidentEdges.assign(curV, {});
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            g_topologyCache.incidentEdges[static_cast<size_t>(boost::source(e, graph))].push_back(e);
            g_topologyCache.incidentEdges[static_cast<size_t>(boost::target(e, graph))].push_back(e);
        }

        g_topologyCache.airconBySetVertex.assign(curV, {});
        g_topologyCache.airconSetVertex.assign(curV, std::numeric_limits<Vertex>::max());
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon && !graph[v].set_node.empty()) {
                auto it = network.getKeyToVertex().find(graph[v].set_node);
                if (it != network.getKeyToVertex().end()) {
                    g_topologyCache.airconBySetVertex[static_cast<size_t>(it->second)].push_back(v);
                    g_topologyCache.airconSetVertex[static_cast<size_t>(v)] = it->second;
                }
            }
        }

        g_topologyCache.advectionEdges.clear();
        g_topologyCache.responseEdges.clear();
        g_topologyCache.airconVertices.clear();
        for (auto e : boost::make_iterator_range(boost::edges(graph))) {
            auto tc = graph[e].getTypeCode();
            if (tc == EdgeProperties::TypeCode::Advection) g_topologyCache.advectionEdges.push_back(e);
            else if (tc == EdgeProperties::TypeCode::ResponseConduction) g_topologyCache.responseEdges.push_back(e);
        }
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            if (graph[v].getTypeCode() == VertexProperties::TypeCode::Aircon) g_topologyCache.airconVertices.push_back(v);
        }

        g_topologyCache.nodeNames.clear();
        g_topologyCache.vertexToParameterIndex.assign(curV, -1);
        g_topologyCache.parameterIndexToVertex.clear();
        size_t pIdx = 0;
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            if (graph[v].calc_t) {
                g_topologyCache.nodeNames.push_back(graph[v].key);
                g_topologyCache.vertexToParameterIndex[static_cast<size_t>(v)] = static_cast<int>(pIdx++);
                g_topologyCache.parameterIndexToVertex.push_back(v);
            }
        }

        g_topologyCache.rowColsPattern.assign(g_topologyCache.nodeNames.size(), {});
        for (size_t r = 0; r < g_topologyCache.nodeNames.size(); ++r) {
            Vertex v = g_topologyCache.parameterIndexToVertex[r];
            std::vector<int> cols = {static_cast<int>(r)};
            auto addV = [&](Vertex vv) {
                int idx = g_topologyCache.vertexToParameterIndex[static_cast<size_t>(vv)];
                if (idx >= 0) cols.push_back(idx);
            };
            for (auto e : g_topologyCache.incidentEdges[static_cast<size_t>(v)]) { addV(boost::source(e, graph)); addV(boost::target(e, graph)); }
            Vertex setV = g_topologyCache.airconSetVertex[static_cast<size_t>(v)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                for (auto e : g_topologyCache.incidentEdges[static_cast<size_t>(setV)]) { addV(boost::source(e, graph)); addV(boost::target(e, graph)); }
            }
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            g_topologyCache.rowColsPattern[r] = std::move(cols);
        }
        g_topologyCache.rowIndexMaps.assign(g_topologyCache.rowColsPattern.size(), {});
        for (size_t r = 0; r < g_topologyCache.rowColsPattern.size(); ++r) g_topologyCache.rowIndexMaps[r].buildFromCols(g_topologyCache.rowColsPattern[r]);

        g_topologyCache.rhsCoeffSig = 0;
        g_topologyCache.initialized = true;
    }

    const size_t n = g_topologyCache.nodeNames.size();
    if (n == 0) return;

    static LinearSystem system;
    static size_t systemN = 0;
    static const Graph* systemGraphPtr = nullptr;
    if (needRebuildTopo || systemGraphPtr != &graph || systemN != n || system.colIndices.size() != n) {
        system.initWithPattern(g_topologyCache.rowColsPattern);
        systemN = n;
        systemGraphPtr = &graph;
    }

    const std::uint64_t coeffSig = computeCoeffSignature(graph, g_topologyCache);
    if (g_directTStats.calls > 1 && s_lastCoeffSig != 0 && coeffSig != s_lastCoeffSig) {
        ++g_directTStats.coeffSigChanged;
    }
    s_lastCoeffSig = coeffSig;
    if (g_topologyCache.rhsCoeffSig != coeffSig ||
        g_topologyCache.fixedRowAirconVertex.size() != n ||
        g_topologyCache.knownTermsByRow.size() != n ||
        g_topologyCache.responseHistByRow.size() != n) {
        ++g_directTStats.rhsPrecomputeRebuild;
        rebuildRhsPrecomputeForCoeffSig(graph, g_topologyCache, coeffSig);
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
        buildRhsOnlyAbsoluteFast(graph, g_topologyCache, system.b);
    } else {
        ++g_directTStats.fullBuild;
        buildLinearSystemAbsoluteFast(graph, g_topologyCache, system);
    }

    std::vector<double> temperatures(n, 0.0);
    bool solved = false;
    std::string method = "LLT";
    if (canReuseFactorization) {
        ++g_directTStats.solveCached;
        Eigen::VectorXd eb(static_cast<int>(n));
        for (size_t i = 0; i < n; ++i) eb[static_cast<int>(i)] = system.b[i];
        solved = solveWithCachedFactorization(eb, temperatures, constants.thermalTolerance, logFile, method);
    } else {
        ++g_directTStats.solveFull;
        solved = solveSparseDirect(system, temperatures, constants.thermalTolerance, logFile, method);
        if (solved) g_sparseLuCache.coeffSig = coeffSig;
    }

    if (!solved) {
        throw std::runtime_error("thermal solve failed (direct absolute T solver)");
    }

    for (size_t i = 0; i < n; ++i) graph[g_topologyCache.parameterIndexToVertex[i]].current_t = temperatures[i];
    
    std::vector<double> heatBalance(curV, 0.0);
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        Vertex sv = boost::source(e, graph), tv = boost::target(e, graph); auto& ep = graph[e];
        double Ts = graph[sv].current_t, Tt = graph[tv].current_t; auto tc = ep.getTypeCode();
        if (tc == EdgeProperties::TypeCode::ResponseConduction) {
            double qs = evalResponseQSrc(ep, Ts, Tt), qt = evalResponseQTgt(ep, Ts, Tt); heatBalance[static_cast<size_t>(sv)] -= qs; heatBalance[static_cast<size_t>(tv)] -= qt; ep.heat_rate = (qs + qt) / 2.0;
        } else if (tc == EdgeProperties::TypeCode::Advection) {
            double Q = HeatCalculation::calcAdvectionHeat(Ts, Tt, ep);
            if (ep.flow_rate > 0) { if (ep.is_aircon_inflow && graph[tv].on) Q = 0.0; heatBalance[static_cast<size_t>(tv)] += Q; }
            else { if (graph[sv].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[sv].on) Q = 0.0; heatBalance[static_cast<size_t>(sv)] -= Q; }
            ep.heat_rate = Q;
        } else {
            double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep); ep.heat_rate = Q; heatBalance[static_cast<size_t>(sv)] -= Q; heatBalance[static_cast<size_t>(tv)] += Q;
        }
    }
    for (size_t i = 0; i < curV; ++i) {
        heatBalance[i] += graph[i].heat_source;
        if (graph[i].getTypeCode() == VertexProperties::TypeCode::Aircon && graph[i].on) {
            Vertex setV = g_topologyCache.airconSetVertex[i]; if (setV != std::numeric_limits<Vertex>::max()) { heatBalance[i] = heatBalance[static_cast<size_t>(setV)]; heatBalance[static_cast<size_t>(setV)] = 0.0; }
        }
    }
    double maxB = 0.0, rmseB = 0.0; for (auto v : g_topologyCache.parameterIndexToVertex) { double b = heatBalance[static_cast<size_t>(v)]; maxB = std::max(maxB, std::abs(b)); rmseB += b * b; }
    rmseB = std::sqrt(rmseB / n); auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime);
    std::ostringstream oss; oss << "--------熱計算(線形): " << (rmseB <= constants.thermalTolerance ? "収束" : "未収束") << " (method=" << method << ", RMSE=" << std::scientific << std::setprecision(6) << rmseB << ", maxBalance=" << maxB << ", time=" << std::fixed << std::setprecision(3) << dur.count()/1000.0 << "s)";
    writeLog(logFile, oss.str()); network.setLastThermalConvergence(rmseB <= constants.thermalTolerance, rmseB, maxB, method);

    constexpr std::uint64_t kStatsLogInterval = 500;
    if ((g_directTStats.calls % kStatsLogInterval) == 0) {
        std::ostringstream ss;
        ss << "--------DirectT cache stats: calls=" << g_directTStats.calls
           << ", coeffSigChanged=" << g_directTStats.coeffSigChanged
           << ", missNotAnalyzed=" << g_directTStats.reuseMissNotAnalyzed
           << ", missNoFactorized=" << g_directTStats.reuseMissNoFactorized
           << ", missSizeMismatch=" << g_directTStats.reuseMissSizeMismatch
           << ", missCoeffSigMismatch=" << g_directTStats.reuseMissCoeffSigMismatch
           << ", topoRebuild=" << g_directTStats.topoRebuild
           << ", rhsPrecomputeRebuild=" << g_directTStats.rhsPrecomputeRebuild
           << ", rhsOnlyBuild=" << g_directTStats.rhsOnlyBuild
           << ", fullBuild=" << g_directTStats.fullBuild
           << ", patternRebuild=" << g_directTStats.patternRebuild
           << ", solveCached=" << g_directTStats.solveCached
           << ", solveFull=" << g_directTStats.solveFull
           << ", cholFactorize=" << g_directTStats.cholFactorize
           << ", luFactorize=" << g_directTStats.luFactorize;
        writeLog(logFile, ss.str());
    }
}

} // namespace ThermalSolverLinearDirect
