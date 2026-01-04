#include "core/thermal_solver_linear_gs.h"
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

namespace ThermalSolverLinearGS {

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
    // 行の列数は小さい（典型 10〜30）ため、小さなオープンアドレス法で O(1) 参照する。
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
                // 32bit mix → table index
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

    std::vector<RowIndexMap> rowIndexMaps;             // [row] -> lookup table

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
    // system.colIndices と同型の「A.valuePtr() へのインデックス」
    // valuePtrIndexByRow[row][k] は (row, system.colIndices[row][k]) の valuePtr 位置。
    // これにより値更新はハッシュ無しの O(nnz) 直書きになる。
    std::vector<std::vector<int>> valuePtrIndexByRow;
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
    // 「非ゼロパターンが対称か」（A(i,j) があるなら A(j,i) もある）だけを保持。
    // 値の対称性チェックは毎回 O(nnz) になるので行わず、Cholesky を試して残差で検証する。
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
    // 非ゼロパターンが対称かどうかだけをチェック（値は見ない）。
    // O(nnz * log(rowLen)) だが、パターン再構築時に 1 回だけ実行する。
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

// A の係数値が変わったかを判定するための軽量シグネチャを作る。
// 温度に依存しない係数（conductance と |flow_rate|）と、固定行（set_node の aircon on）だけを見る。
static uint64_t computeCoeffSignature(const Graph& graph,
                                      const TopologyCache& topo) {
    uint64_t h = 0;

    // advection の係数（|flow_rate|）は時刻で変わり得るため、該当辺のみ走査してハッシュ化
    for (auto e : topo.advectionEdges) {
        const auto& ep = graph[e];
        double flowRate = ep.flow_rate;
        double coeff = 0.0;
        if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
            coeff = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
        }
        // 微小な風量変動での再計算（35ms）を避けるため、係数を丸めてハッシュ化する
        double roundedCoeff = std::round(coeff * 100.0) / 100.0;
        h = hashDoubleBits(h, roundedCoeff);
    }

    // 固定温度行（set_node に紐づく aircon が on）の有無
    for (size_t setV = 0; setV < topo.airconBySetVertex.size(); ++setV) {
        if (topo.airconBySetVertex[setV].empty()) continue;
        bool anyOn = false;
        for (Vertex v_ac : topo.airconBySetVertex[setV]) {
            if (graph[v_ac].on) {
                anyOn = true;
                break;
            }
        }
        if (anyOn) {
            h = fnv1a64_update(h, static_cast<uint64_t>(setV));
            h = fnv1a64_update(h, 1u);
        }
    }

    // aircon shoulder の on/off
    for (auto v : topo.airconVertices) {
        h = fnv1a64_update(h, static_cast<uint64_t>(static_cast<uint32_t>(v)));
        h = fnv1a64_update(h, graph[v].on ? 1u : 0u);
    }

    return h;
}

// b（右辺）だけを構築（A は作らない）
static void buildRhsOnly(const Graph& graph,
                         const TopologyCache& topo,
                         const std::vector<double>* tempsOverride,
                         std::vector<double>& bOut) {
    const size_t n = topo.nodeNames.size();
    bOut.assign(n, 0.0);

    auto getTemp = [&](Vertex v) -> double {
        if (tempsOverride) {
            int idx = topo.vertexToParameterIndex[static_cast<size_t>(v)];
            if (idx >= 0 && static_cast<size_t>(idx) < tempsOverride->size()) return (*tempsOverride)[static_cast<size_t>(idx)];
        }
        return graph[v].current_t;
    };

    // まず「固定温度行（set_node の aircon が ON）」を特定して b を埋める。
    // 係数行列 A はキャッシュ前提なので、固定行の集合が変わるケースではそもそもこの経路に来ない。
    std::vector<uint8_t> isFixedRow(n, 0);
    for (size_t i = 0; i < n; ++i) {
        Vertex nodeVertex = topo.parameterIndexToVertex[i];
        for (auto v_ac : topo.airconBySetVertex[static_cast<size_t>(nodeVertex)]) {
            const auto& nd = graph[v_ac];
            if (nd.getTypeCode() == VertexProperties::TypeCode::Aircon && nd.on) {
                isFixedRow[i] = 1;
                bOut[i] = nd.current_pre_temp - getTemp(nodeVertex);
                break;
            }
        }
    }

    // 次に、全エッジを 1 回だけ走査して各頂点の熱バランス残差（inflow - outflow）を作る。
    // 旧実装は「各ノードで incidentEdges を辿る」ため、実質 2*E 回の熱計算をしていた。
    const size_t curV = static_cast<size_t>(boost::num_vertices(graph));
    std::vector<double> residualByVertex(curV, 0.0); // [vertex] -> inflow - outflow
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        Vertex sv = boost::source(e, graph);
        Vertex tv = boost::target(e, graph);
        const auto& ep = graph[e];

        const double Ts = getTemp(sv);
        const double Tt = getTemp(tv);
        const double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep); // src -> dst を正

        // ノード残差（片方向移流モデルを考慮）
        if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) {
            if (ep.flow_rate > 0) { // sv -> tv
                residualByVertex[static_cast<size_t>(tv)] += Q;
            } else if (ep.flow_rate < 0) { // tv -> sv
                residualByVertex[static_cast<size_t>(sv)] -= Q;
            }
        } else {
            residualByVertex[static_cast<size_t>(sv)] -= Q; // outflow
            residualByVertex[static_cast<size_t>(tv)] += Q; // inflow
        }
    }

    // 最後に、変数ノード（calc_t）について b を作る。
    // A案: aircon ON のとき、aircon 行は set_node の残差を肩代わりする（set は固定温度行）
    for (size_t i = 0; i < n; ++i) {
        if (isFixedRow[i]) continue;
        Vertex nodeVertex = topo.parameterIndexToVertex[i];
        double residual = residualByVertex[static_cast<size_t>(nodeVertex)];

        const auto& nodeData = graph[nodeVertex];
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = topo.airconSetVertex[static_cast<size_t>(nodeVertex)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                residual = residualByVertex[static_cast<size_t>(setV)];
            }
        }

        bOut[i] = -residual;
    }
}

// 線形システム Ax = b を構築
struct LinearSystem {
    std::vector<std::vector<double>> A;  // 係数行列（疎行列として実装）
    std::vector<double> b;               // 右辺ベクトル
    std::vector<std::vector<int>> colIndices; // 各行の非ゼロ列インデックス
    
    void resize(size_t n) {
        A.resize(n);
        b.resize(n, 0.0);
        colIndices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            A[i].clear();
            colIndices[i].clear();
        }
    }

    // トポロジ固定の非ゼロパターンで初期化（colIndicesは固定、Aはゼロで埋める）
    void initWithPattern(const std::vector<std::vector<int>>& rowColsPattern) {
        const size_t n = rowColsPattern.size();
        A.resize(n);
        b.assign(n, 0.0);
        colIndices = rowColsPattern;
        for (size_t i = 0; i < n; ++i) {
            A[i].assign(colIndices[i].size(), 0.0);
        }
    }

    // パターンを保持したまま値だけリセット
    void resetValuesKeepPattern() {
        std::fill(b.begin(), b.end(), 0.0);
        for (auto& rowA : A) {
            std::fill(rowA.begin(), rowA.end(), 0.0);
        }
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

// 線形システムを構築
void buildLinearSystem(
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndex,
    const std::vector<Vertex>& parameterIndexToVertex,
    const std::vector<std::vector<Edge>>& incidentEdges,
    const std::vector<std::vector<Vertex>>& airconBySetVertex,
    const std::vector<Vertex>& airconSetVertex,
    const std::vector<TopologyCache::RowIndexMap>& rowIndexMaps,
    const std::vector<double>* tempsOverride,
    LinearSystem& system) {
    
    size_t n = parameterIndexToVertex.size();
    // 既に固定パターンで初期化されている場合は、それを維持して値のみリセットする。
    // （ここで resize すると colIndices が空になり、係数が一切入らずに破綻する）
    if (system.colIndices.size() == n && system.A.size() == n && system.b.size() == n &&
        (n == 0 || !system.colIndices[0].empty())) {
        system.resetValuesKeepPattern();
    } else {
        system.resize(n);
    }

    auto getTemp = [&](Vertex v) -> double {
        if (tempsOverride) {
            int idx = vertexToParameterIndex[static_cast<size_t>(v)];
            if (idx >= 0 && static_cast<size_t>(idx) < tempsOverride->size()) return (*tempsOverride)[static_cast<size_t>(idx)];
        }
        return graph[v].current_t;
    };
    
    for (size_t i = 0; i < n; ++i) {
        Vertex nodeVertex = parameterIndexToVertex[i];
        const auto& nodeData = graph[nodeVertex];
        const auto& rowMap = rowIndexMaps[i];
        auto addCoeffFast = [&](size_t row, int colParamIdx, double v) {
            const int local = rowMap.get(colParamIdx);
            if (local >= 0) {
                system.addCoefficientLocal(row, local, v);
            }
        };
        
        // エアコンのset_nodeがONの場合は固定温度（制約として扱う）
        bool isFixed = false;
        for (auto v_ac : airconBySetVertex[static_cast<size_t>(nodeVertex)]) {
            const auto& nd = graph[v_ac];
            if (nd.getTypeCode() == VertexProperties::TypeCode::Aircon && nd.on) {
                isFixed = true;
                // 固定温度: T[i] = current_pre_temp（設定温度）
                addCoeffFast(i, static_cast<int>(i), 1.0);
                // 未知数は ΔT を解いて最後に T += ΔT するため、b は (set - current)
                system.b[i] = nd.current_pre_temp - getTemp(nodeVertex);
                break;
            }
        }
        
        if (isFixed) continue;
        
        // 熱バランス方程式: Σ(inflow) - Σ(outflow) = 0
        // A案: aircon ON のとき、この行は set_node の収支=0 を表す（set は固定温度行）
        Vertex balanceVertex = nodeVertex;
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = airconSetVertex[static_cast<size_t>(nodeVertex)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                balanceVertex = setV;
            }
        }
        double inflow = 0.0;
        double outflow = 0.0;
        
        for (auto edge : incidentEdges[static_cast<size_t>(balanceVertex)]) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& eprop = graph[edge];
            
            int sIdx = vertexToParameterIndex[static_cast<size_t>(sv)];
            int tIdx = vertexToParameterIndex[static_cast<size_t>(tv)];
            bool sIsVariable = (sIdx >= 0);
            bool tIsVariable = (tIdx >= 0);
            
            double dQdTs = 0.0, dQdTt = 0.0;
            
            double Ts = getTemp(sv);
            double Tt = getTemp(tv);
            double Q = 0.0;

            if (eprop.getTypeCode() == EdgeProperties::TypeCode::ResponseConduction) {
                // 応答係数: 両端で別の熱流を持つため、ノード側に応じて Q を切り替える
                if (sv == nodeVertex) {
                    const double a0 = eprop.resp_a_src.empty() ? 0.0 : eprop.resp_a_src[0];
                    const double b0 = eprop.resp_b_src.empty() ? 0.0 : eprop.resp_b_src[0];
                    dQdTs = a0;
                    dQdTt = b0;

                    // Q = a0*Ts + b0*Tt + history
                    Q = a0 * Ts + b0 * Tt;
                    for (size_t k = 1; k < eprop.resp_a_src.size(); ++k) {
                        const size_t idx = k - 1;
                        if (idx < eprop.hist_t_src.size()) Q += eprop.resp_a_src[k] * eprop.hist_t_src[idx];
                    }
                    for (size_t k = 1; k < eprop.resp_b_src.size(); ++k) {
                        const size_t idx = k - 1;
                        if (idx < eprop.hist_t_tgt.size()) Q += eprop.resp_b_src[k] * eprop.hist_t_tgt[idx];
                    }
                    for (size_t k = 0; k < eprop.resp_c_src.size(); ++k) {
                        if (k < eprop.hist_q_src.size()) Q += eprop.resp_c_src[k] * eprop.hist_q_src[k];
                    }
                } else if (tv == nodeVertex) {
                    const double a0 = eprop.resp_a_tgt.empty() ? 0.0 : eprop.resp_a_tgt[0];
                    const double b0 = eprop.resp_b_tgt.empty() ? 0.0 : eprop.resp_b_tgt[0];
                    // q_tgt(n) = a0*Tt + b0*Ts + ...
                    dQdTs = b0;
                    dQdTt = a0;

                    Q = a0 * Tt + b0 * Ts;
                    for (size_t k = 1; k < eprop.resp_a_tgt.size(); ++k) {
                        const size_t idx = k - 1;
                        if (idx < eprop.hist_t_tgt.size()) Q += eprop.resp_a_tgt[k] * eprop.hist_t_tgt[idx];
                    }
                    for (size_t k = 1; k < eprop.resp_b_tgt.size(); ++k) {
                        const size_t idx = k - 1;
                        if (idx < eprop.hist_t_src.size()) Q += eprop.resp_b_tgt[k] * eprop.hist_t_src[idx];
                    }
                    for (size_t k = 0; k < eprop.resp_c_tgt.size(); ++k) {
                        if (k < eprop.hist_q_tgt.size()) Q += eprop.resp_c_tgt[k] * eprop.hist_q_tgt[k];
                    }
                }
            } else {
                if (eprop.getTypeCode() == EdgeProperties::TypeCode::Conductance) {
                    dQdTs = eprop.conductance;
                    dQdTt = -eprop.conductance;
                } else if (eprop.getTypeCode() == EdgeProperties::TypeCode::Advection) {
                    double flowRate = eprop.flow_rate;
                    if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
                        double mDotCp = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * flowRate;
                        // 片方向移流モデル：空気の流入先（ターゲット）ノードのみに熱収支を適用する
                        if (flowRate > 0) { // sv -> tv
                            if (tv == balanceVertex) {
                                // 旧バージョン互換: エアコンノードへの流入熱量（還気）は、エアコンON時のみ0とする
                                if (!(eprop.is_aircon_inflow && graph[tv].on)) {
                                    dQdTs = mDotCp;
                                    dQdTt = -mDotCp;
                                } else {
                                    dQdTs = 0.0; dQdTt = 0.0;
                                }
                            } else {
                                dQdTs = 0.0; dQdTt = 0.0;
                            }
                        } else { // tv -> sv
                            if (sv == balanceVertex) {
                                // エアコン還気判定（逆流時）
                                const auto& srcNode = graph[sv];
                                if (!(srcNode.getTypeCode() == VertexProperties::TypeCode::Aircon && srcNode.on)) {
                                    dQdTs = -mDotCp;
                                    dQdTt = mDotCp;
                                } else {
                                    dQdTs = 0.0; dQdTt = 0.0;
                                }
                            } else {
                                dQdTs = 0.0; dQdTt = 0.0;
                            }
                        }
                    }
                } else if (eprop.getTypeCode() == EdgeProperties::TypeCode::HeatGeneration) {
                    // heat_generation は Q=constant なので係数0、導関数0
                }
                Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);
            }
            
            if (sv == balanceVertex) {
                outflow += Q;
                if (sIsVariable) addCoeffFast(i, sIdx, -dQdTs);
                if (tIsVariable) addCoeffFast(i, tIdx, -dQdTt);
            } else if (tv == balanceVertex) {
                if (eprop.getTypeCode() == EdgeProperties::TypeCode::ResponseConduction) {
                    // target側も outflow 扱い（壁体へ出ていく）
                    outflow += Q;
                    if (sIsVariable) addCoeffFast(i, sIdx, -dQdTs);
                    if (tIsVariable) addCoeffFast(i, tIdx, -dQdTt);
                } else {
                    inflow += Q;
                    if (sIsVariable) addCoeffFast(i, sIdx, dQdTs);
                    if (tIsVariable) addCoeffFast(i, tIdx, dQdTt);
                }
            }
        }

        // エアコンノードのset_nodeの熱バランスを肩代わり
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = airconSetVertex[static_cast<size_t>(nodeVertex)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                double setIn = 0.0, setOut = 0.0;
                for (auto e2 : incidentEdges[static_cast<size_t>(setV)]) {
                    Vertex sv2 = boost::source(e2, graph);
                    Vertex tv2 = boost::target(e2, graph);
                    const auto& ep = graph[e2];
                    
                    int sIdx2 = vertexToParameterIndex[static_cast<size_t>(sv2)];
                    int tIdx2 = vertexToParameterIndex[static_cast<size_t>(tv2)];
                    bool sIsVar2 = (sIdx2 >= 0);
                    bool tIsVar2 = (tIdx2 >= 0);
                    
                    double Ts2 = getTemp(sv2);
                    double Tt2 = getTemp(tv2);
                    double dQdTs2 = 0.0, dQdTt2 = 0.0;
                    double Q2 = 0.0;

                    if (ep.getTypeCode() == EdgeProperties::TypeCode::ResponseConduction) {
                        if (sv2 == setV) {
                            const double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0];
                            const double b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0];
                            dQdTs2 = a0;
                            dQdTt2 = b0;
                            Q2 = a0 * Ts2 + b0 * Tt2;
                            for (size_t k = 1; k < ep.resp_a_src.size(); ++k) {
                                const size_t idx = k - 1;
                                if (idx < ep.hist_t_src.size()) Q2 += ep.resp_a_src[k] * ep.hist_t_src[idx];
                            }
                            for (size_t k = 1; k < ep.resp_b_src.size(); ++k) {
                                const size_t idx = k - 1;
                                if (idx < ep.hist_t_tgt.size()) Q2 += ep.resp_b_src[k] * ep.hist_t_tgt[idx];
                            }
                            for (size_t k = 0; k < ep.resp_c_src.size(); ++k) {
                                if (k < ep.hist_q_src.size()) Q2 += ep.resp_c_src[k] * ep.hist_q_src[k];
                            }
                        } else if (tv2 == setV) {
                            const double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0];
                            const double b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0];
                            dQdTs2 = b0;
                            dQdTt2 = a0;
                            Q2 = a0 * Tt2 + b0 * Ts2;
                            for (size_t k = 1; k < ep.resp_a_tgt.size(); ++k) {
                                const size_t idx = k - 1;
                                if (idx < ep.hist_t_tgt.size()) Q2 += ep.resp_a_tgt[k] * ep.hist_t_tgt[idx];
                            }
                            for (size_t k = 1; k < ep.resp_b_tgt.size(); ++k) {
                                const size_t idx = k - 1;
                                if (idx < ep.hist_t_src.size()) Q2 += ep.resp_b_tgt[k] * ep.hist_t_src[idx];
                            }
                            for (size_t k = 0; k < ep.resp_c_tgt.size(); ++k) {
                                if (k < ep.hist_q_tgt.size()) Q2 += ep.resp_c_tgt[k] * ep.hist_q_tgt[k];
                            }
                        }
                    } else {
                        if (ep.getTypeCode() == EdgeProperties::TypeCode::Conductance) {
                            dQdTs2 = ep.conductance;
                            dQdTt2 = -ep.conductance;
                        } else if (ep.getTypeCode() == EdgeProperties::TypeCode::Advection) {
                            double flowRate = ep.flow_rate;
                            if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
                                double mDotCpAbs =
                                    archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
                                dQdTs2 = mDotCpAbs;
                                dQdTt2 = -mDotCpAbs;
                            }
                        }
                        Q2 = HeatCalculation::calculateUnifiedHeat(Ts2, Tt2, ep);
                    }
                    
                    if (sv2 == setV) {
                        setOut += Q2;
                        if (sIsVar2) addCoeffFast(i, sIdx2, dQdTs2);
                        if (tIsVar2) addCoeffFast(i, tIdx2, dQdTt2);
                    } else if (tv2 == setV) {
                        if (ep.getTypeCode() == EdgeProperties::TypeCode::ResponseConduction) {
                            // setV が target の場合も outflow 側として扱う（符号は source と同じ）
                            setOut += Q2;
                            if (sIsVar2) addCoeffFast(i, sIdx2, dQdTs2);
                            if (tIsVar2) addCoeffFast(i, tIdx2, dQdTt2);
                        } else {
                            setIn += Q2;
                            if (sIsVar2) addCoeffFast(i, sIdx2, -dQdTs2);
                            if (tIsVar2) addCoeffFast(i, tIdx2, -dQdTt2);
                        }
                    }
                }
                inflow += (setOut - setIn);
            }
        }
        
        double net = inflow - outflow;
        system.b[i] = -net; // J * delta = -residual
    }
}

bool solveSparseDirect(
    const LinearSystem& system,
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
            // 非ゼロ構造（row,col）からハッシュを構築（値は含めない）
            patternHash = fnv1a64_update(patternHash, (static_cast<uint64_t>(i) << 32) ^ static_cast<uint64_t>(cols[k]));
        }
    }

    Eigen::VectorXd b(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) b[static_cast<int>(i)] = system.b[i];

    // 非ゼロパターンが同一なら「Aの構造」を再生成せず、値だけ更新する
    const bool needRebuildPattern = (!g_sparseLuCache.analyzed) ||
                                   (g_sparseLuCache.n != static_cast<int>(n)) ||
                                   (g_sparseLuCache.nnz != nnz) ||
                                   (g_sparseLuCache.patternHash != patternHash);
    if (needRebuildPattern) {
        // Eigen::SparseLU は noncopyable のため、キャッシュ全体を代入で初期化しない
        g_sparseLuCache.analyzed = false;
        g_sparseLuCache.n = static_cast<int>(n);
        g_sparseLuCache.nnz = nnz;
        g_sparseLuCache.patternHash = patternHash;
        g_sparseLuCache.factorized = false;
        g_sparseLuCache.valueHash = 0;
        g_sparseLuCache.valuePtrIndexByRow.clear();
        g_sparseLuCache.A.resize(0, 0);
        // solver を in-place 再構築（analyzePattern の状態もクリア）
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

        // system.colIndices と同型の valuePtr インデックス表を構築（パターン再構築時のみ）
        g_sparseLuCache.valuePtrIndexByRow.assign(n, {});
        std::vector<std::vector<std::pair<int, int>>> rowEntries;
        rowEntries.assign(n, {});
        // rowLen は小さめなので、だいたいのサイズを予約しておく
        for (size_t r = 0; r < n; ++r) {
            rowEntries[r].reserve(system.colIndices[r].size());
        }
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
                } else {
                    // パターン不一致（想定外）。安全側で -1 のままにし、solve が失敗すればフォールバックする。
                    g_sparseLuCache.valuePtrIndexByRow[r][k] = -1;
                }
            }
        }
        // 想定外（-1）が混入していないかをチェック。混入している場合はキャッシュを無効化してフォールバックさせる。
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
                writeLog(logFile, "--------疎直接法: valuePtrIndexByRow の構築に失敗（パターン不一致）。停止します。");
            g_sparseLuCache.analyzed = false;
            g_sparseLuCache.factorized = false;
            g_sparseLuCache.valueHash = 0;
            g_sparseLuCache.valuePtrIndexByRow.clear();
            // SPD 側も無効化
            g_cholCache.analyzed = false;
            g_cholCache.factorized = false;
            return false;
        }

        g_sparseLuCache.solver.analyzePattern(g_sparseLuCache.A);
        g_sparseLuCache.analyzed = true;
        g_sparseLuCache.valueHash = valueHash;
        // SPD判定用キャッシュも無効化
        g_cholCache.analyzed = false;
        g_cholCache.factorized = false;
        // パターン対称性はこのタイミングで 1 回だけ判定して保持
        g_cholCache.patternSymmetric = isSymmetricPatternByCols(system.colIndices);
    } else {
        // 値だけ更新
        uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& cols = system.colIndices[i];
            const auto& vals = system.A[i];
            for (size_t k = 0; k < cols.size(); ++k) {
                const int p = g_sparseLuCache.valuePtrIndexByRow[i][k];
                if (p < 0) {
                    // ここに来るのは想定外（パターン不一致 or キャッシュ破損）
                    writeLog(logFile, "--------疎直接法: valuePtrIndexByRow が不正（-1）。停止します。");
                    g_sparseLuCache.factorized = false;
                    g_cholCache.factorized = false;
                    return false;
                }
                g_sparseLuCache.A.valuePtr()[p] = vals[k];
                valueHash = hashDoubleBits(valueHash, vals[k]);
            }
        }
        // 値が同一なら factorize を再利用できる
        if (g_sparseLuCache.valueHash != valueHash) {
            g_sparseLuCache.factorized = false;
            g_sparseLuCache.valueHash = valueHash;
            // 値が変わったのでSPD側も再factorizeが必要
            g_cholCache.factorized = false;
        }
    }

    // まず SPD（対称）なら Cholesky 系を試す（失敗したら SparseLU にフォールバック）
    // 値の対称性チェックは毎回 O(nnz) になるため行わず、パターンが対称な場合だけ Cholesky を試す。
    const bool symmetricCandidate = g_cholCache.patternSymmetric;

    Eigen::VectorXd sol;
    bool solved = false;
    if (symmetricCandidate) {
        // パターンが変わったら analyzePattern からやり直す
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
            // LLT（SPD前提）を先に試す
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
            // LLTが失敗ならLDLTを試す（より頑健）
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
                // Cholesky系がダメなら SparseLU へ
                g_cholCache.factorized = false;
                // このパターンで Cholesky が安定しない可能性が高いので、以後は試さない（パターン変化で復活）
                g_cholCache.patternSymmetric = false;
            }
        } else {
            // factorization再利用（bだけ変化）
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
                writeLog(logFile, "--------疎直接法(SparseLU)のfactorizeに失敗。停止します。");
                return false;
            }
            g_sparseLuCache.factorized = true;
        }
        sol = g_sparseLuCache.solver.solve(b);
        if (g_sparseLuCache.solver.info() != Eigen::Success) {
            writeLog(logFile, "--------疎直接法(SparseLU)のsolveに失敗。停止します。");
            return false;
        }
        methodLabel = g_sparseLuCache.factorized ? "SparseLU" : "SparseLU";
    }

    // 残差チェック（最大絶対値）
    Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
    double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
    if (!std::isfinite(maxResidual)) {
        writeLog(logFile, "--------疎直接法で非有限の残差が発生。停止します。");
        return false;
    }
    if (maxResidual > tolerance * 10.0) {
        // 直接法のはずだが、特異/悪条件で残差が大きい場合はフォールバック
        std::ostringstream oss;
        oss << "--------疎直接法の残差が大きいためフォールバックします: maxResidual="
            << std::scientific << std::setprecision(6) << maxResidual
            << " tol=" << tolerance;
        writeLog(logFile, oss.str());
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) {
            writeLog(logFile, "--------疎直接法で非有限の解が発生。停止します。");
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

    // パターンが対称で、Cholesky の factorization があるなら先に試す
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

    // Cholesky が無い/失敗したら SparseLU を試す
    if (!ok && g_sparseLuCache.analyzed && g_sparseLuCache.factorized) {
        sol = g_sparseLuCache.solver.solve(b);
        if (g_sparseLuCache.solver.info() == Eigen::Success) {
            ok = true;
            methodLabel = "SparseLU(cached)";
        }
    }

    if (!ok) {
        return false;
    }

    // 残差チェックは sparse mat-vec が重いので、cached 経路では間引く
    // （factorization が同一で b だけ変わるケースが多く、毎回チェックすると支配的になり得る）
    constexpr uint64_t kCachedResidualCheckInterval = 200; // 200回に1回だけチェック
    if ((s_cachedResidualCheckCounter++ % kCachedResidualCheckInterval) == 0) {
        Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
        double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
        if (!std::isfinite(maxResidual)) {
            writeLog(logFile, "--------疎直接法(cached)で非有限の残差が発生。停止します。");
            return false;
        }
        if (maxResidual > tolerance * 10.0) {
            std::ostringstream oss;
            oss << "--------疎直接法(cached)の残差が大きいためフォールバックします: maxResidual="
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

void solveTemperaturesLinearGS(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    std::ostream& logFile) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto& graph = network.getGraph();
    const size_t curV = boost::num_vertices(graph);
    const size_t curE = boost::num_edges(graph);

    // トポロジ（頂点/辺）が不変なら、毎回の構築コストを避けてキャッシュを使う
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

        // advection 辺・aircon 頂点のリスト（係数シグネチャ計算の高速化）
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

        // 係数行列Aの非ゼロパターン（row/col）を初回に確定（以後は同じ位置に加算する）
        g_topologyCache.rowColsPattern.assign(g_topologyCache.nodeNames.size(), {});
        for (size_t row = 0; row < g_topologyCache.nodeNames.size(); ++row) {
            Vertex v = g_topologyCache.parameterIndexToVertex[row];
            std::vector<int> cols;
            cols.reserve(16);
            cols.push_back(static_cast<int>(row)); // diag

            auto addVar = [&](Vertex vv) {
                int idx = g_topologyCache.vertexToParameterIndex[static_cast<size_t>(vv)];
                if (idx >= 0) cols.push_back(idx);
            };

            for (auto e : g_topologyCache.incidentEdges[static_cast<size_t>(v)]) {
                addVar(boost::source(e, graph));
                addVar(boost::target(e, graph));
            }

            // aircon肩代わりで setV 周辺の変数も参照する可能性がある
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

        // row -> (col -> local index) を構築
        g_topologyCache.rowIndexMaps.assign(g_topologyCache.rowColsPattern.size(), {});
        for (size_t row = 0; row < g_topologyCache.rowColsPattern.size(); ++row) {
            g_topologyCache.rowIndexMaps[row].buildFromCols(g_topologyCache.rowColsPattern[row]);
        }

        g_topologyCache.initialized = true;
    }

    const auto& incidentEdges = g_topologyCache.incidentEdges;
    const auto& airconBySetVertex = g_topologyCache.airconBySetVertex;
    const auto& nodeNames = g_topologyCache.nodeNames;
    const auto& vertexToParameterIndex = g_topologyCache.vertexToParameterIndex;
    const auto& airconSetVertex = g_topologyCache.airconSetVertex;
    const auto& rowIndexMaps = g_topologyCache.rowIndexMaps;
    
    if (nodeNames.empty()) {
        writeLog(logFile, "--警告: 温度計算対象のノードがありません");
        return;
    }
    
    // 初期温度を設定
    std::vector<double> temperatures(nodeNames.size());
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        Vertex v = g_topologyCache.parameterIndexToVertex[i];
        temperatures[i] = graph[v].current_t;
    }
    
    // 線形（温度に関して）を前提に、A が不変なら factorize を再利用して b だけ更新して解く
    // LinearSystem はタイムステップごとに作り直すと、rowごとの vector 再確保が支配的になり得るため、
    // パターン不変の間はバッファを再利用する。
    //
    // 注意: factorization 再利用（b のみ更新）経路では A を一切使わないため、
    // ここで毎回 A をゼロクリアすると逆に遅くなる。A のクリアは buildLinearSystem() 側に任せ、
    // 必要なときだけ行う。
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

    const bool canReuseFactorization =
        g_sparseLuCache.analyzed &&
        (g_sparseLuCache.factorized || (g_cholCache.analyzed && g_cholCache.factorized)) &&
        g_sparseLuCache.n == static_cast<int>(nodeNames.size()) &&
        g_sparseLuCache.coeffSig == coeffSig;

    if (canReuseFactorization) {
        // A は不変とみなせるので b だけ作って solve(b)
        buildRhsOnly(graph, g_topologyCache, &temperatures, system.b);
    } else {
        // A が変わった（または未初期化）ので通常経路で A,b を組み立てて factorize する
        buildLinearSystem(
            graph,
            vertexToParameterIndex,
            g_topologyCache.parameterIndexToVertex,
            incidentEdges,
            airconBySetVertex,
            airconSetVertex,
            rowIndexMaps,
            &temperatures,
            system);
    }

    std::vector<double> delta(nodeNames.size(), 0.0);
    bool solved = false;
    std::string methodLabel = "direct";

    // 疎直接法のみ（フォールバックなし）
    {
        bool ok = false;
        if (canReuseFactorization) {
            // factorize を再利用して solve のみ（LLT/LDLT/SparseLU を自動選択）
            Eigen::VectorXd b(static_cast<int>(delta.size()));
            for (size_t i = 0; i < delta.size(); ++i) b[static_cast<int>(i)] = system.b[i];
            ok = solveWithCachedFactorization(b, delta, constants.thermalTolerance, logFile, methodLabel);
            if (!ok) {
                writeLog(logFile, "--------疎直接法(cached)で失敗。停止します。");
            }
        } else {
            ok = solveSparseDirect(system, delta, constants.thermalTolerance, logFile, methodLabel);
            if (ok) {
                // 現在の係数状態をキャッシュ（次回以降 b のみで回せる）
                g_sparseLuCache.coeffSig = coeffSig;
            }
        }
        solved = ok;
    }

    if (!solved) {
        throw std::runtime_error("thermal solve failed (direct solver only; GS+SOR disabled)");
    }

    // 線形問題ではこの1回の更新で解に到達する（ヤコビアン一定のため）
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        temperatures[i] += delta[i];
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
        double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);

        // 個別ブランチの出力用
        if (eprop.getTypeCode() == EdgeProperties::TypeCode::Advection) {
            // 旧バージョン互換: エアコンノードへの流入熱量（還気）は、エアコンON時のみ0とする
            if (eprop.flow_rate > 0) { // sv -> tv
                if (eprop.is_aircon_inflow && graph[tv].on) Q = 0.0;
            } else if (eprop.flow_rate < 0) { // tv -> sv
                const auto& srcNode = graph[sv];
                if (srcNode.getTypeCode() == VertexProperties::TypeCode::Aircon && srcNode.on) Q = 0.0;
            }
        }
        graph[edge].heat_rate = Q;

        // ノード収支（片方向移流モデルを考慮）
        if (eprop.getTypeCode() == EdgeProperties::TypeCode::Advection) {
            if (eprop.flow_rate > 0) { // sv -> tv
                heatBalanceByVertex[static_cast<size_t>(tv)] += Q;
            } else if (eprop.flow_rate < 0) { // tv -> sv
                heatBalanceByVertex[static_cast<size_t>(sv)] -= Q;
            }
        } else {
            heatBalanceByVertex[static_cast<size_t>(sv)] -= Q;
            heatBalanceByVertex[static_cast<size_t>(tv)] += Q;
        }
    }

    // --- エアコンによるバランス付け替え（収束判定用） ---
    // A案: aircon ON のとき、aircon 行は set_node の熱収支=0 を肩代わりしている。
    // それに合わせて残差も aircon := set, set := 0 として評価する。
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const auto& nodeData = graph[v];
        if (nodeData.getTypeCode() == VertexProperties::TypeCode::Aircon && nodeData.on) {
            Vertex setV = g_topologyCache.airconSetVertex[static_cast<size_t>(v)];
            if (setV != std::numeric_limits<Vertex>::max()) {
                heatBalanceByVertex[static_cast<size_t>(v)] = heatBalanceByVertex[static_cast<size_t>(setV)];
                heatBalanceByVertex[static_cast<size_t>(setV)] = 0.0;
            }
        }
    }

    // 収束判定用のRMSE/maxBalance（calc_t ノードに対して算出）
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
        oss << "--------熱計算(線形): "
            << (ok ? "収束" : "未収束/バランス超過")
            << " (method=" << methodLabel
            << ", RMSE=" << std::scientific << std::setprecision(6) << rmseBalance
            << ", maxBalance=" << std::scientific << std::setprecision(6) << maxBalance
            << ", tol=" << std::scientific << std::setprecision(6) << constants.thermalTolerance
            << ", time=" << std::fixed << std::setprecision(3) << seconds << "s)";
        writeLog(logFile, oss.str());
        // 上位（エアコン制御ループ等）が判断できるように状態として保持
        network.setLastThermalConvergence(ok, rmseBalance, maxBalance, methodLabel);
    }
}

} // namespace ThermalSolverLinearGS


