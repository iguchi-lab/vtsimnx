#include "transport/humidity_solver.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "vtsimnx_solver_timing.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include <boost/range/iterator_range.hpp>

namespace transport {

void updateHumidityIfEnabled(const SimulationConstants& constants,
                             VentilationNetwork& ventNetwork,
                             ThermalNetwork& thermalNetwork,
                             const FlowRateMap& flowRates,
                             std::ostream& logs,
                             TimingList& timings,
                             const std::string& meta) {
    (void)logs;
    if (!constants.humidityCalc) return;

    ScopedTimer timer(timings, "humidity_update", meta);

    auto& tGraph = thermalNetwork.getGraph();
    auto& vGraph = ventNetwork.getGraph();
    const auto& tKeyToV = thermalNetwork.getKeyToVertex();
    const auto& vKeyToV = ventNetwork.getKeyToVertex();

    const double dt = static_cast<double>(constants.timestep);
    if (!(dt > 0.0)) return;

    // 生成項（発湿）: 換気ブランチの humidity_generation を target 側へ集計
    std::unordered_map<Vertex, double> genByVertex;
    genByVertex.reserve(boost::num_vertices(tGraph) / 4 + 1);
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        const double g = ep.current_humidity_generation;
        if (g == 0.0) continue;
        auto itT = tKeyToV.find(ep.target);
        if (itT == tKeyToV.end()) continue;
        genByVertex[itT->second] += g;
    }

    (void)flowRates; // エッジ直接走査方式に統一したため FlowRateMap は不使用

    auto idxOf = [](Vertex v) -> size_t { return static_cast<size_t>(v); };
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    std::vector<double> outSum(nV, 0.0);
    std::vector<std::vector<std::pair<Vertex, double>>> inflow(nV);
    std::vector<std::vector<std::pair<Vertex, double>>> moistureLinks(nV);

    // ベントグラフのエッジを直接走査して inflow/outflow を構築する。
    // concentration_solver と同方式にすることで:
    //   1. 同一ノードペアに複数エッジが存在する場合も各エッジの寄与を独立に処理できる
    //   2. calc_p=true で流量が逆転した際も個別の寄与が正しく反映される
    //   3. FlowRateMap の集計による src/dst の取り違えリスクがなくなる
    for (auto e : boost::make_iterator_range(boost::edges(vGraph))) {
        const auto& ep = vGraph[e];
        const double f = ep.flow_rate; // [m3/s]（正: エッジ定義方向、負: 逆向き）
        if (f == 0.0) continue;

        const Vertex vSv = boost::source(e, vGraph);
        const Vertex vTv = boost::target(e, vGraph);
        const std::string& kS = vGraph[vSv].key;
        const std::string& kT = vGraph[vTv].key;

        auto itTS = tKeyToV.find(kS);
        auto itTT = tKeyToV.find(kT);
        if (itTS == tKeyToV.end() || itTT == tKeyToV.end()) continue;

        // 流量の符号からソース/デスティネーションを決定し、質量流量 [kg/s] に変換
        Vertex src = itTS->second;
        Vertex dst = itTT->second;
        double mDot = f * PhysicalConstants::DENSITY_DRY_AIR;
        if (mDot < 0.0) {
            mDot = -mDot;
            std::swap(src, dst);
        }

        outSum[idxOf(src)] += mDot;
        inflow[idxOf(dst)].push_back({src, mDot});
    }

    // 湿気回路網（Phase1）:
    // thermal_branches の moisture_conductance を「双方向の伝達係数」として扱う。
    // 式の形: dxi/dt += (Kij/Ci) * (xj - xi)
    for (auto e : boost::make_iterator_range(boost::edges(tGraph))) {
        const auto& ep = tGraph[e];
        const double k = ep.moisture_conductance;
        if (!(k > 0.0)) continue;
        const Vertex sv = boost::source(e, tGraph);
        const Vertex tv = boost::target(e, tGraph);
        moistureLinks[idxOf(sv)].push_back({tv, k});
        moistureLinks[idxOf(tv)].push_back({sv, k});
    }

    // 更新対象（calc_x=true）の頂点を key でソートして決定性を確保
    std::vector<Vertex> updateVertices;
    updateVertices.reserve(nV / 4 + 1);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        if (tGraph[v].calc_x) updateVertices.push_back(v);
    }
    std::sort(updateVertices.begin(), updateVertices.end(), [&](Vertex a, Vertex b) {
        return tGraph[a].key < tGraph[b].key;
    });

    // Gauss-Seidel（陰解法）で 1 ステップ更新: (I - dt*A) x_{n+1} = x_n + dt*b
    std::vector<double> xNew(nV, 0.0);
    std::vector<double> xOld(nV, 0.0);
    for (auto v : boost::make_iterator_range(boost::vertices(tGraph))) {
        const size_t i = idxOf(v);
        xNew[i] = tGraph[v].current_x;
        xOld[i] = tGraph[v].current_x;
    }

    constexpr double rho = PhysicalConstants::DENSITY_DRY_AIR; // [kg/m3]（簡易。温度依存は今後拡張）
    const int maxIter = 80;
    const double tol = 1e-9;

    for (int it = 0; it < maxIter; ++it) {
        double maxDiff = 0.0;
        for (Vertex v : updateVertices) {
            const size_t i = idxOf(v);
            const double V = tGraph[v].v; // [m3]
            const double cap = (tGraph[v].moisture_capacity > 0.0)
                                   ? tGraph[v].moisture_capacity
                                   : (rho * V); // [kg/(kg/kg)] 相当
            const double g = [&]() -> double {
                auto itG = genByVertex.find(v);
                return (itG == genByVertex.end()) ? 0.0 : itG->second;
            }();

            // 容量が無い場合（境界ノード等）は流入混合のみ（既存互換）
            if (!(cap > 0.0)) {
                double sumIn = 0.0;
                double sumInX = 0.0;
                for (const auto& in : inflow[i]) {
                    const Vertex sv = in.first;
                    const double md = in.second;
                    sumIn += md;
                    sumInX += md * xNew[idxOf(sv)];
                }
                double x = xNew[i];
                if (sumIn > 0.0) x = sumInX / sumIn;
                maxDiff = std::max(maxDiff, std::abs(x - xNew[i]));
                xNew[i] = x;
                continue;
            }

            const double out = outSum[i];  // [kg/s]
            double denom = 1.0 + dt * out / cap;

            double rhs = xOld[i];
            rhs += dt * (g / cap); // g: [kg/s]
            for (const auto& in : inflow[i]) {
                const Vertex sv = in.first;
                const double md = in.second;
                rhs += dt * (md / cap) * xNew[idxOf(sv)];
            }
            for (const auto& lk : moistureLinks[i]) {
                const Vertex ov = lk.first;
                const double k = lk.second;
                denom += dt * (k / cap);
                rhs += dt * (k / cap) * xNew[idxOf(ov)];
            }

            const double x = rhs / denom;
            maxDiff = std::max(maxDiff, std::abs(x - xNew[i]));
            xNew[i] = x;
        }
        if (maxDiff < tol) break;
    }

    // graph へ反映（thermal/vent 両方に入れておく）
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

} // namespace transport


