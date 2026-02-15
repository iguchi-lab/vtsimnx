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

    auto idxOf = [](Vertex v) -> size_t { return static_cast<size_t>(v); };
    const size_t nV = static_cast<size_t>(boost::num_vertices(tGraph));
    std::vector<double> outSum(nV, 0.0);
    std::vector<std::vector<std::pair<Vertex, double>>> inflow(nV);

    // 流量（m3/s）から inflow/outflow を構築し、質量流量（kg/s）へ変換して扱う
    for (const auto& kv : flowRates) {
        const std::string& a = kv.first.first;
        const std::string& b = kv.first.second;
        const double f = kv.second;
        if (f == 0.0) continue;

        std::string srcKey = a;
        std::string dstKey = b;
        // FlowRateMap は volumetric flow [m3/s]（pressure solver の出力）なので、rho を掛けて [kg/s]
        double mDot = f * PhysicalConstants::DENSITY_DRY_AIR;
        if (mDot < 0.0) {
            mDot = -mDot;
            srcKey = b;
            dstKey = a;
        }
        auto itS = tKeyToV.find(srcKey);
        auto itD = tKeyToV.find(dstKey);
        if (itS == tKeyToV.end() || itD == tKeyToV.end()) continue;

        const Vertex sv = itS->second;
        const Vertex dv = itD->second;
        outSum[idxOf(sv)] += mDot;
        inflow[idxOf(dv)].push_back({sv, mDot});
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
            const double g = [&]() -> double {
                auto itG = genByVertex.find(v);
                return (itG == genByVertex.end()) ? 0.0 : itG->second;
            }();

            // v<=0 の場合は「流入混合のみ」（旧vtsim互換の安全側）で処理
            if (!(V > 0.0)) {
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

            const double mAir = rho * V;   // [kg]
            const double out = outSum[i];  // [kg/s]
            const double denom = 1.0 + dt * out / mAir;

            double rhs = xOld[i];
            rhs += dt * (g / mAir); // g: [kg/s] -> kg/kg/s
            for (const auto& in : inflow[i]) {
                const Vertex sv = in.first;
                const double md = in.second;
                rhs += dt * (md / mAir) * xNew[idxOf(sv)];
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
        auto itV = vKeyToV.find(tGraph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_x = xNew[i];
        }
    }
}

} // namespace transport


