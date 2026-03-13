#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <chrono>

#include "core/thermal/heat_calculation.h"
#include "core/thermal/thermal_direct_internal.h"
#include "network/thermal_network.h"

namespace {

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (!std::isfinite(actual)) {
        throw std::runtime_error(msg + " (actual is not finite)");
    }
    const double diff = std::abs(actual - expected);
    if (diff > tol) {
        std::ostringstream oss;
        oss << msg << " actual=" << actual << " expected=" << expected
            << " diff=" << diff << " tol=" << tol;
        throw std::runtime_error(oss.str());
    }
}

VertexProperties makeNode(const std::string& key, bool calcT, double t) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.calc_t = calcT;
    v.current_t = t;
    v.heat_source = 0.0;
    return v;
}

double runPostprocessCase(double tempA,
                          double tempB,
                          double flowRate,
                          double heatSourceA,
                          const std::string& edgeSource,
                          const std::string& edgeTarget) {
    ThermalNetwork net;
    auto a = makeNode("A", true, tempA);
    auto b = makeNode("B", false, tempB);
    a.heat_source = heatSourceA;
    b.heat_source = 0.0;
    net.addNode(a);
    net.addNode(b);

    EdgeProperties adv{};
    adv.key = "adv";
    adv.unique_id = "adv";
    adv.type = "advection";
    adv.source = edgeSource;
    adv.target = edgeTarget;
    adv.flow_rate = flowRate;
    net.addEdge(adv);

    auto& graph = net.getGraph();
    const auto& map = net.getKeyToVertex();

    ThermalSolverLinearDirect::detail::TopologyCache topo;
    topo.parameterIndexToVertex = {map.at("A")}; // A のみを収支評価対象にする
    topo.airconSetVertex.assign(boost::num_vertices(graph), std::numeric_limits<Vertex>::max());

    SimulationConstants constants{};
    constants.logVerbosity = 0;
    constants.thermalTolerance = 1e-9;

    ThermalSolverLinearDirect::detail::DirectTStats stats{};
    stats.calls = 1;
    std::ostringstream logs;
    ThermalSolverLinearDirect::detail::postprocessAndReport(
        net, graph, topo,
        static_cast<size_t>(boost::num_vertices(graph)),
        topo.parameterIndexToVertex.size(),
        constants,
        "test-postprocess",
        logs,
        std::chrono::high_resolution_clock::now(),
        stats);

    return net.getLastThermalRmseBalance();
}

} // namespace

int main() {
    try {
        // 等価な2ケース：
        // 1) A->B, flow<0  (実流向は B->A)
        // 2) B->A, flow>0
        // A の熱収支は同じになるべき。さらに heat_source を与えて打ち消しゼロを作る。
        //
        // Qref は (case2) の A への移流寄与。case1 でも一致するのが期待値。
        EdgeProperties refEdge{};
        refEdge.type = "advection";
        refEdge.flow_rate = 0.2;
        const double tempA = 30.0;
        const double tempB = 20.0;
        const double qRefAtA = HeatCalculation::calcAdvectionHeat(tempB, tempA, refEdge); // B->A
        const double heatSourceA = -qRefAtA; // 収支をゼロにする

        const double rmseNegFlowCase = runPostprocessCase(
            tempA, tempB, -0.2, heatSourceA, "A", "B");
        const double rmsePosFlowCase = runPostprocessCase(
            tempA, tempB, 0.2, heatSourceA, "B", "A");

        expectNear(rmsePosFlowCase, 0.0, 1e-8,
                   "baseline (positive-flow equivalent case) RMSE should be zero");
        expectNear(rmseNegFlowCase, 0.0, 1e-8,
                   "negative-flow advection RMSE should be zero");
        expectNear(rmseNegFlowCase, rmsePosFlowCase, 1e-12,
                   "negative-flow and positive-flow equivalent cases must match");
        expectTrue(std::isfinite(rmseNegFlowCase) && std::isfinite(rmsePosFlowCase),
                   "RMSE must be finite");

        std::cout << "[OK] all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}

