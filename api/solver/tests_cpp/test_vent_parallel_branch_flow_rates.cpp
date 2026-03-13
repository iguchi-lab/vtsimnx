#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
 
#include "core/ventilation/flow_calculation.h"
#include "network/ventilation_network.h"
 
namespace {
 
int g_failures = 0;
 
void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}
 
void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}
 
void expectNear(double actual, double expected, double tol, const std::string& msg) {
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        fail(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) +
             ", diff=" + std::to_string(diff) + ", tol=" + std::to_string(tol) + ")");
    }
}
 
VertexProperties makeNode(const std::string& key) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.calc_p = true;
    v.current_p = 0.0;
    v.current_t = 20.0; // ℃（density計算が temperature + 273.15 を仮定）
    return v;
}
 
EdgeProperties makeGapEdge(const std::string& key,
                           const std::string& unique_id,
                           const std::string& s,
                           const std::string& t,
                           double a,
                           double n) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = unique_id;
    e.type = "gap";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.a = a;
    e.n = n;
    e.h_from = 0.0;
    e.h_to = 0.0;
    return e;
}
 
} // namespace
 
int main() {
    // 回帰テスト:
    // - FlowRateMap は (source,target) の合算（並列枝は合算）になりうる
    // - それを各枝へそのまま配ると並列枝が同じ flow_rate になってしまう
    // - updateFlowRatesInGraph() は差圧から枝ごとの流量を再計算して更新すべき
    VentilationNetwork net;
    net.addNode(makeNode("A"));
    net.addNode(makeNode("B"));
 
    // A->B の並列枝2本（係数aが異なるので、同一差圧でも流量は異なる）
    net.addEdge(makeGapEdge("A->B(1)", "A->B(1)", "A", "B", /*a=*/0.01, /*n=*/1.0));
    net.addEdge(makeGapEdge("A->B(2)", "A->B(2)", "A", "B", /*a=*/0.02, /*n=*/1.0));
 
    // 差圧 dp = 100Pa を作る（高さ補正は0）
    PressureMap pm;
    pm["A"] = 100.0;
    pm["B"] = 0.0;
    net.updateNodePressures(pm);
 
    // 合算 flowRates（旧実装ではこれが各枝にそのまま配られていた）
    FlowRateMap flowRates;
    flowRates[{ "A", "B" }] = 999.0;
 
    net.updateFlowRatesInGraph(flowRates);
 
    const Graph& g = net.getGraph();
    double q1 = std::numeric_limits<double>::quiet_NaN();
    double q2 = std::numeric_limits<double>::quiet_NaN();
    double exp1 = std::numeric_limits<double>::quiet_NaN();
    double exp2 = std::numeric_limits<double>::quiet_NaN();
 
    for (auto e : boost::make_iterator_range(boost::edges(g))) {
        const auto& ep = g[e];
        if (ep.key == "A->B(1)") {
            q1 = ep.flow_rate;
            exp1 = FlowCalculation::calculateUnifiedFlow(/*dp=*/100.0, ep);
            expectNear(q1, exp1, 1e-12, "parallel(1): flow_rate matches per-edge recompute");
        } else if (ep.key == "A->B(2)") {
            q2 = ep.flow_rate;
            exp2 = FlowCalculation::calculateUnifiedFlow(/*dp=*/100.0, ep);
            expectNear(q2, exp2, 1e-12, "parallel(2): flow_rate matches per-edge recompute");
        }
    }
 
    expectTrue(std::isfinite(q1) && std::isfinite(q2), "both parallel branch flow rates are set");
    expectTrue(std::isfinite(exp1) && std::isfinite(exp2), "expected flow rates are finite");
    expectTrue(std::abs(q1 - q2) > 1e-9, "parallel branches should not share the same flow_rate");
 
    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


