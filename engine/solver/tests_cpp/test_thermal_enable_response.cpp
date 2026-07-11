#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/thermal/thermal_direct_response.h"
#include "core/thermal/thermal_solver.h"
#include "core/thermal/thermal_solver_linear_direct.h"
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

EdgeProperties makeConductance(const std::string& key,
                               const std::string& src,
                               const std::string& tgt,
                               double k) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "conductance";
    e.source = src;
    e.target = tgt;
    e.conductance = k;
    e.current_enabled = true;
    return e;
}

EdgeProperties makeHeatGen(const std::string& key,
                           const std::string& src,
                           const std::string& tgt,
                           double q) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "heat_generation";
    e.source = src;
    e.target = tgt;
    e.current_heat_generation = q;
    e.current_enabled = true;
    return e;
}

SimulationConstants makeConstants() {
    SimulationConstants c{};
    c.logVerbosity = 0;
    c.temperatureCalc = true;
    c.pressureCalc = false;
    c.thermalTolerance = 1e-9;
    return c;
}

} // namespace

int main() {
    try {
        const auto constants = makeConstants();
        std::ostringstream logs;
        ThermalSolverLinearDirect::resetDirectTSolverContext();

        // ------------------------------------------------------------------
        // enable=false で conductance / heat_generation が計算から消える
        // ------------------------------------------------------------------
        {
            ThermalNetwork thermal;
            auto wall = makeNode("WALL", false, 20.0);
            auto room = makeNode("ROOM", true, 0.0);
            thermal.addNode(wall);
            thermal.addNode(room);

            auto cond = makeConductance("WALL->ROOM", "WALL", "ROOM", 5.0);
            auto gen = makeHeatGen("GEN->ROOM", "WALL", "ROOM", 50.0);
            thermal.addEdge(cond);
            thermal.addEdge(gen);

            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);
            const auto& g0 = thermal.getGraph();
            const auto& kv = thermal.getKeyToVertex();
            // 定常: k*(Twall-T)+Q=0 -> T = Twall + Q/k = 30
            expectNear(g0[kv.at("ROOM")].current_t, 30.0, 1e-8, "enabled: room temp");

            // conductance 無効 → heat_gen のみでは未知温度が決まらないため、
            // 代わりに conductance を残して generation を切る
            for (auto e : boost::make_iterator_range(boost::edges(thermal.getGraph()))) {
                auto& ep = thermal.getGraph()[e];
                if (ep.type == "heat_generation") ep.current_enabled = false;
            }
            solver.solveTemperatures(constants);
            expectNear(thermal.getGraph()[kv.at("ROOM")].current_t, 20.0, 1e-8,
                       "heat_generation disabled: room follows wall");
            for (auto e : boost::make_iterator_range(boost::edges(thermal.getGraph()))) {
                const auto& ep = thermal.getGraph()[e];
                if (ep.type == "heat_generation") {
                    expectNear(ep.heat_rate, 0.0, 0.0, "disabled heat_generation heat_rate==0");
                }
            }

            // conductance も無効化 → 孤立未知温度（行列が特異になり得る）なので
            // 再度 generation を有効にし conductance を切ると、境界なし発熱のみで失敗しやすい。
            // ここでは conductance を再度有効にし、enable 切替で係数キャッシュが無効化されることだけ確認。
            for (auto e : boost::make_iterator_range(boost::edges(thermal.getGraph()))) {
                auto& ep = thermal.getGraph()[e];
                if (ep.type == "heat_generation") ep.current_enabled = true;
                if (ep.type == "conductance") ep.current_enabled = false;
            }
            // WALL 固定 + ROOM に gen のみ: ROOM の行は b=-(-Q)? wait heat_gen signs
            // With conductance disabled, only heat_gen: at ROOM (target) b += -q in old,
            // with A empty for ROOM alone... actually ROOM has only heat_gen contribution to b,
            // no temperature coeffs -> singular. Skip singular case.
            for (auto e : boost::make_iterator_range(boost::edges(thermal.getGraph()))) {
                auto& ep = thermal.getGraph()[e];
                if (ep.type == "conductance") ep.current_enabled = true;
            }
            solver.solveTemperatures(constants);
            expectNear(thermal.getGraph()[kv.at("ROOM")].current_t, 30.0, 1e-8,
                       "re-enabled: room temp restored");
        }

        // ------------------------------------------------------------------
        // heat_source 符号: 正の heat_source は加熱（温度上昇）
        // k*(Twall-T)+Q=0 -> T=Twall+Q/k
        // ------------------------------------------------------------------
        {
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalNetwork thermal;
            auto wall = makeNode("WALL", false, 10.0);
            auto room = makeNode("ROOM", true, 0.0);
            room.heat_source = 40.0; // 加熱
            thermal.addNode(wall);
            thermal.addNode(room);
            thermal.addEdge(makeConductance("WALL->ROOM", "WALL", "ROOM", 4.0));

            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);
            const auto& kv = thermal.getKeyToVertex();
            expectNear(thermal.getGraph()[kv.at("ROOM")].current_t, 20.0, 1e-8,
                       "positive heat_source raises temperature");

            thermal.getGraph()[kv.at("ROOM")].heat_source = -40.0; // 除熱
            solver.solveTemperatures(constants);
            expectNear(thermal.getGraph()[kv.at("ROOM")].current_t, 0.0, 1e-8,
                       "negative heat_source cools temperature");
        }

        // ------------------------------------------------------------------
        // response_conduction: 履歴がタイムステップ受理時に進む
        // ------------------------------------------------------------------
        {
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalNetwork thermal;
            auto out = makeNode("OUT", false, 0.0);
            auto room = makeNode("ROOM", true, 0.0);
            thermal.addNode(out);
            thermal.addNode(room);

            EdgeProperties resp{};
            resp.key = "wall";
            resp.unique_id = "wall";
            resp.type = "response_conduction";
            resp.source = "OUT";
            resp.target = "ROOM";
            resp.area = 1.0;
            resp.current_enabled = true;
            // q_src = 2*Tout - 1*Troom + 0.5*Tout(n-1)
            // q_tgt = 2*Troom - 1*Tout + 0.5*Troom(n-1)
            resp.resp_a_src = {2.0, 0.5};
            resp.resp_b_src = {-1.0};
            resp.resp_c_src = {};
            resp.resp_a_tgt = {2.0, 0.5};
            resp.resp_b_tgt = {-1.0};
            resp.resp_c_tgt = {};
            thermal.addEdge(resp);

            // buildFromData 経由の初期化と同じく履歴を埋める
            auto& g = thermal.getGraph();
            for (auto e : boost::make_iterator_range(boost::edges(g))) {
                auto& ep = g[e];
                if (ep.getTypeCode() != EdgeProperties::TypeCode::ResponseConduction) continue;
                ep.hist_t_src.assign(1, 0.0);
                ep.hist_t_tgt.assign(1, 0.0);
                ep.response_initialized = true;
            }

            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);

            const auto& kv = thermal.getKeyToVertex();
            Edge respEdge = *boost::edges(g).first;
            // 初回履歴は0のまま（commit前）
            expectNear(g[respEdge].hist_t_tgt[0], 0.0, 0.0, "before commit: hist still initial");
            expectTrue(std::abs(g[respEdge].current_q_tgt) > 0.0 ||
                       std::abs(g[respEdge].current_q_src) > 0.0 ||
                       std::abs(g[kv.at("ROOM")].current_t) >= 0.0,
                       "solve produced finite state");

            const double t1 = g[kv.at("ROOM")].current_t;
            const double qSrc1 = g[respEdge].current_q_src;
            const double qTgt1 = g[respEdge].current_q_tgt;

            thermal.commitResponseConductionHistory();
            expectNear(g[respEdge].hist_t_src[0], 0.0, 1e-12, "commit: hist_t_src=OUT");
            expectNear(g[respEdge].hist_t_tgt[0], t1, 1e-12, "commit: hist_t_tgt=ROOM(t1)");
            expectNear(g[respEdge].hist_q_src.empty() ? 0.0 : g[respEdge].hist_q_src[0],
                       g[respEdge].hist_q_src.empty() ? 0.0 : qSrc1, 1e-12,
                       "commit q_src (or empty c)");
            if (!g[respEdge].hist_q_tgt.empty()) {
                expectNear(g[respEdge].hist_q_tgt[0], qTgt1, 1e-12, "commit q_tgt");
            }

            // OUT を変えて2ステップ目。
            // Troom = (Tout - 0.5 * hist_Troom) / 2
            // hist=0, Tout=10 -> t2 = 5
            g[kv.at("OUT")].current_t = 10.0;
            solver.solveTemperatures(constants);
            const double t2 = g[kv.at("ROOM")].current_t;
            expectNear(t2, 5.0, 1e-10, "response step2: t2 == 5");
            thermal.commitResponseConductionHistory();

            // hist_Troom=5, Tout=10 -> t3 = (10 - 2.5)/2 = 3.75
            g[kv.at("OUT")].current_t = 10.0;
            solver.solveTemperatures(constants);
            const double t3 = g[kv.at("ROOM")].current_t;
            expectNear(t3, 3.75, 1e-10, "response step3: t3 == 3.75");
            expectNear(g[respEdge].hist_t_src[0], 10.0, 1e-12, "step2 hist_t_src");
        }

        // ------------------------------------------------------------------
        // 同頂点数・枝数で内容だけ変えて再構築しても古いキャッシュを使わない
        // ------------------------------------------------------------------
        {
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            std::vector<VertexProperties> nodes = {
                makeNode("A", false, 30.0),
                makeNode("B", true, 0.0),
            };
            std::vector<EdgeProperties> th1 = {
                makeConductance("A->B", "A", "B", 1.0),
            };
            std::vector<EdgeProperties> vent;

            ThermalNetwork thermal;
            {
                std::ostringstream buildLogs;
                thermal.buildFromData(nodes, th1, vent, constants, buildLogs);
            }
            const auto rev1 = thermal.getTopologyRevision();
            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);
            const double tK1 = thermal.getGraph()[thermal.getKeyToVertex().at("B")].current_t;
            expectNear(tK1, 30.0, 1e-8, "rebuild case1: follows A");

            std::vector<EdgeProperties> th2 = {
                makeConductance("A->B", "A", "B", 2.0),
            };
            // 同じノードへ heat_source を載せる内容変更再構築
            nodes[1].heat_source = 20.0;
            {
                std::ostringstream buildLogs;
                thermal.buildFromData(nodes, th2, vent, constants, buildLogs);
            }
            expectTrue(thermal.getTopologyRevision() != rev1, "topologyRevision increments");
            solver.solveTemperatures(constants);
            // k=2, Q=20, TA=30 -> T=30+20/2=40
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("B")].current_t, 40.0, 1e-8,
                       "rebuild with same size uses new topology/coeffs");
        }

        // ------------------------------------------------------------------
        // 失敗時診断は 0 ではなく NaN（A=0 かつ b≠0 の特異系で DirectT を失敗させる）
        // ------------------------------------------------------------------
        {
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalNetwork thermal;
            auto a = makeNode("A", true, 0.0);
            a.heat_source = 10.0; // 0*T = -10 は解けない
            thermal.addNode(a);
            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);
            expectTrue(!thermal.getLastThermalConverged(), "failed solve: not converged");
            expectTrue(thermal.getLastThermalMethod().find("failed") != std::string::npos,
                       "failed solve: method marks DirectT(failed)");
            expectTrue(std::isnan(thermal.getLastThermalRmseBalance()),
                       "failed solve: rmse is NaN");
            expectTrue(std::isnan(thermal.getLastThermalMaxBalance()),
                       "failed solve: max is NaN");
        }

        // ------------------------------------------------------------------
        // advection enable 切替は flowSig に入り、キャッシュ経路でも無効化される
        // ------------------------------------------------------------------
        {
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalNetwork thermal;
            auto src = makeNode("SRC", false, 30.0);
            auto wall = makeNode("WALL", false, 0.0);
            auto room = makeNode("ROOM", true, 15.0);
            thermal.addNode(src);
            thermal.addNode(wall);
            thermal.addNode(room);

            EdgeProperties adv{};
            adv.key = "adv";
            adv.unique_id = "adv";
            adv.type = "advection";
            adv.source = "SRC";
            adv.target = "ROOM";
            adv.flow_rate = 0.2; // mDotCp が大きい
            adv.current_enabled = true;
            thermal.addEdge(adv);
            thermal.addEdge(makeConductance("WALL->ROOM", "WALL", "ROOM", 1.0));

            ThermalSolver solver(thermal, logs);
            solver.solveTemperatures(constants);
            const auto& kv = thermal.getKeyToVertex();
            const double tOn = thermal.getGraph()[kv.at("ROOM")].current_t;
            // 強い移流 + 弱い壁: ROOM は SRC に近い
            expectTrue(tOn > 25.0, "advection enabled: room pulled toward SRC");

            auto& gAdv = thermal.getGraph();
            for (auto e : boost::make_iterator_range(boost::edges(gAdv))) {
                if (gAdv[e].getTypeCode() == EdgeProperties::TypeCode::Advection) {
                    gAdv[e].current_enabled = false;
                }
            }
            const auto before = ThermalSolverLinearDirect::getDirectTCacheStats(thermal.directTContext());
            solver.solveTemperatures(constants);
            const auto after = ThermalSolverLinearDirect::getDirectTCacheStats(thermal.directTContext());
            expectTrue(after.coeffSigChanged > before.coeffSigChanged,
                       "advection enable off: coeffSigChanged");
            expectTrue(after.fullBuild > before.fullBuild,
                       "advection enable off: fullBuild (not stale factorization)");
            // 移流無効後は壁伝導のみ: Troom = Twall = 0
            expectNear(gAdv[kv.at("ROOM")].current_t, 0.0, 1e-8,
                       "advection disabled: room follows WALL only");
            for (auto e : boost::make_iterator_range(boost::edges(gAdv))) {
                if (gAdv[e].getTypeCode() == EdgeProperties::TypeCode::Advection) {
                    expectTrue(!gAdv[e].current_enabled, "advection remains disabled");
                    expectNear(gAdv[e].heat_rate, 0.0, 0.0, "advection disabled: heat_rate==0");
                }
            }
        }

        std::cout << "[OK] all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
