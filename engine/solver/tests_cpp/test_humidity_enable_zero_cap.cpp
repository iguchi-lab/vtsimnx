#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/range/iterator_range.hpp>

#include "core/humidity/humidity_solver.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_coupled_step.h"
#include "simulation_error.h"
#include "simulation_inner_coupling.h"
#include "simulation_timestep_state.h"
#include "types/common_types.h"

namespace {

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    if (!std::isfinite(actual)) throw std::runtime_error(msg + " (actual non-finite)");
    const double diff = std::abs(actual - expected);
    if (diff > tol) {
        std::ostringstream oss;
        oss << msg << " actual=" << actual << " expected=" << expected
            << " diff=" << diff << " tol=" << tol;
        throw std::runtime_error(oss.str());
    }
}

VertexProperties makeNode(const std::string& key, bool calcX, double V, double x0) {
    VertexProperties v{};
    v.key = key;
    v.type = "normal";
    v.calc_x = calcX;
    v.v = V;
    v.current_x = x0;
    v.current_w = x0;
    v.current_t = 20.0;
    return v;
}

EdgeProperties makeVent(const std::string& key,
                        const std::string& s,
                        const std::string& t,
                        double flow_m3s,
                        double gen = 0.0) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "fixed_flow";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.flow_rate = flow_m3s;
    e.current_vol = flow_m3s;
    e.vol = {flow_m3s};
    e.current_humidity_generation = gen;
    return e;
}

EdgeProperties makeMoistureLink(const std::string& key,
                                const std::string& s,
                                const std::string& t,
                                double k) {
    EdgeProperties e{};
    e.key = key;
    e.unique_id = key;
    e.type = "conductance";
    e.source = s;
    e.target = t;
    e.current_enabled = true;
    e.conductance = 1.0;
    e.moisture_conductance = k;
    return e;
}

SimulationConstants makeConstants() {
    SimulationConstants c{};
    c.logVerbosity = 0;
    c.humidityCalc = true;
    c.pressureCalc = false;
    c.temperatureCalc = false;
    c.timestep = 3600;
    c.humiditySolverTolerance = 1e-10;
    return c;
}

} // namespace

int main() {
    try {
        const auto constants = makeConstants();
        std::ostringstream logs;
        TimingList timings;

        // ------------------------------------------------------------------
        // disabled humidity_generation は 0
        // ------------------------------------------------------------------
        {
            auto src = makeNode("SRC", false, 0.0, 0.010);
            auto room = makeNode("ROOM", true, 100.0, 0.005);
            auto voidN = makeNode("VOID", false, 0.0, 0.0);
            std::vector<VertexProperties> nodes = {src, room, voidN};
            std::vector<EdgeProperties> vent = {
                makeVent("SRC->ROOM", "SRC", "ROOM", 0.0, 1.0e-4),
            };
            vent[0].current_enabled = false;
            std::vector<EdgeProperties> th;

            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes, vent, constants, logs);
            thermal.buildFromData(nodes, th, vent, constants, logs);
            humidity.invalidateCaches();

            // enable=false の枝でも flow/gen を明示
            for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
                ventNet.getGraph()[e].current_enabled = false;
                ventNet.getGraph()[e].current_humidity_generation = 1.0e-4;
                ventNet.getGraph()[e].flow_rate = 0.0;
            }

            const double xBefore = thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x;
            const auto stats = core::humidity::updateHumidityIfEnabled(
                constants, ventNet, thermal.getGraph(), static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "gen-disabled");
            expectTrue(stats.converged, "gen-disabled: converged");
            // 発湿なし・換気なし → 湿度はほぼ不変（容量ありで流入なし）
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x,
                       xBefore, 1e-12, "disabled humidity_generation does not change x");
        }

        // ------------------------------------------------------------------
        // disabled moisture_conductance は作用しない
        // ------------------------------------------------------------------
        {
            auto a = makeNode("A", true, 0.0, 0.010); // ゼロ容量
            a.moisture_capacity = 0.0;
            a.v = 0.0;
            auto b = makeNode("B", true, 0.0, 0.000);
            b.moisture_capacity = 0.0;
            b.v = 0.0;
            // 固定境界として calc_x=false の方が簡単だが、リンク無効なら両方保持
            std::vector<VertexProperties> nodes = {a, b};
            EdgeProperties link = makeMoistureLink("A-B", "A", "B", 0.05);
            link.current_enabled = false;
            std::vector<EdgeProperties> th = {link};
            std::vector<EdgeProperties> vent;

            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            SimulationConstants c = constants;
            c.temperatureCalc = true; // thermal branches を読む
            ventNet.buildFromData(nodes, vent, c, logs);
            thermal.buildFromData(nodes, th, vent, c, logs);
            humidity.invalidateCaches();
            for (auto e : boost::make_iterator_range(boost::edges(thermal.getGraph()))) {
                thermal.getGraph()[e].current_enabled = false;
                thermal.getGraph()[e].moisture_conductance = 0.05;
            }

            const double xA0 = thermal.getGraph()[thermal.getKeyToVertex().at("A")].current_x;
            const double xB0 = thermal.getGraph()[thermal.getKeyToVertex().at("B")].current_x;
            const auto stats = core::humidity::updateHumidityIfEnabled(
                c, ventNet, thermal.getGraph(), static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "link-disabled");
            expectTrue(stats.converged, "link-disabled: converged");
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("A")].current_x,
                       xA0, 1e-12, "disabled moisture link: A unchanged");
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("B")].current_x,
                       xB0, 1e-12, "disabled moisture link: B unchanged");
        }

        // ------------------------------------------------------------------
        // ゼロ容量 + moisture link の定常既知解
        // (k)(xA - xB)=0 with fixed B -> xA = xB when only link? 
        // A calc_x, B fixed: k*(xA-xB)=0 => xA=xB
        // ------------------------------------------------------------------
        {
            auto a = makeNode("A", true, 0.0, 0.020);
            a.moisture_capacity = 0.0;
            auto b = makeNode("B", false, 0.0, 0.005);
            b.moisture_capacity = 0.0;
            std::vector<VertexProperties> nodes = {a, b};
            std::vector<EdgeProperties> th = {makeMoistureLink("A-B", "A", "B", 0.02)};
            std::vector<EdgeProperties> vent;
            SimulationConstants c = constants;
            c.temperatureCalc = true;

            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes, vent, c, logs);
            thermal.buildFromData(nodes, th, vent, c, logs);
            humidity.invalidateCaches();

            const auto stats = core::humidity::updateHumidityIfEnabled(
                c, ventNet, thermal.getGraph(), static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "zero-cap-link");
            expectTrue(stats.converged && stats.updated, "zero-cap-link: updated");
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("A")].current_x,
                       0.005, 1e-10, "zero-cap + link: xA equals fixed xB");
        }

        // ------------------------------------------------------------------
        // ゼロ容量 + 発湿 + 排出の定常既知解
        // outSum*x - 0 = g  => x = g/outSum
        // ------------------------------------------------------------------
        {
            constexpr double rho = PhysicalConstants::DENSITY_DRY_AIR;
            constexpr double q = 0.1; // m3/s
            constexpr double g = 2.0e-4; // kg/s
            auto room = makeNode("ROOM", true, 0.0, 0.0);
            room.moisture_capacity = 0.0;
            room.v = 0.0;
            auto ext = makeNode("EXT", false, 0.0, 0.0);
            auto voidN = makeNode("VOID", false, 0.0, 0.0);
            std::vector<VertexProperties> nodes = {room, ext, voidN};
            std::vector<EdgeProperties> vent = {
                makeVent("VOID->ROOM", "VOID", "ROOM", 0.0, g),
                makeVent("ROOM->EXT", "ROOM", "EXT", q, 0.0),
            };
            // 供給空気湿度 0（VOID の x=0、流量0の gen 枝は流量なし）
            // 排出のみ outSum = rho*q, 流入なし, gen=g
            // (rho*q)*x = g => x = g/(rho*q)
            std::vector<EdgeProperties> th;
            SimulationConstants c = constants;

            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes, vent, c, logs);
            thermal.buildFromData(nodes, th, vent, c, logs);
            humidity.invalidateCaches();
            for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
                auto& ep = ventNet.getGraph()[e];
                if (ep.key.find("VOID") != std::string::npos) {
                    ep.flow_rate = 0.0;
                    ep.current_humidity_generation = g;
                } else {
                    ep.flow_rate = q;
                    ep.current_humidity_generation = 0.0;
                }
                ep.current_enabled = true;
            }

            const auto stats = core::humidity::updateHumidityIfEnabled(
                c, ventNet, thermal.getGraph(), static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "zero-cap-gen");
            expectTrue(stats.converged && stats.updated, "zero-cap-gen: updated");
            const double expected = g / (rho * q);
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x,
                       expected, 1e-10, "zero-cap + gen + exhaust: x=g/(rho q)");
        }

        // ------------------------------------------------------------------
        // 特異系: 未収束時にグラフを変更しない
        // ゼロ容量ノードへ流入のみ・流出なし → diag=0 で分解失敗
        // ------------------------------------------------------------------
        {
            auto room = makeNode("ROOM", true, 0.0, 0.012);
            room.moisture_capacity = 0.0;
            room.v = 0.0;
            auto ext = makeNode("EXT", false, 0.0, 0.010);
            std::vector<VertexProperties> nodes = {room, ext};
            std::vector<EdgeProperties> vent = {
                makeVent("EXT->ROOM", "EXT", "ROOM", 0.1, 0.0),
            };
            std::vector<EdgeProperties> th;
            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes, vent, constants, logs);
            thermal.buildFromData(nodes, th, vent, constants, logs);
            humidity.invalidateCaches();
            for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
                ventNet.getGraph()[e].flow_rate = 0.1;
                ventNet.getGraph()[e].current_enabled = true;
            }

            const double xBefore = thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x;
            const auto stats = core::humidity::updateHumidityIfEnabled(
                constants, ventNet, thermal.getGraph(), static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "singular");
            expectTrue(!stats.converged, "singular: not converged");
            expectTrue(!stats.updated, "singular: not updated");
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x,
                       xBefore, 0.0, "singular: graph unchanged");
        }

        // ------------------------------------------------------------------
        // ゼロ容量 + 発湿のみ（流出なし）→ 有限定常解なし → 未収束・グラフ不変
        // ------------------------------------------------------------------
        {
            constexpr double g = 1.0e-4;
            auto room = makeNode("ROOM", true, 0.0, 0.008);
            room.moisture_capacity = 0.0;
            room.v = 0.0;
            auto voidN = makeNode("VOID", false, 0.0, 0.0);
            std::vector<VertexProperties> nodes = {room, voidN};
            std::vector<EdgeProperties> vent = {
                makeVent("VOID->ROOM", "VOID", "ROOM", 0.0, g),
            };
            std::vector<EdgeProperties> th;
            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes, vent, constants, logs);
            thermal.buildFromData(nodes, th, vent, constants, logs);
            humidity.invalidateCaches();
            for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
                ventNet.getGraph()[e].flow_rate = 0.0;
                ventNet.getGraph()[e].current_humidity_generation = g;
                ventNet.getGraph()[e].current_enabled = true;
            }

            const double xBefore = thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x;
            const auto stats = core::humidity::updateHumidityIfEnabled(
                constants, ventNet, thermal.getGraph(),
                static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "zero-cap-gen-only");
            expectTrue(!stats.converged, "zero-cap-gen-only: not converged");
            expectTrue(!stats.updated, "zero-cap-gen-only: not updated");
            expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x,
                       xBefore, 0.0, "zero-cap-gen-only: graph unchanged");
        }

        // ------------------------------------------------------------------
        // 同一 Graph オブジェクト・同規模再構築でも topologyRevision でキャッシュ無効化
        // （humidity.invalidateCaches() を呼ばない）
        // ------------------------------------------------------------------
        {
            SimulationConstants c = constants;
            c.temperatureCalc = true;
            auto a1 = makeNode("A", true, 0.0, 0.020);
            a1.moisture_capacity = 0.0;
            auto b1 = makeNode("B", false, 0.0, 0.005);
            b1.moisture_capacity = 0.0;
            std::vector<VertexProperties> nodes1 = {a1, b1};
            std::vector<EdgeProperties> th1 = {makeMoistureLink("A-B", "A", "B", 0.02)};
            std::vector<EdgeProperties> vent;

            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes1, vent, c, logs);
            thermal.buildFromData(nodes1, th1, vent, c, logs);
            // 明示 invalidate なしで初回実行 → キャッシュ構築
            {
                const auto stats = core::humidity::updateHumidityIfEnabled(
                    c, ventNet, thermal.getGraph(),
                    static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                    {}, logs, timings, "cache-rev-1");
                expectTrue(stats.converged && stats.updated, "cache-rev-1: updated");
                expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("A")].current_x,
                           0.005, 1e-10, "cache-rev-1: xA");
            }
            const auto rev1 = thermal.getTopologyRevision();

            // 同数ノード・枝だがキーを入れ替え、固定境界湿度を変える
            auto a2 = makeNode("A", true, 0.0, 0.030);
            a2.moisture_capacity = 0.0;
            auto b2 = makeNode("B", false, 0.0, 0.015);
            b2.moisture_capacity = 0.0;
            std::vector<VertexProperties> nodes2 = {a2, b2};
            std::vector<EdgeProperties> th2 = {makeMoistureLink("A-B", "A", "B", 0.02)};
            ventNet.buildFromData(nodes2, vent, c, logs);
            thermal.buildFromData(nodes2, th2, vent, c, logs);
            expectTrue(thermal.getTopologyRevision() != rev1, "cache-rev: topologyRevision bumps");
            // humidity.invalidateCaches() を呼ばない — revision で検出されること
            {
                const auto stats = core::humidity::updateHumidityIfEnabled(
                    c, ventNet, thermal.getGraph(),
                    static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                    {}, logs, timings, "cache-rev-2");
                expectTrue(stats.converged && stats.updated, "cache-rev-2: updated");
                expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("A")].current_x,
                           0.015, 1e-10, "cache-rev-2: uses rebuilt graph, not stale cache");
            }
        }

        // ------------------------------------------------------------------
        // 負風量の湿度移流（向き反転）
        // ------------------------------------------------------------------
        {
            auto src = makeNode("SRC", false, 0.0, 0.010);
            auto room = makeNode("ROOM", true, 100.0, 0.0);
            auto ext = makeNode("EXT", false, 0.0, 0.0);
            std::vector<VertexProperties> nodes = {src, room, ext};
            // ROOM->SRC に負流量 = SRC->ROOM 正流量
            std::vector<EdgeProperties> vent = {
                makeVent("ROOM->SRC", "ROOM", "SRC", -0.1, 0.0),
                makeVent("ROOM->EXT", "ROOM", "EXT", 0.1, 0.0),
            };
            std::vector<EdgeProperties> th;
            VentilationNetwork ventNet;
            ThermalNetwork thermal;
            HumidityNetwork humidity;
            ventNet.buildFromData(nodes, vent, constants, logs);
            thermal.buildFromData(nodes, th, vent, constants, logs);
            humidity.invalidateCaches();
            for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
                auto& ep = ventNet.getGraph()[e];
                if (ep.key.find("SRC") != std::string::npos) ep.flow_rate = -0.1;
                else ep.flow_rate = 0.1;
                ep.current_enabled = true;
            }

            const auto stats = core::humidity::updateHumidityIfEnabled(
                constants, ventNet, thermal.getGraph(), static_cast<const ThermalNetwork&>(thermal).nodeStateView(), humidity,
                {}, logs, timings, "neg-flow");
            expectTrue(stats.converged && stats.updated, "neg-flow: updated");
            // 1ステップで SRC 湿度側へ寄る（完全平衡ではない）
            expectTrue(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x > 0.0,
                       "neg-flow: room humidity increases from SRC");
        }

        // ------------------------------------------------------------------
        // runDecoupledHumidityStep / runInnerCoupling が HumidityNotConverged を送出
        // ------------------------------------------------------------------
        {
            constexpr double g = 1.0e-4;
            auto room = makeNode("ROOM", true, 0.0, 0.008);
            room.moisture_capacity = 0.0;
            room.v = 0.0;
            auto voidN = makeNode("VOID", false, 0.0, 0.0);
            std::vector<VertexProperties> nodes = {room, voidN};
            std::vector<EdgeProperties> vent = {
                makeVent("VOID->ROOM", "VOID", "ROOM", 0.0, g),
            };
            std::vector<EdgeProperties> th;

            auto setupNetworks = [&](VentilationNetwork& ventNet, ThermalNetwork& thermal,
                                     HumidityNetwork& humidity, const SimulationConstants& c) {
                ventNet.buildFromData(nodes, vent, c, logs);
                thermal.buildFromData(nodes, th, vent, c, logs);
                humidity.invalidateCaches();
                for (auto e : boost::make_iterator_range(boost::edges(ventNet.getGraph()))) {
                    ventNet.getGraph()[e].flow_rate = 0.0;
                    ventNet.getGraph()[e].current_humidity_generation = g;
                    ventNet.getGraph()[e].current_enabled = true;
                }
            };

            // 非連成経路
            {
                SimulationConstants c = constants;
                c.humidityCalc = true;
                c.moistureCouplingEnabled = false;
                c.pressureCalc = false;
                c.temperatureCalc = false;
                VentilationNetwork ventNet;
                ThermalNetwork thermal;
                HumidityNetwork humidity;
                setupNetworks(ventNet, thermal, humidity, c);
                const double xBefore =
                    thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x;

                simulation::InnerCouplingContext ctx{
                    ventNet, thermal, humidity, c, logs, timings, "decoupled-hum-fail"};
                const auto initial =
                    simulation::detail::captureTimestepInitialState(thermal, c.humidityCalc);
                CoupledStepData step;
                bool threw = false;
                try {
                    simulation::runDecoupledHumidityStep(ctx, initial, step, 0);
                } catch (const simulation::Error& e) {
                    threw = true;
                    expectTrue(e.code() == simulation::ErrorCode::HumidityNotConverged,
                               "decoupled: HumidityNotConverged code");
                    expectTrue(std::string(simulation::toErrorCodeString(e.code())) ==
                                   "humidity_not_converged",
                               "decoupled: api error string");
                }
                expectTrue(threw, "decoupled: must throw");
                expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x,
                           xBefore, 0.0, "decoupled: graph unchanged");
            }

            // 連成経路（湿気のみでも内側ループで湿気ソルバが走る）
            {
                SimulationConstants c = constants;
                c.humidityCalc = true;
                c.moistureCouplingEnabled = true;
                c.pressureCalc = false;
                c.temperatureCalc = false;
                c.logVerbosity = 1; // 停止ログ経路も通す
                VentilationNetwork ventNet;
                ThermalNetwork thermal;
                HumidityNetwork humidity;
                setupNetworks(ventNet, thermal, humidity, c);
                const double xBefore =
                    thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x;

                simulation::InnerCouplingContext ctx{
                    ventNet, thermal, humidity, c, logs, timings, "coupled-hum-fail", nullptr};
                const auto initial =
                    simulation::detail::captureTimestepInitialState(thermal, c.humidityCalc);
                CoupledStepData step;
                int totalIterations = 0;
                simulation::detail::SeparatedHeatSources heatSources;
                bool threw = false;
                try {
                    simulation::runInnerCoupling(ctx, true, 0, initial, step, totalIterations,
                                                 heatSources, /*forceMinTwo=*/true);
                } catch (const simulation::Error& e) {
                    threw = true;
                    expectTrue(e.code() == simulation::ErrorCode::HumidityNotConverged,
                               "coupled: HumidityNotConverged code");
                    expectTrue(std::string(simulation::toErrorCodeString(e.code())) ==
                                   "humidity_not_converged",
                               "coupled: api error string");
                }
                expectTrue(threw, "coupled: must throw");
                expectNear(thermal.getGraph()[thermal.getKeyToVertex().at("ROOM")].current_x,
                           xBefore, 0.0, "coupled: graph unchanged");
                expectTrue(logs.str().find("湿気ソルバ未収束(停止)") != std::string::npos,
                           "coupled: stop log message");
            }
        }

        std::cout << "[OK] all tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
