#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "aircon/aircon_controller.h"
#include "aircon/aircon_airflow.h"
#include "aircon/aircon_capacity.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_network_utils.h"
#include "aircon/aircon_operation_mode.h"
#include "archenv/include/archenv.h"
#include "core/thermal/thermal_moist_air.h"
#include "core/thermal/thermal_solver_linear_direct.h"
#include "network/thermal_network.h"
#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "simulation_error.h"

#include <unordered_map>
#include <utility>
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

struct CallRecord {
    std::string mode;
    acmodel::InputData input;
};

class FakeAirconModel final : public acmodel::AirconSpec {
public:
    explicit FakeAirconModel(int* callCount, std::vector<CallRecord>* history)
        : acmodel::AirconSpec(nlohmann::json::object()), callCount_(callCount), history_(history) {}

    acmodel::COPResult estimateCOP(const std::string& mode, const acmodel::InputData& inputdata) override {
        if (callCount_) (*callCount_)++;
        if (history_) history_->push_back(CallRecord{mode, inputdata});
        acmodel::COPResult r;
        r.COP = 4.0;
        r.power = 0.2; // [kW]
        r.valid = true;
        return r;
    }

    double calculatePowerConsumption(double, double, double) const override { return 0.0; }
    double calculateCoolingCapacity(double, double, double) const override { return 0.0; }
    bool isValidOperatingCondition(double, double) const override { return true; }
    std::string getModelName() const override { return "FAKE"; }
    nlohmann::json getModelParameters() const override { return nlohmann::json::object(); }

private:
    int* callCount_ = nullptr;
    std::vector<CallRecord>* history_ = nullptr;
};

static VertexProperties makeNode(const std::string& key, const std::string& type, double t) {
    VertexProperties v{};
    v.key = key;
    v.type = type;
    v.current_t = t;
    v.current_mode = "COOLING";
    v.current_requested_pre_temp = 24.0;
    v.current_pre_temp = 24.0;
    v.on = false;
    return v;
}

static void setRequestedAndEffective(VertexProperties& v, double t) {
    v.current_requested_pre_temp = t;
    v.current_pre_temp = t;
}

static nlohmann::json makeAcSpecWithMax(double coolingMaxKw, double heatingMaxKw) {
    return nlohmann::json{
        {"Q",
         {
             {"cooling", {{"max", coolingMaxKw}}},
             {"heating", {{"max", heatingMaxKw}}},
         }},
    };
}

// Issue #24: locate the actual dry/wet transition without duplicating its formula.
void testCoilDewPointContinuity() {
    struct CoilCase { double t, sensible, flow, faceArea, surfaceArea; };
    const std::vector<CoilCase> cases = {
        {27.5, 4725.0, 0.3, 0.133, 4.84},
        {30.0, 3000.0, 0.2, 0.133, 4.84},
        {24.0, 1500.0, 0.15, 0.133, 4.84},
        {27.5, 4725.0, 0.3, 0.2, 6.0},
    };
    for (const auto& c : cases) {
        for (const std::string method : {"coil_aoaf", "aoaf", "literature"}) {
            for (const bool moist : {false, true}) {
                VertexProperties node{};
                node.current_pre_rh = std::numeric_limits<double>::quiet_NaN();
                node.ac_spec = {{"latent_method", method}, {"Af", c.faceArea}, {"Ao", c.surfaceArea}};
                AirconValidationData data{};
                data.indoorTemp = c.t;
                data.airconTemp = c.t - c.sensible /
                    (archenv::SPECIFIC_HEAT_AIR * archenv::DENSITY_DRY_AIR * c.flow);
                auto evaluate = [&](double x) {
                    data.indoorX = x;
                    return aircon::latent::estimateLatentProcess(
                        data, OperationMode::Cooling, c.sensible, c.flow, node, moist);
                };
                const double wetX = archenv::absolute_humidity(c.t, 100.0);
                expectNear(evaluate(0.0).latentHeatCapacity, 0.0, 1e-10, "dry air has no latent load");
                expectTrue(evaluate(wetX).latentHeatCapacity > 1.0, "wet side of coil onset exists");
                double low = 0.0, high = wetX;
                for (int i = 0; i < 60; ++i) {
                    const double mid = 0.5 * (low + high);
                    if (evaluate(mid).latentHeatCapacity > 1e-9) high = mid;
                    else low = mid;
                }
                constexpr double epsilonX = 1e-8;
                const auto below = evaluate(low - epsilonX);
                const auto above = evaluate(high + epsilonX);
                expectNear(below.latentHeatCapacity, 0.0, 1e-7, "below dew boundary stays dry");
                expectTrue(above.latentHeatCapacity > 0.0 && above.latentHeatCapacity < 0.1,
                           "latent load must start continuously at zero: " + method +
                           " above_W=" + std::to_string(above.latentHeatCapacity));
                expectTrue(std::abs(above.supplyX - below.supplyX) < 1e-7,
                           "supply humidity must be continuous across dew boundary");
                const double saturationX = archenv::absolute_humidity(data.airconTemp, 100.0);
                expectTrue(std::abs(evaluate(saturationX + epsilonX).latentHeatCapacity -
                                    evaluate(saturationX - epsilonX).latentHeatCapacity) < 0.1,
                           "latent load stays continuous across outlet saturation limit");
                double previousLoad = 0.0;
                for (int i = 0; i <= 64; ++i) {
                    const double x = wetX * i / 64.0;
                    const auto result = evaluate(x);
                    expectTrue(std::isfinite(result.latentHeatCapacity) && result.latentHeatCapacity >= 0.0,
                               "coil latent load is finite and nonnegative");
                    expectTrue(result.latentHeatCapacity + 1e-7 >= previousLoad,
                               "coil latent load increases with inlet humidity");
                    expectTrue(result.supplyX >= 0.0 && result.supplyX <= x,
                               "cooling coil cannot create water or return negative humidity");
                    expectNear(result.condensationRateKgPerS,
                               archenv::DENSITY_DRY_AIR * c.flow * (x - result.supplyX), 1e-12,
                               "condensate matches air-side water removal");
                    previousLoad = result.latentHeatCapacity;
                }
                if (method == "coil_aoaf" && !moist) {
                    std::cout << "coil onset: T=" << c.t << " sensible_W=" << c.sensible
                              << " x=" << high << " latent_just_above_W=" << above.latentHeatCapacity << "\n";
                }
            }
        }
    }
}

} // namespace

int main() {
    expectNear(archenv::saturation_vapor_pressure(100.0), 101324.6, 1e-7,
               "saturation reference matches Python report conversion");
    testCoilDewPointContinuity();
    // 正味流量: 正方向と逆方向の両方があれば direct - reverse
    {
        FlowRateMap flows;
        flows[{"IN", "AC"}] = 0.0;   // スケジュール0の正方向キー
        flows[{"AC", "IN"}] = 0.25;  // 逆向き表現の実流量
        expectNear(aircon::network_utils::getFlowRate(flows, "IN", "AC"), -0.25, 1e-12,
                   "net flow should be direct - reverse");
        expectNear(aircon::network_utils::getFlowRate(flows, "AC", "IN"), 0.25, 1e-12,
                   "opposite query should flip sign of net flow");
    }

    // 還気+吹出の双方向固定流量でも処理風量は還気枝の絶対値
    {
        FlowRateMap loop;
        loop[{"IN", "AC"}] = 0.2;
        loop[{"AC", "IN"}] = 0.2;
        expectNear(aircon::network_utils::getFlowRate(loop, "IN", "AC"), 0.0, 1e-12,
                   "recirculation net flow should cancel");
        expectNear(aircon::network_utils::getAirconProcessFlowRate(loop, "IN", "AC"), 0.2, 1e-12,
                   "process flow should keep return-duct magnitude");
    }

    ThermalNetwork thermal;

    // 必要なノード（outside/in/aircon）
    thermal.addNode(makeNode("OUT", "normal", 35.0));
    thermal.addNode(makeNode("IN", "normal", 26.0));
    thermal.addNode(makeNode("A", "aircon", 20.0));
    thermal.addNode(makeNode("B", "aircon", 20.0));

    // aircon ノードの関連キー
    {
        auto& a = thermal.getNode("A");
        a.outside_node = "OUT";
        a.in_node = "IN";
        a.set_node.clear();
        a.on = false;
    }
    {
        auto& b = thermal.getNode("B");
        b.outside_node = "OUT";
        b.in_node = "IN";
        b.set_node.clear();
        b.on = true;
    }
    {
        auto& out = thermal.getNode("OUT");
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        out.current_x = 0.018;
        in.current_x = 0.020;
        b.current_x = 0.0;
    }

    // flowRates: IN -> aircon の流量を入れて、heatCapacity が 0 にならないようにする
    FlowRateMap flowRates;
    flowRates[{"IN", "A"}] = 0.1;
    flowRates[{"IN", "B"}] = 0.1;

    AirconController controller;
    int calls = 0;
    std::vector<CallRecord> history;

    // 逆順で登録しても、getAirconKeys() は昇順で返ること
    controller.registerModelForTesting("B", std::make_unique<FakeAirconModel>(&calls, &history));
    controller.registerModelForTesting("A", std::make_unique<FakeAirconModel>(&calls, &history));

    const auto& keys = controller.getAirconKeys();
    expectTrue(keys.size() == 2, "keys size");
    if (keys.size() == 2) {
        expectTrue(keys[0] == "A" && keys[1] == "B", "keys are sorted");
    }

    // OFF(A) は 0.0、ON(B) は estimateCOP が呼ばれて値が入る
    {
        calls = 0;
        history.clear();
        auto powerW = controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(powerW.size() == 2, "power size");
        if (powerW.size() == 2) {
            expectNear(powerW[0], 0.0, 0.0, "A power=0 when off");
            expectNear(powerW[1], 200.0, 1e-9, "B power=0.2kW -> 200W");
        }
        expectTrue(calls == 1, "estimateCOP called only for ON aircon (power)");
        expectTrue(!history.empty(), "history has one entry");
        if (!history.empty()) {
            const auto& in = history.back().input;
            expectTrue(in.Q_L > 0.0, "Q_L should be positive in humid cooling case");
            expectNear(in.Q, in.Q_S + in.Q_L, 1e-6, "Q should equal Q_S + Q_L");
        }
    }
    // 潜熱4方式目(coil_aoaf): 計算が有効になり、Af/Ao を変えると Q_L が変化すること
    {
        auto& b = thermal.getNode("B");
        b.on = true;
        b.current_mode = "COOLING";
        b.current_t = 20.0;
        thermal.getNode("IN").current_x = 0.014;
        b.ac_spec = nlohmann::json{
            {"latent_method", "coil_aoaf"},
            {"Af", 0.133},
            {"Ao", 2.42},
        };

        calls = 0;
        history.clear();
        (void)controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(calls == 1, "coil_aoaf: estimateCOP should be called for ON aircon");
        expectTrue(!history.empty(), "coil_aoaf: history should have one entry");
        double qlDefault = 0.0;
        if (!history.empty()) {
            qlDefault = history.back().input.Q_L;
            expectTrue(qlDefault >= 0.0, "coil_aoaf: Q_L should be non-negative");
        }

        // 飽和制約が支配しない条件ではコイル面積が除湿量に反映される
        b.ac_spec = nlohmann::json{
            {"latent_method", "coil_aoaf"},
            {"Af", 0.133},
            {"Ao", 4.84},
        };
        calls = 0;
        history.clear();
        (void)controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(calls == 1, "coil_aoaf(Ao=4.84): estimateCOP should be called");
        expectTrue(!history.empty(), "coil_aoaf(Ao=4.84): history should have one entry");
        if (!history.empty()) {
            const double qlLargeAo = history.back().input.Q_L;
            expectTrue(qlLargeAo >= 0.0, "coil_aoaf: Q_L should stay non-negative when Ao changes");
            expectTrue(std::abs(qlLargeAo - qlDefault) > 1e-9,
                       "coil_aoaf: Q_L should change when Ao changes");
        }
    }
    {
        calls = 0;
        history.clear();
        auto cop = controller.calculateCOPValues(thermal, flowRates, std::cout);
        expectTrue(cop.size() == 2, "cop size");
        if (cop.size() == 2) {
            expectNear(cop[0], 0.0, 0.0, "A COP=0 when off");
            expectNear(cop[1], 4.0, 1e-12, "B COP=4.0");
        }
        expectTrue(calls == 1, "estimateCOP called only for ON aircon (cop)");
    }

    // ON でも処理熱量 0 なら COP 推定を呼ばず電力・COP=0（エラーにしない）
    {
        FlowRateMap zeroFlow;
        zeroFlow[{"IN", "A"}] = 0.0;
        zeroFlow[{"IN", "B"}] = 0.0;
        calls = 0;
        history.clear();
        auto powerW = controller.calculatePowerValues(thermal, zeroFlow, std::cout);
        auto cop = controller.calculateCOPValues(thermal, zeroFlow, std::cout);
        expectTrue(powerW.size() == 2 && cop.size() == 2, "zero-load power/cop size");
        if (powerW.size() == 2 && cop.size() == 2) {
            expectNear(powerW[0], 0.0, 0.0, "A power=0 (off)");
            expectNear(powerW[1], 0.0, 0.0, "B power=0 when load=0");
            expectNear(cop[0], 0.0, 0.0, "A COP=0 (off)");
            expectNear(cop[1], 0.0, 0.0, "B COP=0 when load=0");
        }
        expectTrue(calls == 0, "estimateCOP not called when load=0");
        expectTrue(history.empty(), "history empty when load=0");
    }

    // sensibleHeatCapacity も OFF は 0、ON のみ正値になること
    {
        auto sensible = controller.collectAirconDataValues(thermal, flowRates, "sensibleHeatCapacity");
        expectTrue(sensible.size() == 2, "sensible heat size");
        if (sensible.size() == 2) {
            expectNear(sensible[0], 0.0, 0.0, "A sensible heat=0 when off");
            expectTrue(sensible[1] > 0.0, "B sensible heat > 0 when on");
        }
    }

    // 出力収集は current_x を変更しない
    {
        auto& b = thermal.getNode("B");
        b.on = true;
        b.current_mode = "COOLING";
        b.current_t = 14.0;
        b.current_x = 0.011111;
        b.ac_spec = nlohmann::json{{"latent_method", "rh95"}};
        thermal.getNode("IN").current_x = 0.020;
        thermal.getNode("IN").current_t = 27.0;
        const double xBefore = b.current_x;
        (void)controller.collectAirconDataValues(thermal, flowRates, "sensibleHeatCapacity");
        (void)controller.collectAirconDataValues(thermal, flowRates, "latentHeatCapacity");
        expectNear(b.current_x, xBefore, 0.0, "collectAirconDataValues must not mutate current_x");
    }

    // OFF/heating humidity belongs to the implicit mixing equation. Controller
    // updates must not overwrite a solved value with a stale inlet snapshot.
    {
        auto& b = thermal.getNode("B");
        auto& in = thermal.getNode("IN");
        b.on = false;
        b.current_mode = "OFF";
        b.in_node = "IN";
        b.current_x = 0.008;
        b.aircon_moisture_removal_kg_s = 0.001;
        in.current_x = 0.020;
        bool supplyChanged = false;
        (void)controller.controlAllAircons(thermal, 0.5, std::cout, &supplyChanged, 1e-9);
        expectTrue(humiditySolveNode(b), "OFF is an algebraic humidity unknown");
        expectTrue(!supplyChanged, "OFF does not write a fixed humidity boundary");
        expectNear(b.current_x, 0.008, 0.0, "control preserves the current solved humidity");
        expectNear(b.aircon_moisture_removal_kg_s, 0.0, 0.0, "OFF clears condensation");
        b.on = true;
        b.current_mode = "HEATING";
        controller.syncHumidityBoundariesBeforeSolve(thermal);
        expectTrue(humiditySolveNode(b), "heating is an algebraic humidity unknown");
        expectNear(b.current_x, 0.008, 0.0, "pre-solve sync does not copy stale inlet humidity");
        b.current_mode = "COOLING";
        b.ac_spec = nlohmann::json{{"latent_method", "coil_aoaf"}};
        expectTrue(humiditySolveNode(b), "inactive cooling is a mixing unknown");
        aircon::latent::LatentProcessResult loads{};
        loads.supplyX = b.current_x - 5e-5;
        loads.condensationRateKgPerS = 1e-6;
        expectTrue(aircon::latent::applySupplyHumidityToAirconNode(thermal, "B", loads, 1e-9),
                   "a 5e-5 humidity defect cannot be hidden by a tolerance floor");
        const double acceptedX = b.current_x;
        loads.supplyX = acceptedX + 5e-10;
        expectTrue(!aircon::latent::applySupplyHumidityToAirconNode(thermal, "B", loads, 1e-9),
                   "an in-tolerance boundary needs no re-solve");
        expectNear(b.current_x, acceptedX, 0.0, "no re-solve means no boundary mutation");
    }

    // Ideal RH refers to the requested controlled-room temperature, even when
    // return air and the capacity-relaxed temperature are different.
    {
        AirconValidationData vd{};
        vd.indoorTemp = 30.0;
        vd.indoorX = 0.020;
        vd.airconTemp = 20.0;
        vd.setTemp = 28.0;
        VertexProperties ac{};
        ac.current_pre_rh = 60.0;
        ac.current_requested_pre_temp = 24.0;
        ac.ac_spec = {{"latent_method", "rh95"}};
        const double target = archenv::absolute_humidity(24.0, 60.0);
        const auto over = aircon::latent::estimateLatentProcess(
            vd, OperationMode::Cooling, 1000.0, 0.2, ac);
        expectNear(over.supplyX, target, 1e-12, "ideal uses requested room temperature");
        expectNear(over.sensibleHeatCapacity, 1000.0, 0.0, "latent does not change sensible load");

        // Below the target, do not fall back to another dehumidification model.
        ac.current_requested_pre_temp = 30.0;
        ac.current_pre_rh = 80.0;
        vd.indoorX = archenv::absolute_humidity(20.0, 97.0);
        const auto dry = aircon::latent::estimateLatentProcess(
            vd, OperationMode::Cooling, 1000.0, 0.2, ac);
        expectNear(dry.supplyX, vd.indoorX, 1e-12, "ideal below target does not use rh95");
        expectNear(dry.condensationRateKgPerS, 0.0, 0.0, "ideal never humidifies");

        vd.airconTemp = 12.0;
        const auto cold = aircon::latent::estimateLatentProcess(
            vd, OperationMode::Cooling, 1000.0, 0.2, ac);
        expectTrue(cold.saturationLimited && cold.supplyRhPercent <= 100.0,
                   "saturation overrides an unreachable ideal target");
        expectNear(cold.supplyX, cold.supplyXUpperBound, 1e-12, "minimum extra condensation");
    }

    // Every model must honor the sensible outlet temperature and derive water
    // and latent energy from the same final (non-supersaturated) state.
    for (const std::string method : {"rh95", "bf", "coil_aoaf", "none", "ideal"}) {
        for (bool moist : {false, true}) {
            AirconValidationData vd{};
            vd.indoorTemp = 30.0;
            vd.indoorX = 0.024;
            vd.airconTemp = 12.0;
            VertexProperties ac{};
            ac.current_requested_pre_temp = 27.0;
            if (method == "ideal") ac.current_pre_rh = 60.0;
            ac.ac_spec = {{"latent_method", method}};
            const auto loads = aircon::latent::estimateLatentProcess(
                vd, OperationMode::Cooling, 4300.0, 0.2, ac, moist);
            const double mass = archenv::DENSITY_DRY_AIR * 0.2 * (vd.indoorX - loads.supplyX);
            expectTrue(loads.supplyRhPercent <= 100.0 && loads.supplyX >= 0.0,
                       "all cooling methods remain unsaturated: " + method);
            expectNear(loads.condensationRateKgPerS, mass, 1e-12, "condensate balances water");
            if (moist) {
                expectNear(aircon::latent::totalHeatCapacity(loads),
                    thermal_moist_air::processedEnthalpyHeatW(
                        vd.indoorTemp, vd.indoorX, vd.airconTemp, loads.supplyX, 0.2, false),
                    1e-8, "enthalpy uses final outlet humidity");
            } else {
                expectNear(loads.sensibleHeatCapacity, 4300.0, 0.0, "sensible has priority");
                expectNear(loads.latentHeatCapacity, mass * archenv::vapor_latent_heat(12.0),
                           1e-8, "latent energy uses final condensate");
            }
            auto& b = thermal.getNode("B");
            b.on = true;
            b.current_mode = "COOLING";
            b.current_x = loads.supplyXUpperBound + 5e-5;
            expectTrue(aircon::latent::applySupplyHumidityToAirconNode(thermal, "B", loads, 1e-3),
                       "coarse tolerance cannot accept supersaturation");
            expectTrue(b.current_x <= loads.supplyXUpperBound, "boundary applies hard saturation cap");
            b.current_x = loads.supplyX - 1e-5;
            expectTrue(aircon::latent::supplyHumidityNeedsUpdate(b, loads, 1e-3),
                       "coarse tolerance cannot retain a stale supply boundary");
        }
    }

    // 潜熱フィードバック: 冷房時に in_node へ負の heat_source が入ること
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 27.0;
        in.current_x = 0.020;
        b.current_t = 20.0;
        b.current_mode = "COOLING";
        b.on = true;
        in.heat_source = 0.0;
        const auto stats = controller.applyLatentFeedbackToThermal(thermal, flowRates, 1.0, std::cout);
        expectTrue(stats.maxAppliedHeatW > 0.0, "latent feedback should apply non-zero heat");
        expectTrue(in.heat_source < 0.0, "latent feedback should be negative heat source at in_node");
    }

    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 18.0;
        b.current_t = 24.0;
        b.current_mode = "HEATING";
        setRequestedAndEffective(b, 26.0);
        b.on = true;
        b.ac_spec = makeAcSpecWithMax(3.3, 0.5);
        b.initializeAirconSpec();

        VentilationNetwork vent;
        SimulationConstants constants{};
        std::ostringstream logs;
        int totalIterations = 0;
        const bool adjusted = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);

        expectTrue(adjusted, "heating over-capacity should trigger adjustment");
        expectTrue(b.current_pre_temp < 26.0, "heating setpoint should decrease");
        expectTrue(b.current_pre_temp > 18.0, "heating setpoint should stay above inlet temp");
        expectNear(b.current_requested_pre_temp, 26.0, 1e-12,
                   "requested setpoint must stay at schedule value");
        expectTrue(b.aircon_control_state == AirconControlState::CapacityLimited,
                   "over-capacity should mark CapacityLimited");
    }

    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 30.0;
        b.current_t = 22.0;
        b.current_mode = "COOLING";
        setRequestedAndEffective(b, 24.0);
        b.on = true;
        b.ac_spec = makeAcSpecWithMax(0.5, 5.4);
        b.initializeAirconSpec();

        VentilationNetwork vent;
        SimulationConstants constants{};
        std::ostringstream logs;
        int totalIterations = 0;
        const bool adjusted = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);

        expectTrue(adjusted, "cooling over-capacity should trigger adjustment");
        expectTrue(b.current_pre_temp > 24.0, "cooling setpoint should increase");
        expectTrue(b.current_pre_temp < 30.0, "cooling setpoint should stay below inlet temp");
        expectNear(b.current_requested_pre_temp, 24.0, 1e-12,
                   "cooling requested setpoint must stay at schedule value");
    }

    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 18.0;
        b.current_t = 24.0;
        b.current_mode = "HEATING";
        setRequestedAndEffective(b, 26.0);
        b.on = true;
        b.ac_spec = nlohmann::json{
            {"Q", {{"cooling", {{"rtd", 2.2}}}, {"heating", {{"rtd", 2.5}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        SimulationConstants constants{};
        std::ostringstream logs;
        int totalIterations = 0;
        const bool adjusted = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);

        expectTrue(!adjusted, "missing Q.max should skip capacity adjustment");
        expectNear(b.current_pre_temp, 26.0, 1e-12, "setpoint unchanged when Q.max is absent");
    }

    // DUCT_CENTRAL: Q.max が無く Q.mid のみの場合も能力上限として扱う（最終フォールバック）
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 18.0;
        b.current_t = 24.0;
        b.current_mode = "HEATING";
        setRequestedAndEffective(b, 26.0);
        b.on = true;
        b.ac_spec = nlohmann::json{
            {"Q", {{"cooling", {{"mid", 2.0}}}, {"heating", {{"mid", 0.5}}}}},  // kW; mid only
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        SimulationConstants constants{};
        std::ostringstream logs;
        int totalIterations = 0;
        const bool adjusted = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);

        expectTrue(adjusted, "Q.mid only (no max/rtd) should still apply capacity limit");
        expectTrue(b.current_pre_temp < 26.0, "heating setpoint should decrease when over mid capacity");
        expectTrue(logs.str().find("Q.heating.mid") != std::string::npos,
                   "source label should indicate mid fallback");
    }

    // DUCT_CENTRAL: max 無し・rtd と mid がある場合は定格 rtd を上限にする（mid ではない）
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 18.0;
        b.current_t = 24.0;
        b.current_mode = "HEATING";
        setRequestedAndEffective(b, 26.0);
        b.on = true;
        // ΔT=6K, V=0.1 → Q≈ρ·cp·V·ΔT ≈ 724 W。mid=0.5kW なら超過、rtd=7.2kW なら余裕。
        b.ac_spec = nlohmann::json{
            {"Q", {{"cooling", {{"mid", 2.0}, {"rtd", 7.2}}},
                   {"heating", {{"mid", 0.5}, {"rtd", 7.2}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        SimulationConstants constants{};
        std::ostringstream logs;
        int totalIterations = 0;
        const bool adjusted = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);

        expectTrue(!adjusted, "with rtd>measured, mid must not become capacity cap");
        expectNear(b.current_pre_temp, 26.0, 1e-12, "setpoint unchanged when under Q.rtd");
        expectTrue(logs.str().find("Q.heating.rtd") != std::string::npos,
                   "capacity log should cite Q.heating.rtd");
    }

    // 処理熱量: 暖房で出口<=入口なら0、冷房で入口<=出口なら0
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 26.0;
        b.current_t = 20.0;  // 出口 < 入口
        b.on = true;
        const double heatH = controller.calculateHeatCapacity(thermal, "heating", "IN", "B", flowRates);
        expectNear(heatH, 0.0, 1e-9, "heating: outlet < inlet => heat 0");
        in.current_t = 20.0;
        b.current_t = 26.0;  // 入口 < 出口
        const double heatC = controller.calculateHeatCapacity(thermal, "cooling", "IN", "B", flowRates);
        expectNear(heatC, 0.0, 1e-9, "cooling: inlet < outlet => heat 0");
    }

    // 能力超過で bracket を使った後、処理熱量が不足（0）になったら設定温度を上げて二分探索継続すること
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 25.0;
        b.current_t = 30.0;
        b.current_mode = "HEATING";
        setRequestedAndEffective(b, 20.0);
        b.on = true;
        b.ac_spec = makeAcSpecWithMax(3.3, 0.5);  // 500W
        b.initializeAirconSpec();

        VentilationNetwork vent;
        SimulationConstants constants{};
        std::ostringstream logs;
        int totalIterations = 0;
        const bool adjusted1 = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);
        expectTrue(adjusted1, "first call: over capacity should adjust (bracket path when nullopt)");
        const double setpointAfter1 = b.current_pre_temp;

        // 処理熱量が0になるようにする（出口 <= 入口）
        b.current_t = 24.0;
        in.current_t = 25.0;
        const bool adjusted2 = controller.checkAndAdjustCapacity(
            thermal, vent, constants, flowRates, logs, totalIterations);
        expectTrue(adjusted2, "second call: under capacity with bracket should request recalc");
        expectTrue(b.current_pre_temp > setpointAfter1,
                   "under capacity: setpoint should increase toward max capacity");
        // 設定温度が動いたなら必ず recompute 要求（tautology にならないよう明示）
        expectTrue(std::abs(b.current_pre_temp - setpointAfter1) > 1e-9,
                   "under capacity should move effective setpoint");
        expectTrue(adjusted2, "moved setpoint must request recompute");
    }

    // capacityConverged: 現在点が上限近傍なら設定温度を動かさず bracket を終了し、再計算しない
    {
        VertexProperties node;
        node.key = "AC1";
        node.on = true;
        node.current_mode = "COOLING";
        node.current_requested_pre_temp = 24.0;
        node.current_pre_temp = 24.0;
        node.aircon_control_state = AirconControlState::CapacityLimited;

        // 冷房: bracket 幅は大きいが、現在能力は max±tol 内（tol = 500*0.001+1 = 1.5W）
        aircon::capacity::CapacityBracketMap brackets;
        brackets["AC1"] = aircon::capacity::CapacityBracket{24.0, 34.0, false};
        bool adjustmentMade = false;
        std::ostringstream oss;
        // inlet≈setpoint にして公式補正を nullopt にし、bracket 経路へ
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC1",
            node,
            OperationMode::Cooling,
            /*indoorTemp=*/24.0,
            /*airFlowRate=*/0.1,
            /*maxHeatCapacity=*/500.0,
            /*currentTotal=*/500.4,
            brackets,
            oss,
            adjustmentMade);

        expectNear(node.current_pre_temp, 24.0, 1e-12,
                   "capacityConverged must keep current effective setpoint");
        expectTrue(!adjustmentMade,
                   "capacityConverged at current point must not request recompute");
        expectTrue(brackets.find("AC1") == brackets.end(),
                   "capacityConverged should erase bracket");
        expectTrue(node.aircon_control_state == AirconControlState::CapacityLimited,
                   "state stays CapacityLimited");
        expectTrue(oss.str().find("二分探索収束") != std::string::npos,
                   "log should indicate capacity convergence");
    }

    // 未収束なら中点へ動かし、bracket は残す（次回も探索継続）
    {
        VertexProperties node;
        node.key = "AC2";
        node.on = true;
        node.current_mode = "COOLING";
        node.current_requested_pre_temp = 24.0;
        node.current_pre_temp = 24.0;
        node.aircon_control_state = AirconControlState::SetpointControlled;

        aircon::capacity::CapacityBracketMap brackets;
        brackets["AC2"] = aircon::capacity::CapacityBracket{24.0, 34.0, false};
        bool adjustmentMade = false;
        std::ostringstream oss;
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC2",
            node,
            OperationMode::Cooling,
            /*indoorTemp=*/24.0,
            /*airFlowRate=*/0.1,
            /*maxHeatCapacity=*/500.0,
            /*currentTotal=*/820.0,  // 明らかに超過 → 中点へ
            brackets,
            oss,
            adjustmentMade);

        expectTrue(adjustmentMade, "over capacity far from max must request recompute");
        expectNear(node.current_pre_temp, 29.0, 1e-9,
                   "cooling over capacity should move to bracket midpoint");
        expectTrue(brackets.find("AC2") != brackets.end(),
                   "non-converged step must keep bracket for next iteration");
        expectNear(brackets["AC2"].tLow, 24.0, 1e-12, "cooling: tLow becomes current");
        expectNear(brackets["AC2"].tHigh, 34.0, 1e-12, "cooling: tHigh unchanged when over");
        expectTrue(!brackets["AC2"].finalVerificationPending,
                   "midpoint step should not arm final verification");
    }

    // bracket 幅だけ収束: 最終検証1回のあと、能力未達でも外側ループを止められること
    {
        VertexProperties node;
        node.key = "AC3";
        node.on = true;
        node.current_mode = "COOLING";
        node.current_requested_pre_temp = 24.0;
        node.current_pre_temp = 24.0;
        node.aircon_control_state = AirconControlState::CapacityLimited;

        // 既に幅が tol 以下の bracket（冷房: 超過側=low, 非超過側=high）
        aircon::capacity::CapacityBracketMap brackets;
        brackets["AC3"] = aircon::capacity::CapacityBracket{24.0000, 24.0005, false};

        bool adjustmentMade = false;
        std::ostringstream oss1;
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC3", node, OperationMode::Cooling, 24.0, 0.1, 500.0,
            /*currentTotal=*/900.0, brackets, oss1, adjustmentMade);

        expectTrue(adjustmentMade, "bracket-width converge should request final verification");
        expectTrue(brackets.find("AC3") != brackets.end(), "bracket kept until final verification");
        expectTrue(brackets["AC3"].finalVerificationPending, "finalVerificationPending armed");
        // 冷房の可行端は tHigh
        expectNear(node.current_pre_temp, 24.0005, 1e-9, "feasible endpoint adopted");

        // 最終検証: 能力誤差はまだ大きいが、pending 消化で終了（拡張不能なら打ち切り）
        adjustmentMade = false;
        std::ostringstream oss2;
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC3", node, OperationMode::Cooling, 24.0, 0.1, 500.0,
            /*currentTotal=*/900.0, brackets, oss2, adjustmentMade);

        // inlet≈setpoint だと推定能力0なので拡張は「可行」と判定されうるが、
        // いずれにせよ最終検証後に無限ループへ入らないこと
        if (brackets.find("AC3") != brackets.end()) {
            // 拡張して継続する場合は再計算要求があり、pending は落ちている
            expectTrue(adjustmentMade, "expanded search should request recompute");
            expectTrue(!brackets["AC3"].finalVerificationPending,
                       "pending cleared after final verification handling");
            // 2回目の最終検証相当をシミュレート: 再び幅収束→pending→検証で終了させる
            brackets["AC3"].tLow = node.current_pre_temp;
            brackets["AC3"].tHigh = node.current_pre_temp + 5e-4;
            brackets["AC3"].finalVerificationPending = false;
            adjustmentMade = false;
            std::ostringstream oss3;
            aircon::capacity::applyExceededCapacityAdjustment(
                "AC3", node, OperationMode::Cooling, 24.0, 0.1, 500.0, 900.0,
                brackets, oss3, adjustmentMade);
            expectTrue(brackets["AC3"].finalVerificationPending, "re-arm pending on width converge");
            adjustmentMade = false;
            // 天井まで拡張済みに見せるため tHigh を天井に固定して拡張失敗→未解決例外
            brackets["AC3"].tLow = 49.999;
            brackets["AC3"].tHigh = 50.0;
            brackets["AC3"].finalVerificationPending = true;
            node.current_pre_temp = 50.0;
            bool threw = false;
            try {
                std::ostringstream oss4;
                aircon::capacity::applyExceededCapacityAdjustment(
                    "AC3", node, OperationMode::Cooling, 24.0, 0.1, 500.0, 900.0,
                    brackets, oss4, adjustmentMade);
            } catch (const simulation::Error& e) {
                threw = true;
                expectTrue(e.code() == simulation::ErrorCode::CapacityConstraintUnresolved,
                           "unresolved capacity should use CapacityConstraintUnresolved");
            }
            expectTrue(threw, "give up after final verify must throw");
            expectTrue(brackets.find("AC3") == brackets.end(),
                       "bracket erased before throw");
        } else {
            expectTrue(!adjustmentMade || oss2.str().find("最終検証") != std::string::npos,
                       "final verification should end or log verification");
        }
    }

    // 最終検証で能力が上限以内なら bracket を消して再計算しない
    {
        VertexProperties node;
        node.key = "AC4";
        node.on = true;
        node.current_pre_temp = 28.0;
        node.aircon_control_state = AirconControlState::CapacityLimited;
        aircon::capacity::CapacityBracketMap brackets;
        brackets["AC4"] = aircon::capacity::CapacityBracket{24.0, 28.0, true};
        bool adjustmentMade = false;
        std::ostringstream oss;
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC4", node, OperationMode::Cooling, 24.0, 0.1, 500.0,
            /*currentTotal=*/499.0, brackets, oss, adjustmentMade);
        expectTrue(brackets.find("AC4") == brackets.end(), "final OK erases bracket");
        expectTrue(!adjustmentMade, "final OK should not request another recompute");
        expectNear(node.current_pre_temp, 28.0, 1e-12, "final OK keeps verified setpoint");
    }

    // 大きな処理熱量を近似入口負荷だけで無視せず、実ネットワークで再評価する
    {
        VertexProperties node;
        node.key = "AC_UNPHYS";
        node.on = true;
        node.current_mode = "HEATING";
        node.current_requested_pre_temp = 20.0;
        node.current_pre_temp = 20.0;
        node.aircon_control_state = AirconControlState::SetpointControlled;
        aircon::capacity::CapacityBracketMap brackets;
        bool adjustmentMade = false;
        std::ostringstream oss;
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC_UNPHYS",
            node,
            OperationMode::Heating,
            /*indoorTemp=*/18.84,
            /*airFlowRate=*/0.05,
            /*maxHeatCapacity=*/7734.0,
            /*currentTotal=*/61198.1,
            brackets,
            oss,
            adjustmentMade);
        expectTrue(adjustmentMade, "exceeded measured capacity must request recompute");
        expectTrue(node.current_pre_temp < 20.0, "capacity search lowers heating target");
        expectTrue(!brackets.empty(), "inlet approximation cannot bypass capacity search");
    }

    // 入口・設定・風量からも超過するときは、従来どおり設定探索する
    {
        VertexProperties node;
        node.key = "AC_REAL";
        node.on = true;
        node.current_mode = "HEATING";
        node.current_requested_pre_temp = 20.0;
        node.current_pre_temp = 20.0;
        node.aircon_control_state = AirconControlState::SetpointControlled;
        aircon::capacity::CapacityBracketMap brackets;
        bool adjustmentMade = false;
        std::ostringstream oss;
        aircon::capacity::applyExceededCapacityAdjustment(
            "AC_REAL",
            node,
            OperationMode::Heating,
            /*indoorTemp=*/0.0,
            /*airFlowRate=*/0.5,
            /*maxHeatCapacity=*/5000.0,
            /*currentTotal=*/12000.0,
            brackets,
            oss,
            adjustmentMade);
        expectTrue(adjustmentMade, "physical over-capacity still searches setpoint");
        expectTrue(node.current_pre_temp < 20.0,
                   "physical over-capacity lowers heating setpoint");
    }

    // 符号付き必要負荷: 暖房ONでも冷房需要ならOFFへ
    {
        VertexProperties ac;
        ac.key = "H1";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = true;
        ac.current_requested_pre_temp = 20.0;
        ac.current_pre_temp = 20.0;
        ac.required_heat_w = -120.0; // 冷房需要
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, /*currentTemp=*/20.0, /*targetTemp=*/20.0, 0.5, logs,
                                          /*useRequiredHeat=*/true, ac.required_heat_w, 1.0);
        expectTrue(r.stateChanged && !r.on, "heating with cooling demand must turn OFF");
    }
    // 符号付き必要負荷: 冷房ONでも暖房需要ならOFFへ
    {
        VertexProperties ac;
        ac.key = "C1";
        ac.set_node = "ROOM";
        ac.current_mode = "COOLING";
        ac.on = true;
        ac.required_heat_w = 80.0;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, 24.0, 24.0, 0.5, logs, true, ac.required_heat_w, 1.0);
        expectTrue(r.stateChanged && !r.on, "cooling with heating demand must turn OFF");
    }
    // OFF中は室温で再起動判定（負荷未評価）
    {
        VertexProperties ac;
        ac.key = "H2";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = false;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, /*currentTemp=*/18.0, /*targetTemp=*/20.0, 0.5, logs,
                                          /*useRequiredHeat=*/false, 0.0, 1.0);
        expectTrue(r.stateChanged && r.on, "cold free room should turn heating ON");
    }
    // 極小 thermal tol でも温度バンドは最低 1.0K（Qreq≈0 OFF 直後の再 ON 振動防止）
    {
        VertexProperties ac;
        ac.key = "C_CHATTER";
        ac.set_node = "ROOM";
        ac.current_mode = "COOLING";
        ac.on = false;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, /*currentTemp=*/27.003, /*targetTemp=*/27.0,
                                          /*tolerance=*/1e-6, logs, false, 0.0, 1.0);
        expectTrue(!r.stateChanged && !r.on,
                   "cooling OFF must stay OFF for +0.003K drift when band floor is 1.0K");
    }
    // 遠隔 set: Qreq OFF 後に室温が設定−0.7K 程度まで落ちても、1.0K 帯内なら再 ON しない
    {
        VertexProperties ac;
        ac.key = "H_CHATTER";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = false;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, /*currentTemp=*/19.29, /*targetTemp=*/20.0,
                                          /*tolerance=*/1e-6, logs, false, 0.0, 1.0);
        expectTrue(!r.stateChanged && !r.on,
                   "heating OFF must stay OFF at −0.71K when band floor is 1.0K");
    }
    {
        VertexProperties ac;
        ac.key = "H_NEED";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = false;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, /*currentTemp=*/18.5, /*targetTemp=*/20.0,
                                          /*tolerance=*/1e-6, logs, false, 0.0, 1.0);
        expectTrue(r.stateChanged && r.on, "heating OFF must turn ON when clearly below 1.0K band");
    }
    {
        VertexProperties ac;
        ac.key = "C_NEED";
        ac.set_node = "ROOM";
        ac.current_mode = "COOLING";
        ac.on = false;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, /*currentTemp=*/28.5, /*targetTemp=*/27.0,
                                          /*tolerance=*/1e-6, logs, false, 0.0, 1.0);
        expectTrue(r.stateChanged && r.on, "cooling OFF must turn ON when clearly above 1.0K band");
    }

    // 同一 set_node を複数空調が制御すると fixed-row が頂点順依存になるため禁止する
    {
        ThermalNetwork t2;
        t2.addNode(makeNode("R", "normal", 20.0));
        auto a1 = makeNode("AC_A", "aircon", 20.0);
        a1.set_node = "R";
        a1.model = "IDEAL";
        auto a2 = makeNode("AC_B", "aircon", 20.0);
        a2.set_node = "R";
        a2.model = "IDEAL";
        t2.addNode(a1);
        t2.addNode(a2);
        AirconController c2;
        bool threw = false;
        try {
            std::ostringstream logs;
            c2.initializeModels(t2, logs, 0);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("multiple aircons") != std::string::npos;
        }
        expectTrue(threw, "duplicate set_node must be rejected at initializeModels");
    }

    // 符号付き必要負荷: 暖房でも Qreq≈0 なら OFF（能力0Wで室温拘束を残さない）
    {
        VertexProperties ac;
        ac.key = "H0";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = true;
        ac.required_heat_w = 0.0;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, 20.0, 20.0, 0.5, logs, true, 0.0, 1.0);
        expectTrue(r.stateChanged && !r.on, "heating with Qreq≈0 must turn OFF");
    }

    // 設定維持中、制御用 min_process 以下なら停止せず最低能力分を処理する。
    // カタログ Q.min だけでは止まらない。
    {
        VertexProperties ac;
        ac.key = "HMIN";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = true;
        ac.required_heat_w = 160.0;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, 20.0, 20.0, 0.5, logs, true, ac.required_heat_w, 1.0,
                                          /*minProcessHeatW=*/2706.9, /*holdAtMinimumCapacity=*/true);
        expectTrue(!r.stateChanged && r.on, "heating below min_process stays ON when hold is set");
        expectTrue(r.logMessage.find("min_process=") != std::string::npos,
                   "min_process hold should cite min_process");

        ac.required_heat_w = 2800.0;
        auto stay = controller.controlAircon(ac, 20.0, 20.0, 0.5, logs, true, ac.required_heat_w, 1.0,
                                             2706.9, /*holdAtMinimumCapacity=*/false);
        expectTrue(!stay.stateChanged && stay.on, "heating above min_process stays ON");

        // 暖房で負の Qreq は min_process ホールド対象外（誤って hold しても OFF）。
        ac.required_heat_w = -65.9;
        auto neg = controller.controlAircon(ac, 20.0, 20.0, 0.5, logs, true, ac.required_heat_w, 1.0,
                                            773.4, /*holdAtMinimumCapacity=*/true);
        expectTrue(neg.stateChanged && !neg.on,
                   "heating negative Qreq must turn OFF even if hold flag is set");
        expectTrue(neg.logMessage.find("最低能力帯のON維持") == std::string::npos,
                   "negative heating Qreq must not cite min-process hold");
    }
    // RAC / CRIEPI / DUCT: カタログ Q.min だけでは最低能力停止しない。
    // 明示 min_process が無ければ Q.rtd の 10% を既定とし、小負荷はそれを処理する。
    {
        auto runMinProcess = [&](const std::string& key, const std::string& model,
                                 const nlohmann::json& spec, double qReq, bool expectHold,
                                 double expectedHoldW, const std::string& msg) {
            ThermalNetwork net;
            auto room = makeNode("ROOM", "normal", 20.0);
            auto ac = makeNode(key, "aircon", 22.0);
            ac.model = model;
            ac.current_mode = "HEATING";
            ac.on = true;
            ac.set_node = "ROOM";
            ac.in_node = "ROOM";
            ac.current_pre_temp = 20.0;
            ac.current_requested_pre_temp = 20.0;
            ac.required_heat_w = qReq;
            ac.ac_spec = spec;
            ac.initializeAirconSpec();
            net.addNode(room);
            net.addNode(ac);
            AirconController local;
            std::ostringstream logs;
            local.initializeModels(net, logs, 0);
            FlowRateMap flows;
            flows[{"ROOM", key}] = 0.1;
            (void)local.controlAllAircons(net, 0.5, logs, nullptr, 1e-9, nullptr, &flows);
            auto& held = net.getNode(key);
            if (expectHold) {
                expectTrue(held.on, msg + " stays ON");
                expectNear(held.required_heat_w, expectedHoldW, 1e-6, msg + " processes min_process");
                expectTrue(logs.str().find("min_process=") != std::string::npos,
                           msg + " should cite min_process");
            } else {
                expectTrue(held.on, msg + " stays ON without min_process stop");
                expectNear(held.required_heat_w, qReq, 1e-6, msg + " keeps original Qreq");
            }
        };
        const auto racCatalogMinOnly = nlohmann::json{
            {"Q", {{"heating", {{"min", 0.7}, {"rtd", 2.5}, {"max", 5.4}}},
                   {"cooling", {{"min", 0.7}, {"rtd", 2.2}, {"max", 3.3}}}}},
            {"P", {{"heating", {{"min", 0.095}, {"rtd", 0.39}, {"max", 1.36}}},
                   {"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.78}}}}},
        };
        // 既定 10% of Q.rtd = 0.25 kW。カタログ Q.min=0.7 は使わない。
        runMinProcess("RAC1", "RAC", racCatalogMinOnly, 160.0, true, 250.0,
                      "RAC default min_process is 10% of Q.rtd");
        runMinProcess("RAC1A", "RAC", racCatalogMinOnly, 400.0, false, 0.0,
                      "RAC load above 1.2x default 10% keeps Qreq");

        const auto racWithMinProcess = nlohmann::json{
            {"Q", {{"heating", {{"min", 0.7}, {"rtd", 2.5}, {"max", 5.4}}},
                   {"cooling", {{"min", 0.7}, {"rtd", 2.2}, {"max", 3.3}}}}},
            {"P", {{"heating", {{"min", 0.095}, {"rtd", 0.39}, {"max", 1.36}}},
                   {"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.78}}}}},
            {"min_process", {{"heating", 0.7}, {"cooling", 0.7}}},
        };
        runMinProcess("RAC1B", "RAC", racWithMinProcess, 160.0, true, 700.0,
                      "RAC with min_process clamps small load");
        runMinProcess("RAC1C", "RAC", racWithMinProcess, 900.0, false, 0.0,
                      "RAC with min_process keeps load above 1.2x threshold");
        runMinProcess("RAC1D", "RAC", racWithMinProcess, 800.0, true, 700.0,
                      "RAC 1.14x min_process still holds minimum");

        // 暖房で負の Qreq（過昇温・日射など）は最低能力ホールドせず OFF。
        {
            ThermalNetwork net;
            auto room = makeNode("ROOM", "normal", 20.0);
            auto ac = makeNode("RAC_NEG", "aircon", 22.0);
            ac.model = "RAC";
            ac.current_mode = "HEATING";
            ac.on = true;
            ac.set_node = "ROOM";
            ac.in_node = "ROOM";
            ac.current_pre_temp = 20.0;
            ac.current_requested_pre_temp = 20.0;
            ac.required_heat_w = -65.9;
            ac.ac_spec = racWithMinProcess;
            ac.initializeAirconSpec();
            net.addNode(room);
            net.addNode(ac);
            AirconController local;
            std::ostringstream logs;
            local.initializeModels(net, logs, 0);
            FlowRateMap flows;
            flows[{"ROOM", "RAC_NEG"}] = 0.1;
            (void)local.controlAllAircons(net, 0.5, logs, nullptr, 1e-9, nullptr, &flows);
            auto& held = net.getNode("RAC_NEG");
            expectTrue(!held.on, "heating negative Qreq turns OFF (no abs min_process hold)");
            expectTrue(logs.str().find("最低能力帯のON維持") == std::string::npos,
                       "heating negative Qreq must not log min-process hold");
            expectTrue(!std::isfinite(held.required_heat_w),
                       "heating negative Qreq clears required heat on OFF");
        }

        // 冷房で正の Qreq も同様にホールドせず OFF。
        {
            ThermalNetwork net;
            auto room = makeNode("ROOM", "normal", 26.0);
            auto ac = makeNode("RAC_POS_COOL", "aircon", 22.0);
            ac.model = "RAC";
            ac.current_mode = "COOLING";
            ac.on = true;
            ac.set_node = "ROOM";
            ac.in_node = "ROOM";
            ac.current_pre_temp = 26.0;
            ac.current_requested_pre_temp = 26.0;
            ac.required_heat_w = 65.9;
            ac.ac_spec = racWithMinProcess;
            ac.initializeAirconSpec();
            net.addNode(room);
            net.addNode(ac);
            AirconController local;
            std::ostringstream logs;
            local.initializeModels(net, logs, 0);
            FlowRateMap flows;
            flows[{"ROOM", "RAC_POS_COOL"}] = 0.1;
            (void)local.controlAllAircons(net, 0.5, logs, nullptr, 1e-9, nullptr, &flows);
            auto& held = net.getNode("RAC_POS_COOL");
            expectTrue(!held.on, "cooling positive Qreq turns OFF (no abs min_process hold)");
            expectTrue(logs.str().find("最低能力帯のON維持") == std::string::npos,
                       "cooling positive Qreq must not log min-process hold");
        }

        const auto criepi = nlohmann::json{
            {"Q", {{"heating", {{"min", 0.7}, {"rtd", 2.5}, {"max", 5.4}}},
                   {"cooling", {{"min", 0.7}, {"rtd", 2.2}, {"max", 3.3}}}}},
            {"P", {{"heating", {{"min", 0.095}, {"rtd", 0.39}, {"max", 1.36}}},
                   {"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.78}}}}},
            {"V_inner", {{"heating", {{"rtd", 0.2}}}, {"cooling", {{"rtd", 0.2}}}}},
            {"V_outer", {{"heating", {{"rtd", 0.4}}}, {"cooling", {{"rtd", 0.4}}}}},
            {"min_process", {{"heating", 0.7}, {"cooling", 0.7}}},
        };
        runMinProcess("CR1", "CRIEPI", criepi, 160.0, true, 700.0,
                      "CRIEPI with min_process clamps small load");
        runMinProcess("CR1B", "CRIEPI", criepi, 700.0, true, 700.0,
                      "CRIEPI at min_process still clamps to min_process");
    }

    // OFF 後は Qreq が残っていても温度バンド内なら再起動しない
    {
        VertexProperties ac;
        ac.key = "HMIN_OFF";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = false;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, 19.2, 20.0, 0.5, logs, false, 160.0, 1.0, 2706.9);
        expectTrue(!r.stateChanged && !r.on, "OFF must not restart inside 1K band");
    }

    // holdAtMinimumCapacity=true なら最低能力で継続する。
    {
        VertexProperties ac;
        ac.key = "HMIN_HOLD";
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = true;
        ac.required_heat_w = 160.0;
        std::ostringstream logs;
        auto r = controller.controlAircon(ac, 20.0, 20.0, 0.5, logs, true, ac.required_heat_w, 1.0,
                                          2706.9, /*holdAtMinimumCapacity=*/true);
        expectTrue(!r.stateChanged && r.on, "hold must stay ON at minimum capacity");
        expectTrue(r.logMessage.find("最低能力帯のON維持") != std::string::npos,
                   "hold-min log should cite min process");
    }

    // inMinProcessHoldBand の符号表（コントローラを通さない単体）。
    {
        using aircon::onoff::inMinProcessHoldBand;
        using aircon::onoff::signedMinProcessHeatW;
        using aircon::onoff::buildRequiredHeatTrust;
        using aircon::onoff::resolveMinProcessHeatW;
        expectTrue(inMinProcessHoldBand("heating", 160.0, 773.4),
                   "heating small positive is in hold band");
        expectTrue(!inMinProcessHoldBand("heating", -65.9, 773.4),
                   "heating negative is outside hold band");
        expectTrue(!inMinProcessHoldBand("heating", 0.0, 773.4),
                   "heating zero is outside hold band");
        expectTrue(inMinProcessHoldBand("heating", 800.0, 773.4),
                   "heating 1.03x min_process stays in hold band");
        expectTrue(inMinProcessHoldBand("heating", 844.34, 773.4),
                   "heating 1.09x min_process stays in hold band");
        expectTrue(!inMinProcessHoldBand("heating", 1000.0, 773.4),
                   "heating above 1.2x min_process is outside hold band");
        expectTrue(inMinProcessHoldBand("heating", 963.4, 773.4, aircon::onoff::kLoadDeadbandW, true),
                   "heating 1.25x min_process stays in hold once entered");
        expectTrue(!inMinProcessHoldBand("heating", 963.4, 773.4),
                   "heating 1.25x min_process does not enter hold");
        expectTrue(inMinProcessHoldBand("heating", 1200.0, 773.4, aircon::onoff::kLoadDeadbandW, true),
                   "heating 1.55x min_process stays in hold until timestep end");
        expectTrue(inMinProcessHoldBand("heating", 1254.18, 773.4, aircon::onoff::kLoadDeadbandW, true),
                   "heating 1.62x min_process (year-run period-2) stays in hold");
        expectTrue(!inMinProcessHoldBand("heating", 1254.18, 773.4),
                   "heating 1.62x min_process does not enter hold");
        expectTrue(!inMinProcessHoldBand("heating", -65.9, 773.4, aircon::onoff::kLoadDeadbandW, true),
                   "heating opposite-sign Qreq exits hold even if previously held");
        expectTrue(inMinProcessHoldBand("cooling", -1254.18, 773.4, aircon::onoff::kLoadDeadbandW, true),
                   "cooling 1.62x min_process stays in hold once entered");
        expectTrue(inMinProcessHoldBand("cooling", -160.0, 773.4),
                   "cooling small negative is in hold band");
        expectTrue(!inMinProcessHoldBand("cooling", 65.9, 773.4),
                   "cooling positive is outside hold band");
        expectNear(signedMinProcessHeatW("heating", 773.4), 773.4, 1e-12,
                   "heating signed min is positive");
        expectNear(signedMinProcessHeatW("cooling", 773.4), -773.4, 1e-12,
                   "cooling signed min is negative");
        expectNear(resolveMinProcessHeatW(0.7), 700.0, 1e-12,
                   "resolveMinProcessHeatW converts kW to W");
        expectNear(resolveMinProcessHeatW(std::nullopt), 0.0, 0.0,
                   "resolveMinProcessHeatW nullopt is 0");
        const auto trustOk = buildRequiredHeatTrust(20.0, 20.0, 20.0, 1.0, false, 0.1);
        expectTrue(trustOk.nearEffectiveSetpoint && !trustOk.capacityLimited &&
                       !trustOk.effectiveDetunedFromRequest && !trustOk.processFlowTooLow,
                   "buildRequiredHeatTrust healthy case");
        const auto trustLowFlow = buildRequiredHeatTrust(20.0, 20.0, 20.0, 1.0, false, 1e-5);
        expectTrue(trustLowFlow.processFlowTooLow, "buildRequiredHeatTrust flags tiny flow");
        const auto trustNoFlow = buildRequiredHeatTrust(20.0, 20.0, 20.0, 1.0, true, std::nullopt);
        expectTrue(trustNoFlow.capacityLimited && !trustNoFlow.processFlowTooLow,
                   "buildRequiredHeatTrust skips flow check when nullopt");
    }
    {
        auto runHoldMin = [&](const std::string& mode, double freeTemp, double qReq,
                              bool expectHold, const std::string& msg) {
            ThermalNetwork net;
            auto room = makeNode("ROOM", "normal", freeTemp);
            auto ac = makeNode("HOLD", "aircon", freeTemp);
            ac.model = "IDEAL";
            ac.current_mode = mode;
            ac.on = false;
            ac.set_node = "ROOM";
            ac.in_node = "ROOM";
            setRequestedAndEffective(ac, 20.0);
            ac.ac_spec = nlohmann::json{
                {"Q", {{"heating", {{"min", 2.7069}, {"rtd", 7.2}}},
                       {"cooling", {{"min", 2.7069}, {"rtd", 7.2}}}}},
                {"min_process", {{"heating", 2.7069}, {"cooling", 2.7069}}},
            };
            ac.initializeAirconSpec();
            net.addNode(room);
            net.addNode(ac);
            AirconController local;
            std::ostringstream logs;
            local.initializeModels(net, logs, 0);
            FlowRateMap flows;
            flows[{"ROOM", "HOLD"}] = 0.1;

            (void)local.controlAllAircons(net, 0.5, logs, nullptr, 1e-9, nullptr, &flows);

            net.getNode("ROOM").current_t = 20.0;
            auto& held = net.getNode("HOLD");
            held.on = true;
            held.required_heat_w = qReq;
            setRequestedAndEffective(held, 20.0);
            std::ostringstream logs2;
            (void)local.controlAllAircons(net, 0.5, logs2, nullptr, 1e-9, nullptr, &flows);

            const double signedMin = (mode == "COOLING") ? -2706.9 : 2706.9;
            if (expectHold) {
                expectTrue(held.on, msg + " stays ON");
                expectNear(held.required_heat_w, signedMin, 1e-6, msg + " processes signed min_process");
                expectTrue(logs2.str().find("最低能力帯のON維持") != std::string::npos,
                           msg + " cites min process");
            } else {
                // |Qreq| がデッドバンド以下なら OFF（min_process でも負荷0は止める）
                expectTrue(!held.on, msg + " turns OFF");
                expectTrue(!std::isfinite(held.required_heat_w), msg + " clears required heat");
            }
        };
        runHoldMin("HEATING", 18.9, 160.0, true,
                   "heating below min_process processes minimum");
        runHoldMin("HEATING", 19.5, 0.0, false,
                   "heating with near-zero load turns OFF");
        runHoldMin("HEATING", 20.0, -65.9, false,
                   "heating with negative Qreq turns OFF (not abs hold)");
        runHoldMin("COOLING", 21.2, -160.0, true,
                   "cooling below min_process processes minimum");
        runHoldMin("COOLING", 20.0, 65.9, false,
                   "cooling with positive Qreq turns OFF (not abs hold)");
    }

    // AirconStateProposal: ON/OFF 変化で OnOffChanged が立つ
    {
        thermal.addNode(makeNode("ROOM2", "normal", 18.0));
        auto& room = thermal.getNode("ROOM2");
        auto& ac = thermal.getNode("B");
        ac.set_node = "ROOM2";
        ac.current_mode = "HEATING";
        ac.on = false;
        setRequestedAndEffective(ac, 26.0);
        room.current_t = 18.0;
        std::ostringstream logs;
        std::vector<AirconStateProposal> proposals;
        (void)controller.controlAllAircons(thermal, 0.5, logs, nullptr, 1e-9, &proposals);
        expectTrue(ac.on, "cold room should turn heating ON");
        bool sawOnOff = false;
        for (const auto& p : proposals) {
            if (p.airconKey == "B" && hasReason(p.reasons, AirconRecomputeReason::OnOffChanged)) {
                sawOnOff = true;
                expectNear(p.requestedSetpoint, 26.0, 1e-12, "proposal requested");
                expectNear(p.effectiveSetpoint, 26.0, 1e-12, "proposal effective after ON");
            }
        }
        expectTrue(sawOnOff, "ON transition should set OnOffChanged in proposal");
    }

    // ON/OFF は要求設定温度を参照し、能力補正後の実効設定には引きずられない
    {
        thermal.addNode(makeNode("ROOM", "normal", 21.0));
        auto& room = thermal.getNode("ROOM");
        auto& ac = thermal.getNode("B");
        ac.set_node = "ROOM";
        ac.current_mode = "HEATING";
        ac.on = true;
        ac.current_requested_pre_temp = 26.0;
        ac.current_pre_temp = 22.0;  // 能力制限後の実効値を模擬
        ac.aircon_control_state = AirconControlState::CapacityLimited;
        room.current_t = 21.0;
        std::ostringstream logs;
        (void)controller.controlAllAircons(thermal, 0.5, logs);
        expectTrue(ac.on, "heating below requested setpoint should stay ON");
        expectNear(ac.current_requested_pre_temp, 26.0, 1e-12, "requested unchanged by control");
        // CapacityLimited 中は実効設定を維持
        expectNear(ac.current_pre_temp, 22.0, 1e-12, "effective setpoint kept while CapacityLimited");
    }

    // 能力制限中に Qreq が負でも OFF しない（実効設定を下げた拘束の反転負荷）
    {
        thermal.addNode(makeNode("ROOM_CAP", "normal", 18.0));
        auto& room = thermal.getNode("ROOM_CAP");
        auto& ac = thermal.getNode("B");
        ac.set_node = "ROOM_CAP";
        ac.current_mode = "HEATING";
        ac.on = true;
        ac.current_requested_pre_temp = 23.0;
        ac.current_pre_temp = 18.0;
        ac.aircon_control_state = AirconControlState::CapacityLimited;
        ac.required_heat_w = -3203.4;  // 実効18℃拘束では除熱側に見える
        room.current_t = 18.0;
        std::ostringstream logs;
        (void)controller.controlAllAircons(thermal, 0.5, logs);
        expectTrue(ac.on, "CapacityLimited heating must not OFF on negative Qreq");
        expectNear(ac.current_pre_temp, 18.0, 1e-12, "effective setpoint kept after Qreq skip");
    }

    // DUCT_CENTRAL: 処理熱量に応じて風量を補正すること
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        auto& out = thermal.getNode("OUT");
        in.current_t = 20.0;
        b.current_t = 30.0;
        out.current_t = 7.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.outside_node = "OUT";
        b.set_node.clear();
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_requested_pre_temp = 20.0;
        b.current_pre_temp = 20.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.2}}}, {"cooling", {{"dsgn", 0.2}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties e{};
        e.key = "vb_in_b";
        e.unique_id = "vb_in_b";
        e.type = "fixed_flow";
        e.source = "IN";
        e.target = "B";
        e.current_vol = 0.3;
        e.flow_rate = 0.3;
        (void)boost::add_edge(vIn, vB, e, vg);

        FlowRateMap ductFlowRates;
        ductFlowRates[{"IN", "B"}] = 0.3;
        std::ostringstream logs;
        const bool adjusted = controller.checkAndAdjustDuctCentralAirflow(
            thermal, vent, ductFlowRates, logs);
        expectTrue(adjusted, "duct_central airflow should trigger adjustment when flow mismatches load");

        bool found = false;
        double adjustedFlow = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            const auto src = vg[boost::source(edge, vg)].key;
            const auto dst = vg[boost::target(edge, vg)].key;
            if (src == "IN" && dst == "B") {
                found = true;
                adjustedFlow = vg[edge].current_vol;
                break;
            }
        }
        expectTrue(found, "updated ventilation graph should contain IN->B edge");
        if (found) {
            const double qW = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * 0.3 * (30.0 - 20.0);
            const double targetFlow = 0.2 * std::clamp(qW / (7.2 * 1000.0), 0.0, 1.0);
            expectNear(adjustedFlow, targetFlow, 2e-4, "duct_central target flow should follow Q/Q_rtd * V_dsgn");
        }
    }

    // DUCT_CENTRAL: 吸込と吹出を同じ風量にする（in != out / in == out）
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 20.0;
        b.current_t = 30.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node.clear();
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_requested_pre_temp = 20.0;
        b.current_pre_temp = 20.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.2}}}, {"cooling", {{"dsgn", 0.2}}}}},
        };
        b.initializeAirconSpec();

        thermal.addNode(makeNode("SUPPLY", "normal", 22.0));
        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        const auto vSupply = boost::add_vertex(makeNode("SUPPLY", "normal", 22.0), vg);

        auto addFixed = [&](const std::string& key, auto src, auto dst,
                            const std::string& source, const std::string& target, double vol) {
            EdgeProperties e{};
            e.key = key;
            e.unique_id = key;
            e.type = "fixed_flow";
            e.subtype = "aircon";
            e.source = source;
            e.target = target;
            e.current_vol = vol;
            e.flow_rate = vol;
            (void)boost::add_edge(src, dst, e, vg);
        };
        addFixed("in_b", vIn, vB, "IN", "B", 0.3);
        addFixed("b_supply", vB, vSupply, "B", "SUPPLY", 0.3);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.3;
        std::ostringstream logs;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs),
                   "duct circuit should adjust both intake and supply");

        double intake = 0.0;
        double supply = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            const auto src = vg[boost::source(edge, vg)].key;
            const auto dst = vg[boost::target(edge, vg)].key;
            if (src == "IN" && dst == "B") intake = vg[edge].current_vol;
            if (src == "B" && dst == "SUPPLY") supply = vg[edge].current_vol;
        }
        const double qW = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * 0.3 * (30.0 - 20.0);
        const double targetFlow = 0.2 * std::clamp(qW / (7.2 * 1000.0), 0.0, 1.0);
        expectNear(intake, targetFlow, 2e-4, "intake flow follows target");
        expectNear(supply, targetFlow, 2e-4, "supply flow matches intake");
        expectNear(supply, intake, 1e-12, "intake and supply share the same signed magnitude");

        // 吸込枝は目標のまま、吹出だけ戻しても吹出を再同期する
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            if (vg[boost::source(edge, vg)].key == "B" && vg[boost::target(edge, vg)].key == "SUPPLY") {
                vg[edge].current_vol = 0.3;
                vg[edge].flow_rate = 0.3;
            }
        }
        flows[{"IN", "B"}] = intake;
        std::ostringstream logsSupplyOnly;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsSupplyOnly),
                   "stale supply should still be corrected when intake already matches");
        double intakeAfter = 0.0;
        double supplyAfter = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            const auto src = vg[boost::source(edge, vg)].key;
            const auto dst = vg[boost::target(edge, vg)].key;
            if (src == "IN" && dst == "B") intakeAfter = vg[edge].current_vol;
            if (src == "B" && dst == "SUPPLY") supplyAfter = vg[edge].current_vol;
        }
        expectTrue(std::abs(supplyAfter - 0.3) > 1e-6, "stale supply must leave the original vol");
        expectNear(supplyAfter, intakeAfter, 1e-12, "stale supply resynced to intake");
    }

    // DUCT_CENTRAL: ファン枝があるときは vol を書かず、定格 PQ を λ と λ² で縮める
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 20.0;
        b.current_t = 30.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node.clear();
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_requested_pre_temp = 20.0;
        b.current_pre_temp = 20.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.2}}}, {"cooling", {{"dsgn", 0.2}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        const auto vSupply = boost::add_vertex(makeNode("SUPPLY", "normal", 22.0), vg);

        EdgeProperties fan{};
        fan.key = "in_b";
        fan.unique_id = "in_b";
        fan.type = "fan";
        fan.subtype = "aircon";
        fan.source = "IN";
        fan.target = "B";
        fan.p_max = 80.0;
        fan.p1 = 40.0;
        fan.q1 = 0.1;
        fan.q_max = 0.2;
        fan.fan_curve_rated = true;
        fan.p_max_rated = 80.0;
        fan.p1_rated = 40.0;
        fan.q1_rated = 0.1;
        fan.q_max_rated = 0.2;
        (void)boost::add_edge(vIn, vB, fan, vg);

        EdgeProperties loss{};
        loss.key = "b_supply";
        loss.unique_id = "b_supply";
        loss.type = "pressure_loss";
        loss.subtype = "aircon";
        loss.source = "B";
        loss.target = "SUPPLY";
        loss.area = 0.05;
        loss.k_total = 1.0;
        (void)boost::add_edge(vB, vSupply, loss, vg);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.2;
        std::ostringstream logs;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs),
                   "fan affinity should update when speed ratio is below 1");

        double qMax = -1.0;
        double pMax = -1.0;
        bool lossUnchanged = false;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            const auto src = vg[boost::source(edge, vg)].key;
            const auto dst = vg[boost::target(edge, vg)].key;
            if (src == "IN" && dst == "B") {
                qMax = vg[edge].q_max;
                pMax = vg[edge].p_max;
                expectTrue(!vg[edge].has_prescribed_vol, "fan branch must not carry prescribed vol");
            }
            if (src == "B" && dst == "SUPPLY") {
                lossUnchanged = vg[edge].type == "pressure_loss" && vg[edge].k_total == 1.0;
            }
        }
        const double qW = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * 0.2 * (30.0 - 20.0);
        const double lambda = std::clamp((0.2 * std::clamp(qW / (7.2 * 1000.0), 0.0, 1.0)) / 0.2, 0.0, 1.0);
        const double appliedLambda = 0.5 * (1.0 + lambda);
        expectNear(qMax, 0.2 * appliedLambda, 1e-6, "fan q_max scales with damped speed ratio");
        expectNear(pMax, 80.0 * appliedLambda * appliedLambda, 1e-6, "fan pressure follows damped affinity square");
        expectTrue(lossUnchanged, "pressure_loss connector is not overwritten");
    }

    // A provisional heating solve may need more fan speed before inner convergence.
    {
        Graph trial;
        auto ac = boost::add_vertex(makeNode("AC", "aircon", 150.0), trial);
        auto& node = trial[ac];
        node.on = true;
        node.model = "DUCT_CENTRAL";
        node.current_mode = "HEATING";
        node.set_node = "ROOM";
        node.in_node = "ROOM";
        node.current_requested_pre_temp = 20.0;
        node.required_heat_w = 3600.0;
        node.aircon_control_state = AirconControlState::SetpointControlled;
        node.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.2}}}}},
        };
        node.initializeAirconSpec();
        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        auto room = boost::add_vertex(makeNode("ROOM", "normal", 20.0), vg);
        auto supply = boost::add_vertex(makeNode("AC", "aircon", 20.0), vg);
        EdgeProperties fan{};
        fan.type = "fan";
        fan.subtype = "aircon";
        fan.fan_curve_rated = true;
        fan.p_max_rated = 80.0;
        fan.p1_rated = 40.0;
        fan.q1_rated = 0.1;
        fan.q_max_rated = 0.2;
        fan.p_max = 0.8;
        fan.p1 = 0.4;
        fan.q1 = 0.01;
        fan.q_max = 0.02;
        const auto edge = boost::add_edge(room, supply, fan, vg).first;
        std::ostringstream logs;
        const auto update = [&]() {
            return aircon::airflow::increaseAirflowForCoupledSolve(vent, trial, logs, true);
        };
        expectTrue(update(), "heating candidate increases stale minimum fan speed");
        expectNear(vg[edge].q_max, 0.1, 1e-12, "half rated load requests half fan speed");
        expectNear(vg[edge].p_max, 20.0, 1e-12, "fan pressure follows affinity square");
        expectNear(node.current_t, 150.0, 1e-12, "fan predictor does not clamp the thermal candidate");
        expectTrue(!update(), "unchanged fan must not request another retry");
        node.required_heat_w = 2000.0;
        expectTrue(!update(), "inner predictor never reduces fan speed");
        node.required_heat_w = 72000.0;
        expectTrue(update(), "large demand can reach rated fan speed");
        expectNear(vg[edge].q_max, 0.2, 1e-12, "predictor never exceeds rated fan speed");

        vg[edge] = fan;
        node.required_heat_w = 3600.0;
        node.aircon_control_state = AirconControlState::CapacityLimited;
        expectTrue(!update(), "capacity bracket keeps its fan fixed");
        node.aircon_control_state = AirconControlState::Off;
        expectTrue(update(), "preset ON can precede the outer control state update");
        vg[edge] = fan;
        node.aircon_control_state = AirconControlState::SetpointControlled;
        node.on = false;
        expectTrue(!update(), "OFF unit does not increase fan");
        node.on = true;
        node.current_mode = "COOLING";
        expectTrue(!update(), "cooling with a heating load is left to ON/OFF control");
        node.current_mode = "HEATING";
        node.required_heat_w = -3600.0;
        expectTrue(!update(), "opposite load sign is left to ON/OFF control");
        node.required_heat_w = std::numeric_limits<double>::quiet_NaN();
        expectTrue(!update(), "non-finite load does not change fan");
        node.required_heat_w = 3600.0;
        node.aircon_min_process_hold = true;
        expectTrue(update(), "ON/OFF hold must not freeze the fan below the required load");
        expectNear(vg[edge].q_max, 0.1, 1e-12, "held unit still follows increasing load");
        node.aircon_min_process_hold = false;
        // A load that falls as fan speed rises has a known fixed point.
        // Undamped updates have slope -1.25 and oscillate; damping must solve
        // the original equation, without accepting the smaller applied step.
        vg[edge] = fan;
        bool fanConverged = false;
        for (int iter = 0; iter < 100; ++iter) {
            const double ratio = vg[edge].q_max / fan.q_max_rated;
            const double targetFlow = 0.2 * (4500.0 - 9000.0 * ratio) / 7200.0;
            if (!aircon::airflow::updateDuctCentralFanAffinity(
                    vent, node, OperationMode::Heating, node.key, targetFlow,
                    1e-3, nullptr, false, true)) {
                fanConverged = true;
                break;
            }
        }
        expectTrue(fanConverged, "fan/load feedback converges within its iteration budget");
        expectNear(vg[edge].q_max / fan.q_max_rated, 4500.0 / 16200.0, 1e-3,
                   "damped fan solves the unchanged load/flow fixed point");
        vg[edge].type = "fixed_flow";
        vg[edge].current_vol = 0.02;
        expectTrue(update(), "fixed-flow equipment also retries a growing load");
        expectNear(vg[edge].current_vol, 0.1, 1e-12, "fixed-flow path follows the same target");
        expectTrue(!update(), "unchanged fixed flow needs no retry");
        node.current_mode = "COOLING";
        node.required_heat_w = -3600.0;
        node.ac_spec["Q"]["cooling"] = node.ac_spec["Q"]["heating"];
        node.ac_spec["V_inner"]["cooling"] = node.ac_spec["V_inner"]["heating"];
        node.initializeAirconSpec();
        vg[edge].current_vol = 0.02;
        expectTrue(update(), "cooling fixed flow retries a growing negative load");
        expectNear(vg[edge].current_vol, 0.1, 1e-12, "cooling target uses load magnitude");
    }

    // DUCT_CENTRAL: in == out のループでも吹出を逆符号にしない
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 20.0;
        b.current_t = 30.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node.clear();
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.2}}}, {"cooling", {{"dsgn", 0.2}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties ret{};
        ret.key = "in_b_loop";
        ret.unique_id = "in_b_loop";
        ret.type = "fixed_flow";
        ret.subtype = "aircon";
        ret.source = "IN";
        ret.target = "B";
        ret.current_vol = 0.3;
        ret.flow_rate = 0.3;
        (void)boost::add_edge(vIn, vB, ret, vg);
        EdgeProperties sup{};
        sup.key = "b_in_loop";
        sup.unique_id = "b_in_loop";
        sup.type = "fixed_flow";
        sup.subtype = "aircon";
        sup.source = "B";
        sup.target = "IN";
        sup.current_vol = 0.3;
        sup.flow_rate = 0.3;
        (void)boost::add_edge(vB, vIn, sup, vg);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.3;
        std::ostringstream logs;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs),
                   "same-room duct loop should adjust");
        double intake = 0.0;
        double supply = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            const auto src = vg[boost::source(edge, vg)].key;
            const auto dst = vg[boost::target(edge, vg)].key;
            if (src == "IN" && dst == "B") intake = vg[edge].current_vol;
            if (src == "B" && dst == "IN") supply = vg[edge].current_vol;
        }
        expectTrue(intake > 0.0 && supply > 0.0, "loop intake and supply stay positive");
        expectNear(supply, intake, 1e-12, "same-room supply matches intake and is not negated");
    }

    // DUCT_CENTRAL + CapacityLimited: 計測熱が落ちても Q_max 基準の風量を維持（0 縮小防止）
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        auto& out = thermal.getNode("OUT");
        in.current_t = 20.0;
        // 風量を下げたあとに計測熱が小さく見える状況を模擬（ΔT はそのまま）
        b.current_t = 25.0;
        out.current_t = 7.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.outside_node = "OUT";
        b.set_node.clear();
        b.aircon_control_state = AirconControlState::CapacityLimited;
        b.current_pre_temp = 15.4;
        b.current_requested_pre_temp = 20.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}, {"max", 3.867}}}, {"cooling", {{"rtd", 7.2}, {"max", 3.867}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.28}}}, {"cooling", {{"dsgn", 0.28}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties e{};
        e.key = "vb_in_b_cap";
        e.unique_id = "vb_in_b_cap";
        e.type = "fixed_flow";
        e.source = "IN";
        e.target = "B";
        e.current_vol = 0.28;
        e.flow_rate = 0.28;
        (void)boost::add_edge(vIn, vB, e, vg);

        FlowRateMap ductFlowRates;
        ductFlowRates[{"IN", "B"}] = 0.28;
        std::ostringstream logs1;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, ductFlowRates, logs1),
                   "capacity-limited duct flow should adjust once from Q_max");

        double flowAfterFirst = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            if (vg[boost::source(edge, vg)].key == "IN" && vg[boost::target(edge, vg)].key == "B") {
                flowAfterFirst = vg[edge].current_vol;
                break;
            }
        }
        const double expectedFromQmax = 0.28 * std::clamp(3867.0 / 7200.0, 0.0, 1.0);
        expectNear(flowAfterFirst, expectedFromQmax, 2e-4,
                   "capacity-limited duct flow should follow Q_max/Q_rtd * V_dsgn");

        // 再計算後に計測熱が減ったように見える（流量マップを縮小）しても、目標風量は変わらない
        ductFlowRates[{"IN", "B"}] = flowAfterFirst;
        b.current_t = 20.0 + (25.0 - 20.0) * (flowAfterFirst / 0.28);  // ΔT 維持気味でも Q∝V
        std::ostringstream logs2;
        const bool adjustedAgain =
            controller.checkAndAdjustDuctCentralAirflow(thermal, vent, ductFlowRates, logs2);
        expectTrue(!adjustedAgain,
                   "second duct adjust must not chase measured Q after capacity-limited target is set");

        double flowAfterSecond = flowAfterFirst;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            if (vg[boost::source(edge, vg)].key == "IN" && vg[boost::target(edge, vg)].key == "B") {
                flowAfterSecond = vg[edge].current_vol;
                break;
            }
        }
        expectNear(flowAfterSecond, flowAfterFirst, 1e-12,
                   "duct flow must stay at Q_max-based target (no collapse toward 0)");
    }

    // 要求設定未達（CapacityLimited 解除後）でも Q_max 風量を維持する
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        auto& out = thermal.getNode("OUT");
        thermal.addNode(makeNode("ROOM_UNMET", "normal", 16.0));
        auto& room = thermal.getNode("ROOM_UNMET");
        in.current_t = 16.0;
        b.current_t = 18.0;  // 小さいΔT → 計測熱は小さい
        out.current_t = 5.0;
        room.current_t = 16.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node = "ROOM_UNMET";
        b.outside_node = "OUT";
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_pre_temp = 20.0;
        b.current_requested_pre_temp = 20.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}, {"max", 3.867}}}, {"cooling", {{"rtd", 7.2}, {"max", 3.867}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.28}}}, {"cooling", {{"dsgn", 0.28}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties e{};
        e.key = "vb_unmet";
        e.unique_id = "vb_unmet";
        e.type = "fixed_flow";
        e.source = "IN";
        e.target = "B";
        e.current_vol = 0.01;
        e.flow_rate = 0.01;
        (void)boost::add_edge(vIn, vB, e, vg);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.01;
        std::ostringstream logs;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs),
                   "unmet setpoint should restore duct flow from Q_max");
        double adjustedFlow = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            if (vg[boost::source(edge, vg)].key == "IN" && vg[boost::target(edge, vg)].key == "B") {
                adjustedFlow = vg[edge].current_vol;
                break;
            }
        }
        const double expectedFromQmax = 0.28 * std::clamp(3867.0 / 7200.0, 0.0, 1.0);
        expectNear(adjustedFlow, expectedFromQmax, 2e-4,
                   "unmet room should use Q_max-based duct flow");
    }

    // ON+set_node で設定を保持できているときは室負荷で風量を決める。
    // 計測コイル熱で追従すると V∝Q∝V で縮小するので使わない。
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        auto& out = thermal.getNode("OUT");
        thermal.addNode(makeNode("ROOM_FIXED", "normal", 20.0));
        auto& room = thermal.getNode("ROOM_FIXED");
        in.current_t = 20.0;
        b.current_t = 21.0;  // 小さいΔT → 計測熱は小さい
        out.current_t = 5.0;
        room.current_t = 20.0;  // fixed-row で設定どおり
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node = "ROOM_FIXED";
        b.outside_node = "OUT";
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_pre_temp = 20.0;
        b.current_requested_pre_temp = 20.0;
        b.required_heat_w = 1800.0;  // 部分負荷。計測熱より大きく、Q_max より小さい
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}, {"max", 3.867}}}, {"cooling", {{"rtd", 7.2}, {"max", 3.867}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.28}}}, {"cooling", {{"dsgn", 0.28}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties e{};
        e.key = "vb_fixed";
        e.unique_id = "vb_fixed";
        e.type = "fixed_flow";
        e.source = "IN";
        e.target = "B";
        e.current_vol = 0.28;
        e.flow_rate = 0.28;
        (void)boost::add_edge(vIn, vB, e, vg);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.28;
        std::ostringstream logs1;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs1),
                   "setpoint-held duct flow should adjust from required heat");
        double flowAfterFirst = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            if (vg[boost::source(edge, vg)].key == "IN" && vg[boost::target(edge, vg)].key == "B") {
                flowAfterFirst = vg[edge].current_vol;
                break;
            }
        }
        const double expectedFromLoad = 0.28 * std::clamp(1800.0 / 7200.0, 0.0, 1.0);
        expectNear(flowAfterFirst, expectedFromLoad, 2e-4,
                   "setpoint-held duct flow should follow |Qreq|/Q_rtd * V_dsgn");

        flows[{"IN", "B"}] = flowAfterFirst;
        b.current_t = 20.0 + (21.0 - 20.0) * (flowAfterFirst / 0.28);
        std::ostringstream logs2;
        expectTrue(!controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs2),
                   "setpoint-held duct must not chase measured Q on second call");
    }

    // 室負荷が無い設定固定は Q_max に戻し、計測熱では縮小しない
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        thermal.addNode(makeNode("ROOM_FIXED_FALLBACK", "normal", 20.0));
        in.current_t = 20.0;
        b.current_t = 21.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node = "ROOM_FIXED_FALLBACK";
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_pre_temp = 20.0;
        b.current_requested_pre_temp = 20.0;
        b.required_heat_w = std::numeric_limits<double>::quiet_NaN();
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}, {"max", 3.867}}}, {"cooling", {{"rtd", 7.2}, {"max", 3.867}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.28}}}, {"cooling", {{"dsgn", 0.28}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties e{};
        e.key = "vb_fixed_fb";
        e.unique_id = "vb_fixed_fb";
        e.type = "fixed_flow";
        e.source = "IN";
        e.target = "B";
        e.current_vol = 0.28;
        e.flow_rate = 0.28;
        (void)boost::add_edge(vIn, vB, e, vg);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.28;
        std::ostringstream logs;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs),
                   "missing room load should fall back to Q_max");
        double adjustedFlow = 0.0;
        for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
            if (vg[boost::source(edge, vg)].key == "IN" && vg[boost::target(edge, vg)].key == "B") {
                adjustedFlow = vg[edge].current_vol;
                break;
            }
        }
        const double expectedFromQmax = 0.28 * std::clamp(3867.0 / 7200.0, 0.0, 1.0);
        expectNear(adjustedFlow, expectedFromQmax, 2e-4,
                   "missing room load uses Q_max-based duct flow");
    }

    // 設定維持: 負荷0→風量0、定格以上→設計風量、冷房は絶対値、Q_max で頭打ち
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        thermal.addNode(makeNode("ROOM_ENDS", "normal", 22.0));
        auto& room = thermal.getNode("ROOM_ENDS");
        in.current_t = 22.0;
        b.current_t = 24.0;
        room.current_t = 22.0;
        b.current_mode = "COOLING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node = "ROOM_ENDS";
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_pre_temp = 22.0;
        b.current_requested_pre_temp = 22.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}, {"max", 3.867}}}, {"cooling", {{"rtd", 7.2}, {"max", 3.867}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.28}}}, {"cooling", {{"dsgn", 0.28}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        const auto vSupply = boost::add_vertex(makeNode("SUPPLY_ENDS", "normal", 18.0), vg);
        auto addFixed = [&](const std::string& key, auto src, auto dst,
                            const std::string& source, const std::string& target, double vol) {
            EdgeProperties e{};
            e.key = key;
            e.unique_id = key;
            e.type = "fixed_flow";
            e.subtype = "aircon";
            e.source = source;
            e.target = target;
            e.current_vol = vol;
            e.flow_rate = vol;
            (void)boost::add_edge(src, dst, e, vg);
        };
        addFixed("ends_in", vIn, vB, "IN", "B", 0.2);
        addFixed("ends_out", vB, vSupply, "B", "SUPPLY_ENDS", 0.2);

        auto readPair = [&](double& intake, double& supply) {
            intake = 0.0;
            supply = 0.0;
            for (auto edge : boost::make_iterator_range(boost::edges(vg))) {
                const auto src = vg[boost::source(edge, vg)].key;
                const auto dst = vg[boost::target(edge, vg)].key;
                if (src == "IN" && dst == "B") intake = vg[edge].current_vol;
                if (src == "B" && dst == "SUPPLY_ENDS") supply = vg[edge].current_vol;
            }
        };

        b.required_heat_w = 0.0;
        FlowRateMap flows;
        flows[{"IN", "B"}] = 0.2;
        std::ostringstream logsZero;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsZero),
                   "zero room load should zero duct flow");
        double intake = 0.0;
        double supply = 0.0;
        readPair(intake, supply);
        expectNear(intake, 0.0, 1e-12, "zero load intake is 0");
        expectNear(supply, 0.0, 1e-12, "zero load supply is 0");

        // 冷房の負号負荷は絶対値。Q_max が定格以上なら、定格以上の負荷は設計風量。
        b.ac_spec["Q"]["cooling"]["max"] = 9.0;
        b.initializeAirconSpec();
        b.required_heat_w = -9000.0;
        flows[{"IN", "B"}] = 0.0;
        std::ostringstream logsMax;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsMax),
                   "load at rating should raise duct flow to design");
        readPair(intake, supply);
        expectNear(intake, 0.28, 2e-4, "cooling |Qreq| at Q.rtd uses V_dsgn");
        expectNear(supply, 0.28, 2e-4, "design flow is applied to supply too");

        // Q_max が定格未満のとき、室負荷が上限を超えても Q_max/Q_rtd * V_dsgn で止める
        b.ac_spec["Q"]["cooling"]["max"] = 3.867;
        b.initializeAirconSpec();
        b.required_heat_w = -5000.0;
        flows[{"IN", "B"}] = 0.28;
        std::ostringstream logsCap;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsCap),
                   "room load above Q_max should clamp duct flow");
        readPair(intake, supply);
        const double expectedClamp = 0.28 * std::clamp(3867.0 / 7200.0, 0.0, 1.0);
        expectNear(intake, expectedClamp, 2e-4, "room load is clamped to Q_max");
        expectNear(supply, expectedClamp, 2e-4, "clamped flow matches on supply");

        // カタログ Q.min だけでは風量床をかけない。明示 min_process が無ければ既定 10% of Q.rtd。
        b.current_mode = "HEATING";
        b.vol_zero = 0.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"min", 2.7}, {"rtd", 7.2}}}, {"cooling", {{"min", 2.7}, {"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.36}}}, {"cooling", {{"dsgn", 0.36}}}}},
        };
        b.initializeAirconSpec();
        b.required_heat_w = 160.0;
        flows[{"IN", "B"}] = 0.16;
        std::ostringstream logsDefaultMin;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsDefaultMin),
                   "default min_process floor should apply without explicit min_process");
        readPair(intake, supply);
        const double expectedDefaultMin = 0.36 * 0.1;  // 10% of Q.rtd
        expectNear(intake, expectedDefaultMin, 2e-4, "small load floors at 10% of Q.rtd by default");
        expectNear(supply, expectedDefaultMin, 2e-4, "default min flow matches on supply");
        expectTrue(logsDefaultMin.str().find("最低風量維持") != std::string::npos,
                   "default min-flow hold should be logged");

        // 明示 min_process があるとき、その以下の負荷は最低能力相当風量。
        b.ac_spec["min_process"] = nlohmann::json{{"heating", 2.7}, {"cooling", 2.7}};
        b.initializeAirconSpec();
        b.required_heat_w = 160.0;
        flows[{"IN", "B"}] = expectedDefaultMin;
        std::ostringstream logsMin;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsMin),
                   "min_process should hold minimum duct flow");
        readPair(intake, supply);
        const double expectedMin = 0.36 * (2.7 / 7.2);
        expectNear(intake, expectedMin, 2e-4, "load below min_process holds V_dsgn * min_process/Q.rtd");
        expectNear(supply, expectedMin, 2e-4, "minimum flow matches on supply");
        expectTrue(logsMin.str().find("最低風量維持") != std::string::npos,
                   "min-flow hold should be logged");

        // 明示 min_process が勝つ（既定 10% ではなく明示値で床）。
        // 負荷が既定 10% を超える場合は線形（明示なし）。
        // 直前の小負荷でホールドが立っているので、タイムステップ先頭相当で落とす。
        b.aircon_min_process_hold = false;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.36}}}, {"cooling", {{"dsgn", 0.36}}}}},
        };
        b.initializeAirconSpec();
        b.required_heat_w = 1200.0;  // 1.2 kW > 0.72 kW (10%)
        flows[{"IN", "B"}] = 0.05;
        std::ostringstream logsLinear;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsLinear),
                   "load above default 10% should stay linear");
        readPair(intake, supply);
        const double expectedLinear = 0.36 * (1200.0 / 7200.0);
        expectNear(intake, expectedLinear, 2e-4, "load above default min stays linear");
        expectNear(supply, expectedLinear, 2e-4, "linear flow matches on supply");
        expectTrue(logsLinear.str().find("最低風量維持") == std::string::npos,
                   "above-default load must not log min-flow hold");

        // Minimum flow is a continuous lower bound, independent of ON/OFF history.
        {
            VertexProperties specNode;
            specNode.model = "DUCT_CENTRAL";
            specNode.ac_spec = nlohmann::json{
                {"Q", {{"heating", {{"rtd", 7.73399}}}, {"cooling", {{"rtd", 7.73399}}}}},
                {"V_inner", {{"heating", {{"dsgn", 0.4}}}, {"cooling", {{"dsgn", 0.4}}}}},
            };
            specNode.initializeAirconSpec();
            const double minProcessW = 773.399;
            for (const auto mode : {OperationMode::Heating, OperationMode::Cooling}) {
                for (const bool wasHeld : {false, true}) {
                    specNode.aircon_min_process_hold = wasHeld;
                    double previous = 0.0;
                    for (const double heat : {771.815, minProcessW, minProcessW + 1e-5,
                                               844.34, 963.4, 1254.18}) {
                        bool held = false;
                        const auto flow = aircon::airflow::computeTargetFlowFromProcessedHeat(
                            specNode, mode, heat, 0.0, minProcessW, &held);
                        expectTrue(flow.has_value(), "positive load has a target flow");
                        expectTrue(*flow >= previous, "flow never decreases with required load");
                        expectTrue(held == (heat <= minProcessW), "only sub-minimum load uses floor");
                        expectNear(*flow, 0.4 * std::max(heat, minProcessW) / 7733.99, 1e-12,
                                   "ON/OFF history must not cap the fan at minimum speed");
                        previous = *flow;
                    }
                }
            }
        }

        // 未達 0.5 K 縁: 一度 Q_max に入ったら設定到達まで維持する。
        {
            VertexProperties node;
            node.type = "aircon";
            node.on = true;
            node.set_node = "ROOM";
            node.aircon_control_state = AirconControlState::SetpointControlled;
            node.current_requested_pre_temp = 20.0;
            node.current_pre_temp = 20.0;
            node.required_heat_w = 1800.0;
            node.ac_spec = nlohmann::json{
                {"Q", {{"heating", {{"rtd", 7.2}, {"max", 3.867}}}}},
            };
            node.initializeAirconSpec();
            const auto enter = aircon::airflow::selectFlowHeatBasis(
                node, OperationMode::Heating, 400.0, 19.4, /*previouslyUnmetBoost=*/false);
            expectTrue(enter.unmetBoost, "T=19.4 enters unmet Q_max boost");
            expectNear(enter.heatW, 3867.0, 1e-9, "enter unmet uses Q_max");

            const auto hold = aircon::airflow::selectFlowHeatBasis(
                node, OperationMode::Heating, 400.0, 19.7, /*previouslyUnmetBoost=*/true);
            expectTrue(hold.unmetBoost, "T=19.7 after unmet still holds Q_max");
            expectNear(hold.heatW, 3867.0, 1e-9, "hold unmet uses Q_max");

            const auto noEnter = aircon::airflow::selectFlowHeatBasis(
                node, OperationMode::Heating, 400.0, 19.7, /*previouslyUnmetBoost=*/false);
            expectTrue(!noEnter.unmetBoost, "T=19.7 without prior unmet does not enter");
            expectNear(noEnter.heatW, 1800.0, 1e-9, "inside 0.5K uses |Qreq|");

            const auto exitBoost = aircon::airflow::selectFlowHeatBasis(
                node, OperationMode::Heating, 400.0, 20.0, /*previouslyUnmetBoost=*/true);
            expectTrue(!exitBoost.unmetBoost, "T=setpoint exits unmet boost");
            expectNear(exitBoost.heatW, 1800.0, 1e-9, "exit unmet uses |Qreq|");
        }

        {
            expectTrue(!aircon::capacity::capacityExceedsLimit(7734.5, 7734.0),
                       "0.5W over capacity is within abs tol");
            expectTrue(aircon::capacity::capacityExceedsLimit(7800.0, 7734.0),
                       "66W over capacity exceeds limit");
            expectTrue(!aircon::capacity::capacityWellBelowLimit(7733.5, 7734.0),
                       "0.5W under capacity is not well below");
            expectTrue(aircon::capacity::capacityWellBelowLimit(7700.0, 7734.0),
                       "34W under capacity is well below");
        }

        // 負荷 0 は既定で風量 0。vol_zero があればその風量。
        b.required_heat_w = 0.0;
        flows[{"IN", "B"}] = expectedMin;
        std::ostringstream logsZeroDefault;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsZeroDefault),
                   "zero load should return to zero flow by default");
        readPair(intake, supply);
        expectNear(intake, 0.0, 1e-12, "zero load without vol_zero is 0");
        expectNear(supply, 0.0, 1e-12, "zero load supply without vol_zero is 0");

        b.vol_zero = 0.05;
        flows[{"IN", "B"}] = 0.0;
        std::ostringstream logsVolZero;
        expectTrue(controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logsVolZero),
                   "zero load with vol_zero should keep ventilation flow");
        readPair(intake, supply);
        expectNear(intake, 0.05, 2e-4, "zero load uses vol_zero on intake");
        expectNear(supply, 0.05, 2e-4, "zero load uses vol_zero on supply");
    }

    // 極小風量同士の差では再計算しない（0.00→0.00 ループ防止）
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        auto& out = thermal.getNode("OUT");
        // 入口と吹出が同温なら処理熱≈0 → 目標は vol_zero（0）。
        // 現風量 5e-5 も 1e-4 未満で 0 扱いなので、再計算を要求しない。
        in.current_t = 20.0;
        b.current_t = 20.0;
        out.current_t = 7.0;
        b.current_mode = "HEATING";
        b.on = true;
        b.model = "DUCT_CENTRAL";
        b.in_node = "IN";
        b.set_node.clear();
        b.outside_node = "OUT";
        b.vol_zero = 0.0;
        b.aircon_control_state = AirconControlState::SetpointControlled;
        b.current_pre_temp = 20.0;
        b.current_requested_pre_temp = 20.0;
        b.required_heat_w = 0.0;
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.28}}}, {"cooling", {{"dsgn", 0.28}}}}},
        };
        b.initializeAirconSpec();

        VentilationNetwork vent;
        auto& vg = vent.getGraph();
        const auto vIn = boost::add_vertex(makeNode("IN", "normal", in.current_t), vg);
        const auto vB = boost::add_vertex(makeNode("B", "aircon", b.current_t), vg);
        EdgeProperties e{};
        e.key = "vb_tiny";
        e.unique_id = "vb_tiny";
        e.type = "fixed_flow";
        e.source = "IN";
        e.target = "B";
        e.current_vol = 5e-5;
        e.flow_rate = 5e-5;
        (void)boost::add_edge(vIn, vB, e, vg);

        FlowRateMap flows;
        flows[{"IN", "B"}] = 5e-5;
        std::ostringstream logs;
        expectTrue(!controller.checkAndAdjustDuctCentralAirflow(thermal, vent, flows, logs),
                   "tiny-to-tiny duct flow must not request recompute");
    }

    // AUTOモード: 室内温と吹出温の関係で operationMode が cooling/heating に分岐すること
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        b.on = true;
        b.current_mode = "AUTO";
        b.in_node = "IN";

        // indoor > airconTemp -> cooling
        in.current_t = 27.0;
        b.current_t = 20.0;
        calls = 0;
        history.clear();
        (void)controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(calls == 1, "AUTO mode (cooling path): estimateCOP called");
        expectTrue(!history.empty() && history.back().mode == "cooling",
                   "AUTO mode (cooling path): mode should be cooling");

        // indoor <= airconTemp -> heating
        in.current_t = 20.0;
        b.current_t = 24.0;
        calls = 0;
        history.clear();
        (void)controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(calls == 1, "AUTO mode (heating path): estimateCOP called");
        expectTrue(!history.empty() && history.back().mode == "heating",
                   "AUTO mode (heating path): mode should be heating");
    }

    // 複数エアコンが同じ set_node を持つ場合、潜熱フィードバック注入をスキップすること
    {
        auto& in = thermal.getNode("IN");
        auto& a = thermal.getNode("A");
        auto& b = thermal.getNode("B");
        in.current_t = 27.0;
        in.current_x = 0.020;
        in.heat_source = 0.0;
        a.current_t = 20.0;
        b.current_t = 20.0;
        a.current_mode = "COOLING";
        b.current_mode = "COOLING";
        a.in_node = "IN";
        b.in_node = "IN";
        a.set_node = "IN";
        b.set_node = "IN";
        a.on = true;
        b.on = true;

        const auto stats = controller.applyLatentFeedbackToThermal(thermal, flowRates, 1.0, std::cout);
        expectNear(in.heat_source, 0.0, 1e-12,
                   "latent feedback should be skipped when in_node is active setpoint node");
        expectNear(stats.maxAppliedHeatW, 0.0, 1e-12,
                   "latent feedback stats should remain zero when skipped");
    }

    // 異常系: set_node が存在しない場合は黙って 0℃ にせず例外
    {
        auto& b = thermal.getNode("B");
        b.on = true;
        b.current_mode = "COOLING";
        b.in_node = "IN";
        b.set_node = "NO_SUCH_SET";
        bool threw = false;
        try {
            (void)controller.controlAllAircons(thermal, 0.5, std::cout);
        } catch (const std::exception&) {
            threw = true;
        }
        expectTrue(threw, "missing set_node should throw");
        b.set_node.clear();
    }

    // 異常系: in_node が不正なら例外を握りつぶして電力0で継続すること
    {
        auto& b = thermal.getNode("B");
        b.on = true;
        b.current_mode = "COOLING";
        b.in_node = "NO_SUCH_NODE";

        calls = 0;
        history.clear();
        auto powerW = controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(powerW.size() == 2, "invalid in_node case: power vector size");
        if (powerW.size() == 2) {
            expectNear(powerW[1], 0.0, 0.0, "invalid in_node case: power should fall back to 0");
        }
        expectTrue(calls == 1, "invalid in_node case: only valid unit should call estimateCOP");
    }

    // IDEAL（モデルなし）も制御キーに入る
    {
        ThermalNetwork tIdeal;
        tIdeal.addNode(makeNode("R", "normal", 20.0));
        auto ac = makeNode("AC_IDEAL", "aircon", 20.0);
        ac.set_node = "R";
        ac.model = "IDEAL";
        tIdeal.addNode(ac);
        AirconController cIdeal;
        std::ostringstream logs;
        cIdeal.initializeModels(tIdeal, logs, 0);
        const auto& keys = cIdeal.getAirconKeys();
        expectTrue(keys.size() == 1 && keys[0] == "AC_IDEAL", "IDEAL aircon must be in getAirconKeys");
        expectTrue(cIdeal.getModel("AC_IDEAL") == nullptr, "IDEAL has no COP model");
    }

    // applyPreset: モード継続時は前ステップの ON/OFF を維持
    {
        ThermalNetwork tP;
        tP.addNode(makeNode("R", "normal", 20.0));
        auto ac = makeNode("AC_P", "aircon", 20.0);
        ac.set_node = "R";
        ac.model = "IDEAL";
        ac.current_mode = "HEATING";
        tP.addNode(ac);
        AirconController cP;
        std::ostringstream logs;
        cP.initializeModels(tP, logs, 0);
        cP.applyPreset(tP, logs);
        expectTrue(tP.getNode("AC_P").on, "first preset starts ON");
        tP.getNode("AC_P").on = false;
        cP.applyPreset(tP, logs);
        expectTrue(!tP.getNode("AC_P").on, "continued HEATING keeps previous OFF");
        tP.getNode("AC_P").current_mode = "COOLING";
        cP.applyPreset(tP, logs);
        expectTrue(tP.getNode("AC_P").on, "mode switch HEATING→COOLING restarts ON");
    }

    // 符号付き処理熱量ヘルパ
    {
        const double qHeat = thermal_moist_air::signedProcessedHeatW(
            20.0, 0.0, 30.0, 0.0, 0.1, /*moist=*/false);
        const double qCool = thermal_moist_air::signedProcessedHeatW(
            26.0, 0.0, 16.0, 0.0, 0.1, /*moist=*/false);
        expectTrue(qHeat > 0.0, "heating deltaT → positive signed heat");
        expectTrue(qCool < 0.0, "cooling deltaT → negative signed heat");
        FlowRateMap fr;
        fr[{"IN", "B"}] = 0.1;
        thermal.getNode("IN").current_t = 20.0;
        thermal.getNode("B").current_t = 30.0;
        const double qCtrl = controller.calculateSignedProcessedHeat(thermal, "IN", "B", fr);
        expectNear(qCtrl, qHeat, 1e-6, "controller signed heat matches helper");
    }

    // 熱ソルバが required_heat_w を符号付き処理熱量として書く
    {
        auto makeSolveConstants = []() {
            SimulationConstants c{};
            c.temperatureCalc = true;
            c.thermalTolerance = 1e-3;
            c.thermalBalanceToleranceW = 1.0;
            c.timestep = 3600;
            return c;
        };

        auto addAirconLoop = [](ThermalNetwork& net, const std::string& roomKey,
                                const std::string& acKey, const std::string& outKey,
                                double flow, bool acBeforeRoom) {
            VertexProperties out{};
            out.key = outKey;
            out.type = "normal";
            out.calc_t = false;
            out.current_t = 0.0;
            VertexProperties room{};
            room.key = roomKey;
            room.type = "normal";
            room.calc_t = true;
            room.current_t = 20.0;
            VertexProperties ac{};
            ac.key = acKey;
            ac.type = "aircon";
            ac.calc_t = true;
            ac.on = true;
            ac.set_node = roomKey;
            ac.in_node = roomKey;
            ac.current_pre_temp = 20.0;
            ac.current_requested_pre_temp = 20.0;
            ac.current_t = 20.0;
            ac.current_mode = "HEATING";
            if (acBeforeRoom) {
                net.addNode(ac);
                net.addNode(room);
            } else {
                net.addNode(room);
                net.addNode(ac);
            }
            net.addNode(out);

            EdgeProperties cond{};
            cond.key = "cond_" + roomKey;
            cond.unique_id = cond.key;
            cond.type = "conductance";
            cond.subtype = "conduction";
            cond.source = roomKey;
            cond.target = outKey;
            cond.conductance = 50.0; // 外気 0℃・設定 20℃ → 約 1000W 暖房需要
            net.addEdge(cond);

            EdgeProperties ret{};
            ret.key = "ret_" + acKey;
            ret.unique_id = ret.key;
            ret.type = "advection";
            ret.source = roomKey;
            ret.target = acKey;
            ret.flow_rate = flow;
            ret.is_aircon_inflow = true;
            net.addEdge(ret);

            EdgeProperties sup{};
            sup.key = "sup_" + acKey;
            sup.unique_id = sup.key;
            sup.type = "advection";
            sup.source = acKey;
            sup.target = roomKey;
            sup.flow_rate = flow;
            net.addEdge(sup);
        };

        // 暖房需要: required_heat_w > 0
        {
            ThermalNetwork net;
            addAirconLoop(net, "ROOM", "AC", "OUT", 0.2, /*acBeforeRoom=*/false);
            std::ostringstream logs;
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, makeSolveConstants(), logs);
            const double q = net.getNode("AC").required_heat_w;
            expectTrue(std::isfinite(q) && q > 100.0, "heating load case: required_heat_w > 0");
            // 外気0℃・設定20℃・conductance=50 → 約1000W の暖房負荷（容量過渡を含む）
            expectTrue(q > 800.0, "heating load roughly matches UA*dT scale");
        }
        // 頂点順を逆にしても同じ
        {
            ThermalNetwork net;
            addAirconLoop(net, "ROOM2", "AC2", "OUT2", 0.2, /*acBeforeRoom=*/true);
            std::ostringstream logs;
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, makeSolveConstants(), logs);
            const double q = net.getNode("AC2").required_heat_w;
            expectTrue(std::isfinite(q) && q > 100.0, "vertex-order independent heating Qreq > 0");
        }
        // 冷房需要: 外気高温・設定低め
        {
            ThermalNetwork net;
            addAirconLoop(net, "ROOMC", "ACC", "OUTC", 0.2, false);
            net.getNode("OUTC").current_t = 35.0;
            net.getNode("ACC").current_pre_temp = 24.0;
            net.getNode("ACC").current_requested_pre_temp = 24.0;
            net.getNode("ACC").current_mode = "COOLING";
            net.getNode("ROOMC").current_t = 24.0;
            std::ostringstream logs;
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, makeSolveConstants(), logs);
            const double q = net.getNode("ACC").required_heat_w;
            expectTrue(std::isfinite(q) && q < -100.0, "cooling load case: required_heat_w < 0");
        }
        // 負荷なしに近い: 外気=設定、伝導のみ → Q≈0
        {
            ThermalNetwork net;
            addAirconLoop(net, "ROOM0", "AC0", "OUT0", 0.2, false);
            net.getNode("OUT0").current_t = 20.0;
            std::ostringstream logs;
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, makeSolveConstants(), logs);
            const double q = net.getNode("AC0").required_heat_w;
            expectTrue(std::isfinite(q) && std::abs(q) < 5.0, "no-load case: required_heat_w ≈ 0");
        }
        // 遠隔 set: set=LDK, in/out=階間（AC↔set 直結なし）。階間→LDK の伝導で間接制御。
        {
            ThermalNetwork net;
            VertexProperties out{};
            out.key = "OUT_R";
            out.type = "normal";
            out.calc_t = false;
            out.current_t = 0.0;
            VertexProperties ldk{};
            ldk.key = "LDK";
            ldk.type = "normal";
            ldk.calc_t = true;
            ldk.current_t = 20.0;
            VertexProperties zone{};
            zone.key = "ZONE";
            zone.type = "normal";
            zone.calc_t = true;
            zone.current_t = 20.0;
            VertexProperties ac{};
            ac.key = "AC_R";
            ac.type = "aircon";
            ac.calc_t = true;
            ac.on = true;
            ac.set_node = "LDK";
            ac.in_node = "ZONE";
            ac.current_pre_temp = 20.0;
            ac.current_requested_pre_temp = 20.0;
            ac.current_t = 20.0;
            ac.current_mode = "HEATING";
            net.addNode(ldk);
            net.addNode(zone);
            net.addNode(ac);
            net.addNode(out);

            EdgeProperties loss{};
            loss.key = "loss_ldk";
            loss.unique_id = loss.key;
            loss.type = "conductance";
            loss.subtype = "conduction";
            loss.source = "LDK";
            loss.target = "OUT_R";
            loss.conductance = 50.0;
            net.addEdge(loss);

            EdgeProperties couple{};
            couple.key = "zone_ldk";
            couple.unique_id = couple.key;
            couple.type = "conductance";
            couple.subtype = "conduction";
            couple.source = "ZONE";
            couple.target = "LDK";
            couple.conductance = 200.0;
            net.addEdge(couple);

            EdgeProperties ret{};
            ret.key = "ret_remote";
            ret.unique_id = ret.key;
            ret.type = "advection";
            ret.source = "ZONE";
            ret.target = "AC_R";
            ret.flow_rate = 0.2;
            ret.is_aircon_inflow = true;
            net.addEdge(ret);

            EdgeProperties sup{};
            sup.key = "sup_remote";
            sup.unique_id = sup.key;
            sup.type = "advection";
            sup.source = "AC_R";
            sup.target = "ZONE";
            sup.flow_rate = 0.2;
            net.addEdge(sup);

            std::ostringstream logs;
            ThermalSolverLinearDirect::resetDirectTSolverContext();
            ThermalSolverLinearDirect::solveTemperaturesLinearDirect(net, makeSolveConstants(), logs);
            const double q = net.getNode("AC_R").required_heat_w;
            expectTrue(std::isfinite(q) && q > 100.0,
                       "remote-set heating: required_heat_w > 0 (coil fallback)");
            expectNear(net.getNode("LDK").current_t, 20.0, 1e-3,
                       "remote-set heating: LDK held at setpoint");
        }
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


