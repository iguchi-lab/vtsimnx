#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"
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
    v.current_pre_temp = 24.0;
    v.on = false;
    return v;
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

} // namespace

int main() {
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
        b.ac_spec = nlohmann::json{
            {"latent_method", "coil_aoaf"},
            {"Af", 0.133},
            {"Ao", 4.84},
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

        // Ao を大きくすると潜熱側の処理量が増える傾向になることを確認
        b.ac_spec = nlohmann::json{
            {"latent_method", "coil_aoaf"},
            {"Af", 0.133},
            {"Ao", 9.68},
        };
        calls = 0;
        history.clear();
        (void)controller.calculatePowerValues(thermal, flowRates, std::cout);
        expectTrue(calls == 1, "coil_aoaf(Ao=9.68): estimateCOP should be called");
        expectTrue(!history.empty(), "coil_aoaf(Ao=9.68): history should have one entry");
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

    // sensibleHeatCapacity も OFF は 0、ON のみ正値になること
    {
        auto sensible = controller.collectAirconDataValues(thermal, flowRates, "sensibleHeatCapacity");
        expectTrue(sensible.size() == 2, "sensible heat size");
        if (sensible.size() == 2) {
            expectNear(sensible[0], 0.0, 0.0, "A sensible heat=0 when off");
            expectTrue(sensible[1] > 0.0, "B sensible heat > 0 when on");
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
        b.current_pre_temp = 26.0;
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
    }

    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 30.0;
        b.current_t = 22.0;
        b.current_mode = "COOLING";
        b.current_pre_temp = 24.0;
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
    }

    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 18.0;
        b.current_t = 24.0;
        b.current_mode = "HEATING";
        b.current_pre_temp = 26.0;
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

    // DUCT_CENTRAL / LATENT_EVALUATE 用: Q.max が無く Q.mid のみの場合も能力上限として扱う
    {
        auto& in = thermal.getNode("IN");
        auto& b = thermal.getNode("B");
        in.current_t = 18.0;
        b.current_t = 24.0;
        b.current_mode = "HEATING";
        b.current_pre_temp = 26.0;
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

        expectTrue(adjusted, "Q.mid only (no max) should still apply capacity limit");
        expectTrue(b.current_pre_temp < 26.0, "heating setpoint should decrease when over mid capacity");
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
        b.current_pre_temp = 20.0;
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

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


