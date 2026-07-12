#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "aircon/aircon_controller.h"
#include "aircon/aircon_capacity.h"
#include "aircon/aircon_network_utils.h"
#include "aircon/aircon_operation_mode.h"
#include "core/thermal/thermal_moist_air.h"
#include "core/thermal/thermal_solver_linear_direct.h"
#include "network/thermal_network.h"
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

} // namespace

int main() {
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

    // OFF（送風継続）: 乾燥 supplyX を入口湿度へ戻し、再計算要求
    {
        auto& b = thermal.getNode("B");
        auto& in = thermal.getNode("IN");
        b.on = false;
        b.current_mode = "COOLING";
        b.in_node = "IN";
        b.current_x = 0.008;
        b.aircon_moisture_removal_kg_s = 0.001;
        in.current_x = 0.020;
        bool supplyChanged = false;
        const bool controlled =
            controller.controlAllAircons(thermal, 0.5, std::cout, &supplyChanged, 1e-9);
        expectTrue(controlled, "already OFF stays controlled");
        expectTrue(supplyChanged, "OFF passthrough requests supply humidity recompute");
        expectNear(b.current_x, 0.020, 1e-15, "OFF follows inlet humidity");
        expectNear(b.aircon_moisture_removal_kg_s, 0.0, 0.0, "OFF clears condensation");
        // 継続OFFで入口が変わると再追従
        in.current_x = 0.018;
        supplyChanged = false;
        (void)controller.controlAllAircons(thermal, 0.5, std::cout, &supplyChanged, 1e-9);
        expectTrue(supplyChanged, "OFF continues tracking inlet x");
        expectNear(b.current_x, 0.018, 1e-15, "OFF tracks updated inlet");
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

    // DUCT_CENTRAL / LATENT_EVALUATE 用: Q.max が無く Q.mid のみの場合も能力上限として扱う
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

        expectTrue(adjusted, "Q.mid only (no max) should still apply capacity limit");
        expectTrue(b.current_pre_temp < 26.0, "heating setpoint should decrease when over mid capacity");
        expectTrue(logs.str().find("Q.heating.mid") != std::string::npos,
                   "source label should indicate mid fallback");
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
        b.ac_spec = nlohmann::json{
            {"Q", {{"heating", {{"rtd", 7.2}}}, {"cooling", {{"rtd", 7.2}}}}},
            {"V_inner", {{"heating", {{"dsgn", 0.2}}}, {"cooling", {{"dsgn", 0.2}}}}},
        };

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
            const double qW = 1.2 * 1005.0 * 0.3 * (30.0 - 20.0);
            const double targetFlow = 0.2 * std::clamp(qW / (7.2 * 1000.0), 0.0, 1.0);
            expectNear(adjustedFlow, targetFlow, 2e-4, "duct_central target flow should follow Q/Q_rtd * V_dsgn");
        }
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


