#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"

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

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


