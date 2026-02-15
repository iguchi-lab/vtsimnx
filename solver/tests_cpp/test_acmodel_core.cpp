#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include "acmodel/acmodel.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectNear(double actual, double expected, double absTol, const std::string& msg) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        fail(msg + " (non-finite)");
        return;
    }
    if (std::abs(actual - expected) > absTol) {
        std::ostringstream oss;
        oss << msg << " (actual=" << actual << ", expected=" << expected << ", absTol=" << absTol << ")";
        fail(oss.str());
    }
}

template <class Fn>
void expectThrows(Fn fn, const std::string& msg) {
    try {
        fn();
        fail(msg + " (expected throw)");
    } catch (const std::exception&) {
        // ok
    }
}

} // namespace

int main() {
    using namespace acmodel;

    // -----------------------------
    // getTypeFromString: mapping + unknown throws
    // -----------------------------
    {
        expectTrue(AirconModelFactory::getTypeFromString("CRIEPI") == AirconType::CRIEPI, "type: CRIEPI");
        expectTrue(AirconModelFactory::getTypeFromString("RAC") == AirconType::RAC, "type: RAC");
        expectTrue(AirconModelFactory::getTypeFromString("DUCT_CENTRAL") == AirconType::DUCT_CENTRAL, "type: DUCT_CENTRAL");
        expectTrue(AirconModelFactory::getTypeFromString("LATENT_EVALUATE") == AirconType::LATENT_EVALUATE, "type: LATENT_EVALUATE");
        expectThrows([&]() { (void)AirconModelFactory::getTypeFromString("UNKNOWN"); }, "type: unknown throws");
    }

    // -----------------------------
    // logger + verbosity gating + prefix
    // -----------------------------
    {
        std::vector<std::string> captured;
        setLogger([&](const std::string& s) { captured.push_back(s); });

        setLogVerbosity(1);
        log("hello", 1);
        log("hidden", 2);
        expectTrue(captured.size() == 1, "verbosity=1 logs only level<=1");
        if (!captured.empty()) {
            expectTrue(captured[0] == std::string("　　[acmodel] ") + "hello", "logger: prefix added exactly once");
        }

        setLogVerbosity(2);
        log("show", 2);
        expectTrue(captured.size() == 2, "verbosity=2 logs level<=2");
        if (captured.size() >= 2) {
            expectTrue(captured[1] == std::string("　　[acmodel] ") + "show", "logger: prefix (level=2)");
        }

        // cleanup
        setLogger(nullptr);
    }

    // -----------------------------
    // CRIEPI: coefficients/Pc regression (highspec/standard)
    // -----------------------------
    {
        // ログはテストでは不要（出すとノイズになる）なので無効化
        setLogger(nullptr);

        auto runCase = [&](const std::string& name, const nlohmann::json& spec,
                           double pcCoolingExp, const std::vector<double>& coeffCoolingExp,
                           double pcHeatingExp, const std::vector<double>& coeffHeatingExp) {
            std::unique_ptr<AirconSpec> model;
            try {
                model = AirconModelFactory::createModel("CRIEPI", spec);
            } catch (const std::exception& e) {
                fail("CRIEPI createModel failed (" + name + "): " + e.what());
                return;
            }

            nlohmann::json params;
            try {
                params = model->getModelParameters();
            } catch (const std::exception& e) {
                fail("CRIEPI getModelParameters failed (" + name + "): " + e.what());
                return;
            }

            // 期待値は「もちろん計算誤差あり」だが、回帰としてはもう少し厳しめに見る
            // 現状の実測（ビルド環境差）でブレやすい項があるため、
            // それでも回帰として意味がある範囲まで厳しくする。
            const double pcTol = 1.3e-3;     // kW
            const double coeffTol = 1.3e-2;  // 無次元

            try {
                expectNear(params.at("constant_power").at("cooling").get<double>(), pcCoolingExp, pcTol,
                           "CRIEPI Pc cooling (" + name + ")");
                expectNear(params.at("constant_power").at("heating").get<double>(), pcHeatingExp, pcTol,
                           "CRIEPI Pc heating (" + name + ")");

                auto coeffC = params.at("coefficients").at("cooling");
                auto coeffH = params.at("coefficients").at("heating");
                expectTrue(coeffC.is_array() && coeffC.size() == 3, "CRIEPI coeff cooling shape (" + name + ")");
                expectTrue(coeffH.is_array() && coeffH.size() == 3, "CRIEPI coeff heating shape (" + name + ")");

                for (int i = 0; i < 3; i++) {
                    expectNear(coeffC.at(i).get<double>(), coeffCoolingExp.at(i), coeffTol,
                               "CRIEPI coeff cooling[" + std::to_string(i) + "] (" + name + ")");
                    expectNear(coeffH.at(i).get<double>(), coeffHeatingExp.at(i), coeffTol,
                               "CRIEPI coeff heating[" + std::to_string(i) + "] (" + name + ")");
                }
            } catch (const std::exception& e) {
                fail("CRIEPI params access failed (" + name + "): " + e.what());
            }
        };

        // 入力spec（Python側辞書と同等構造）
        const nlohmann::json highspec = {
            {"Q", {{"cooling", {{"min", 0.700}, {"rtd", 2.200}, {"max", 3.300}}},
                   {"heating", {{"min", 0.700}, {"rtd", 2.500}, {"max", 5.400}}}}},
            {"P", {{"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.780}}},
                   {"heating", {{"min", 0.095}, {"rtd", 0.390}, {"max", 1.360}}}}},
            {"V_inner", {{"cooling", {{"rtd", 12.1 / 60.0}}}, {"heating", {{"rtd", 13.1 / 60.0}}}}},
            {"V_outer", {{"cooling", {{"rtd", 28.2 / 60.0}}}, {"heating", {{"rtd", 25.5 / 60.0}}}}}
        };

        const nlohmann::json standard = {
            {"Q", {{"cooling", {{"min", 0.900}, {"rtd", 2.200}, {"max", 2.800}}},
                   {"heating", {{"min", 0.900}, {"rtd", 2.200}, {"max", 3.600}}}}},
            {"P", {{"cooling", {{"min", 0.170}, {"rtd", 0.455}, {"max", 0.745}}},
                   {"heating", {{"min", 0.135}, {"rtd", 0.385}, {"max", 1.070}}}}},
            {"V_inner", {{"cooling", {{"rtd", 12.0 / 60.0}}}, {"heating", {{"rtd", 12.0 / 60.0}}}}},
            {"V_outer", {{"cooling", {{"rtd", 27.6 / 60.0}}}, {"heating", {{"rtd", 22.5 / 60.0}}}}}
        };

        // 期待値（計算誤差は許容）
        runCase("highspec",
                highspec,
                /*Pc cooling*/ 0.0361, /*coeff cooling*/ {-0.018, 0.052, 0.513},
                /*Pc heating*/ 0.0303, /*coeff heating*/ {-0.006, 0.019, 0.626});

        runCase("standard",
                standard,
                /*Pc cooling*/ 0.0820, /*coeff cooling*/ {-0.082, 0.255, 0.365},
                /*Pc heating*/ 0.0300, /*coeff heating*/ {-0.044, 0.136, 0.479});
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


