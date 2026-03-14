#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "acmodel/acmodel.h"
#include "../archenv/include/archenv.h"

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

            // JIS条件での回帰:
            // highspec / standard から生成したモデルに JIS 条件を与えたとき、
            // 各 rating(min/rtd/max) の入力能力 Q に対する推定電力 power[kW] が
            // 元 spec の P に十分近いことを確認する。
            const auto runJisPoint = [&](const std::string& mode,
                                         const std::string& rating,
                                         double tIn, double tEx,
                                         double xIn, double xEx) {
                InputData in{};
                in.T_in = tIn;
                in.T_ex = tEx;
                in.X_in = xIn;
                in.X_ex = xEx;
                in.Q = spec.at("Q").at(mode).at(rating).get<double>() * 1000.0; // kW -> W
                // spec上は rtd 風量のみ保持しているため、全 rating で同じ値を使用
                in.V_inner = spec.at("V_inner").at(mode).at("rtd").get<double>();
                in.V_outer = spec.at("V_outer").at(mode).at("rtd").get<double>();

                COPResult out{};
                try {
                    out = model->estimateCOP(mode, in);
                } catch (const std::exception& e) {
                    fail("CRIEPI estimateCOP failed (" + name + ", " + mode + ", " + rating + "): " + e.what());
                    return;
                }
                expectTrue(out.valid, "CRIEPI estimateCOP valid (" + name + ", " + mode + ", " + rating + ")");

                const double expectedPowerKw = spec.at("P").at(mode).at(rating).get<double>();
                const double powerTolKw = 7.0e-2; // JIS点の回帰許容（収束/近似差を吸収）
                expectNear(out.power, expectedPowerKw, powerTolKw,
                           "CRIEPI JIS power (" + name + ", " + mode + ", " + rating + ")");
            };

            for (const auto& rating : {"min", "rtd", "max"}) {
                runJisPoint("cooling", rating,
                            archenv::jis::T_C_IN, archenv::jis::T_C_EX,
                            archenv::jis::X_C_IN, archenv::jis::X_C_EX);
                runJisPoint("heating", rating,
                            archenv::jis::T_H_IN, archenv::jis::T_H_EX,
                            archenv::jis::X_H_IN, archenv::jis::X_H_EX);
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

    // -----------------------------
    // CRIEPI: 暖房の係数テーブル値（R, Pc）回帰
    // ユーザー提示の表値に対して、暖房側を明示チェックする。
    // -----------------------------
    {
        setLogger(nullptr);

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

        const auto checkHeatingTable = [&](const std::string& name,
                                           const nlohmann::json& spec,
                                           double pcHeatingKw,
                                           const std::vector<double>& coeffHeating) {
            std::unique_ptr<AirconSpec> model;
            try {
                model = AirconModelFactory::createModel("CRIEPI", spec);
            } catch (const std::exception& e) {
                fail("CRIEPI heating table createModel failed (" + name + "): " + e.what());
                return;
            }

            nlohmann::json params;
            try {
                params = model->getModelParameters();
            } catch (const std::exception& e) {
                fail("CRIEPI heating table getModelParameters failed (" + name + "): " + e.what());
                return;
            }

            const double pcTol = 1.3e-3;
            const double coeffTol = 1.3e-2;
            try {
                expectNear(params.at("constant_power").at("heating").get<double>(),
                           pcHeatingKw, pcTol,
                           "CRIEPI heating table Pc (" + name + ")");

                auto coeffH = params.at("coefficients").at("heating");
                expectTrue(coeffH.is_array() && coeffH.size() == 3,
                           "CRIEPI heating table coeff shape (" + name + ")");
                for (int i = 0; i < 3; i++) {
                    expectNear(coeffH.at(i).get<double>(), coeffHeating.at(i), coeffTol,
                               "CRIEPI heating table coeff[" + std::to_string(i) + "] (" + name + ")");
                }
            } catch (const std::exception& e) {
                fail("CRIEPI heating table params access failed (" + name + "): " + e.what());
            }
        };

        // 表値:
        // - highspec heating: R = -0.006 Q^2 + 0.019 Q + 0.636, Pc=30.3W(=0.0303kW)
        // - standard heating: R = -0.044 Q^2 + 0.136 Q + 0.479, Pc=30.0W(=0.0300kW)
        checkHeatingTable("highspec", highspec, 0.0303, {-0.006, 0.019, 0.636});
        checkHeatingTable("standard", standard, 0.0300, {-0.044, 0.136, 0.479});
    }

    // -----------------------------
    // CRIEPI highspec: 図ベース近似回帰（冷房 COP-負荷曲線）
    // 画像から読み取った近似点を使い、過度に厳密でない回帰チェックを行う。
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json highspec = {
            {"Q", {{"cooling", {{"min", 0.700}, {"rtd", 2.200}, {"max", 3.300}}},
                   {"heating", {{"min", 0.700}, {"rtd", 2.500}, {"max", 5.400}}}}},
            {"P", {{"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.780}}},
                   {"heating", {{"min", 0.095}, {"rtd", 0.390}, {"max", 1.360}}}}},
            {"V_inner", {{"cooling", {{"rtd", 12.1 / 60.0}}}, {"heating", {{"rtd", 13.1 / 60.0}}}}},
            {"V_outer", {{"cooling", {{"rtd", 28.2 / 60.0}}}, {"heating", {{"rtd", 25.5 / 60.0}}}}}
        };

        std::unique_ptr<AirconSpec> model;
        try {
            model = AirconModelFactory::createModel("CRIEPI", highspec);
        } catch (const std::exception& e) {
            fail(std::string("CRIEPI highspec createModel failed (figure approx): ") + e.what());
            model.reset();
        }

        if (model) {
            const double tIn = 27.0;
            const double rhIn = 47.1;
            const double xIn = archenv::absolute_humidity(tIn, rhIn);
            const double vInner = highspec.at("V_inner").at("cooling").at("rtd").get<double>();
            const double vOuter = highspec.at("V_outer").at("cooling").at("rtd").get<double>();

            const auto estimateCopAt = [&](double tEx, double rhEx, double qKw) -> double {
                InputData in{};
                in.T_in = tIn;
                in.T_ex = tEx;
                in.X_in = xIn;
                in.X_ex = archenv::absolute_humidity(tEx, rhEx);
                in.Q = qKw * 1000.0; // kW -> W
                in.V_inner = vInner;
                in.V_outer = vOuter;
                COPResult out = model->estimateCOP("cooling", in);
                expectTrue(out.valid, "CRIEPI highspec figure approx: estimateCOP valid");
                return out.COP;
            };

            // 図からの近似読み取り点（冷房, 室温27℃/室内RH47.1%/外気RH40.5%）
            // 25℃, 30℃, 35℃ それぞれ Q=[0.7, 2.2, 3.3]kW で評価
            const double rhEx = 40.5;
            const double qMin = 0.7;
            const double qRtd = 2.2;
            const double qMax = 3.3;

            const double cop25Min = estimateCopAt(25.0, rhEx, qMin);
            const double cop25Rtd = estimateCopAt(25.0, rhEx, qRtd);
            const double cop25Max = estimateCopAt(25.0, rhEx, qMax);

            const double cop30Min = estimateCopAt(30.0, rhEx, qMin);
            const double cop30Rtd = estimateCopAt(30.0, rhEx, qRtd);
            const double cop30Max = estimateCopAt(30.0, rhEx, qMax);

            const double cop35Min = estimateCopAt(35.0, rhEx, qMin);
            const double cop35Rtd = estimateCopAt(35.0, rhEx, qRtd);
            const double cop35Max = estimateCopAt(35.0, rhEx, qMax);

            // 図の○点（メーカー公表値）近傍: 35℃線の min/rtd/max
            const double figTolCop = 0.8;
            expectNear(cop35Min, 7.4, figTolCop, "CRIEPI highspec figure approx: COP35@0.7kW");
            expectNear(cop35Rtd, 5.6, figTolCop, "CRIEPI highspec figure approx: COP35@2.2kW");
            expectNear(cop35Max, 4.3, figTolCop, "CRIEPI highspec figure approx: COP35@3.3kW");

            // 温度依存の序列（同一負荷で 25℃ > 30℃ > 35℃）
            expectTrue(cop25Min > cop30Min && cop30Min > cop35Min,
                       "CRIEPI highspec figure approx: COP order at 0.7kW");
            expectTrue(cop25Rtd > cop30Rtd && cop30Rtd > cop35Rtd,
                       "CRIEPI highspec figure approx: COP order at 2.2kW");
            expectTrue(cop25Max > cop30Max && cop30Max > cop35Max,
                       "CRIEPI highspec figure approx: COP order at 3.3kW");

            // 各温度線で、負荷増加に伴って COP が低下（図の右下がり傾向）
            expectTrue(cop25Min > cop25Rtd && cop25Rtd > cop25Max,
                       "CRIEPI highspec figure approx: COP slope at 25C");
            expectTrue(cop30Min > cop30Rtd && cop30Rtd > cop30Max,
                       "CRIEPI highspec figure approx: COP slope at 30C");
            expectTrue(cop35Min > cop35Rtd && cop35Rtd > cop35Max,
                       "CRIEPI highspec figure approx: COP slope at 35C");
        }
    }

    // -----------------------------
    // CRIEPI highspec: 図ベース近似回帰（冷房 COP-室内温度曲線）
    // Q=2.2kW 一定で、室内温度上昇に伴う COP 上昇傾向を確認。
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json highspec = {
            {"Q", {{"cooling", {{"min", 0.700}, {"rtd", 2.200}, {"max", 3.300}}},
                   {"heating", {{"min", 0.700}, {"rtd", 2.500}, {"max", 5.400}}}}},
            {"P", {{"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.780}}},
                   {"heating", {{"min", 0.095}, {"rtd", 0.390}, {"max", 1.360}}}}},
            {"V_inner", {{"cooling", {{"rtd", 12.1 / 60.0}}}, {"heating", {{"rtd", 13.1 / 60.0}}}}},
            {"V_outer", {{"cooling", {{"rtd", 28.2 / 60.0}}}, {"heating", {{"rtd", 25.5 / 60.0}}}}}
        };

        std::unique_ptr<AirconSpec> model;
        try {
            model = AirconModelFactory::createModel("CRIEPI", highspec);
        } catch (const std::exception& e) {
            fail(std::string("CRIEPI highspec createModel failed (figure Tin sweep): ") + e.what());
            model.reset();
        }

        if (model) {
            const double rhIn = 47.1;
            const double rhEx = 40.5;
            const double qKw = 2.2; // 熱処理量固定
            const double vInner = highspec.at("V_inner").at("cooling").at("rtd").get<double>();
            const double vOuter = highspec.at("V_outer").at("cooling").at("rtd").get<double>();

            const auto estimateCopAtTin = [&](double tIn, double tEx) -> double {
                InputData in{};
                in.T_in = tIn;
                in.T_ex = tEx;
                in.X_in = archenv::absolute_humidity(tIn, rhIn);
                in.X_ex = archenv::absolute_humidity(tEx, rhEx);
                in.Q = qKw * 1000.0; // kW -> W
                in.V_inner = vInner;
                in.V_outer = vOuter;
                COPResult out = model->estimateCOP("cooling", in);
                expectTrue(out.valid, "CRIEPI highspec Tin sweep: estimateCOP valid");
                return out.COP;
            };

            // 室内温度 15/20/25/30℃
            const double c25_t15 = estimateCopAtTin(15.0, 25.0);
            const double c25_t20 = estimateCopAtTin(20.0, 25.0);
            const double c25_t25 = estimateCopAtTin(25.0, 25.0);
            const double c25_t30 = estimateCopAtTin(30.0, 25.0);

            const double c30_t15 = estimateCopAtTin(15.0, 30.0);
            const double c30_t20 = estimateCopAtTin(20.0, 30.0);
            const double c30_t25 = estimateCopAtTin(25.0, 30.0);
            const double c30_t30 = estimateCopAtTin(30.0, 30.0);

            const double c35_t15 = estimateCopAtTin(15.0, 35.0);
            const double c35_t20 = estimateCopAtTin(20.0, 35.0);
            const double c35_t25 = estimateCopAtTin(25.0, 35.0);
            const double c35_t30 = estimateCopAtTin(30.0, 35.0);

            // 図の近似端点チェック（画像読み取り値なので余裕を持たせる）
            const double figTolCop = 1.0;
            expectNear(c25_t15, 5.1, figTolCop, "CRIEPI highspec Tin sweep: COP25C@Tin15");
            expectNear(c25_t30, 10.9, figTolCop, "CRIEPI highspec Tin sweep: COP25C@Tin30");
            expectNear(c30_t15, 4.4, figTolCop, "CRIEPI highspec Tin sweep: COP30C@Tin15");
            expectNear(c30_t30, 8.0, figTolCop, "CRIEPI highspec Tin sweep: COP30C@Tin30");
            expectNear(c35_t15, 3.8, figTolCop, "CRIEPI highspec Tin sweep: COP35C@Tin15");
            expectNear(c35_t30, 6.3, figTolCop, "CRIEPI highspec Tin sweep: COP35C@Tin30");

            // 同一外気温で Tin が上がると COP は単調増加
            expectTrue(c25_t15 < c25_t20 && c25_t20 < c25_t25 && c25_t25 < c25_t30,
                       "CRIEPI highspec Tin sweep: COP increases with Tin at 25C");
            expectTrue(c30_t15 < c30_t20 && c30_t20 < c30_t25 && c30_t25 < c30_t30,
                       "CRIEPI highspec Tin sweep: COP increases with Tin at 30C");
            expectTrue(c35_t15 < c35_t20 && c35_t20 < c35_t25 && c35_t25 < c35_t30,
                       "CRIEPI highspec Tin sweep: COP increases with Tin at 35C");

            // 同一 Tin で 25℃線 > 30℃線 > 35℃線
            expectTrue(c25_t15 > c30_t15 && c30_t15 > c35_t15,
                       "CRIEPI highspec Tin sweep: COP order at Tin15");
            expectTrue(c25_t20 > c30_t20 && c30_t20 > c35_t20,
                       "CRIEPI highspec Tin sweep: COP order at Tin20");
            expectTrue(c25_t25 > c30_t25 && c30_t25 > c35_t25,
                       "CRIEPI highspec Tin sweep: COP order at Tin25");
            expectTrue(c25_t30 > c30_t30 && c30_t30 > c35_t30,
                       "CRIEPI highspec Tin sweep: COP order at Tin30");
        }
    }

    // -----------------------------
    // RAC: 元コード式との整合回帰（冷房 Q_dash_T_C 補正式）
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json racStandard = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.2432}}},
                   {"heating", {{"rtd", 6.68530 / 4.1573}}}}},
            {"dualcompressor", false}
        };

        std::unique_ptr<AirconSpec> model;
        try {
            model = AirconModelFactory::createModel("RAC", racStandard);
        } catch (const std::exception& e) {
            fail(std::string("RAC createModel failed: ") + e.what());
            model.reset();
        }

        if (model) {
            InputData in{};
            in.T_ex = 25.3;
            in.X_ex = 0.0180;        // 18.0 g/kg' -> 0.018 kg/kg'
            in.Q_S = 499.05856;      // W
            in.Q_L = 0.0;            // W
            // RACModelでは冷房/暖房で主に Q_S/Q_L を使うため、他入力はこのテストで未使用
            in.T_in = 27.0;
            in.X_in = 0.010;
            in.Q = 0.0;
            in.V_inner = 0.0;
            in.V_outer = 0.0;

            COPResult out{};
            try {
                out = model->estimateCOP("cooling", in);
            } catch (const std::exception& e) {
                fail(std::string("RAC estimateCOP(cooling) failed: ") + e.what());
                out.valid = false;
            }
            expectTrue(out.valid, "RAC baseline: estimateCOP valid");

            // 元コード式での基準値（kW）
            const double expectedPowerKw = 0.0606481310760;
            const double tolKw = 1.0e-6;
            expectNear(out.power, expectedPowerKw, tolKw, "RAC baseline: power regression");
        }
    }

    // -----------------------------
    // RAC: 冷房の部分負荷比 x>1 領域の回帰
    // （f_C_Theta の上限クリップを掛けない実装と整合）
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json racStandard = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.2432}}},
                   {"heating", {{"rtd", 6.68530 / 4.1573}}}}},
            {"dualcompressor", false}
        };

        std::unique_ptr<AirconSpec> model;
        try {
            model = AirconModelFactory::createModel("RAC", racStandard);
        } catch (const std::exception& e) {
            fail(std::string("RAC createModel failed (x>1 case): ") + e.what());
            model.reset();
        }

        if (model) {
            InputData in{};
            in.T_ex = 31.0;
            in.X_ex = 0.0147; // 14.7 g/kg' -> 0.0147 kg/kg'
            in.Q_S = 10.896264 / 0.0036; // MJ/h -> W
            in.Q_L = 10.505394 / 0.0036; // MJ/h -> W
            in.T_in = 27.0;
            in.X_in = 0.010;
            in.Q = in.Q_S + in.Q_L;
            in.V_inner = 0.0;
            in.V_outer = 0.0;

            COPResult out{};
            try {
                out = model->estimateCOP("cooling", in);
            } catch (const std::exception& e) {
                fail(std::string("RAC estimateCOP(cooling) failed (x>1 case): ") + e.what());
                out.valid = false;
            }
            expectTrue(out.valid, "RAC x>1 case: estimateCOP valid");

            // 比較対象ソフトの同一条件内部値に合わせた基準（kW）
            const double expectedPowerKw = 1.779127167;
            const double tolKw = 2.0e-4;
            expectNear(out.power, expectedPowerKw, tolKw, "RAC x>1 case: power regression");
        }
    }

    // -----------------------------
    // RAC: 参照ソフト出力との代表点比較（抜粋）+ 統計チェック
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json racStandard = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.2432}}},
                   {"heating", {{"rtd", 6.68530 / 4.1573}}}}},
            {"dualcompressor", false}
        };

        struct RefPoint {
            double tExC;
            double xExGkg;
            double lCsMjh;
            double lClMjh;
            double powerW;
        };

        // ユーザー提供の大量データから、運転域を広くカバーする代表点を抜粋。
        const std::vector<RefPoint> refs = {
            {23.8, 10.6, 1.5591197, 0.0, 50.44308058},
            {25.4, 11.0, 5.1483645, 0.0, 137.9340836},
            {21.5, 11.9, 0.057832718, 0.0, 17.25106159},
            {28.2, 15.8, 9.612888, 8.681185, 1053.493181},
            {31.0, 14.7, 10.896264, 10.505394, 1779.127167},
            {31.0, 13.8, 8.497747, 6.964105, 837.2822338},
            {29.4, 13.4, 6.4589367, 1.9476247, 287.1672207},
            {28.4, 14.4, 10.24782, 6.3700576, 858.5126446},
            {26.1, 14.5, 7.202132, 1.5770469, 255.2829304},
            {27.6, 14.8, 9.401802, 5.4555054, 650.9999576},
            {30.1, 12.3, 11.597869, 7.2062006, 1233.758099},
            {30.3, 8.9, 11.470021, 3.7860305, 788.0770358},
            {31.0, 11.3, 11.597416, 8.21071, 1467.766292},
            {32.5, 12.1, 11.022491, 4.4615307, 897.9336348},
            {29.7, 15.0, 10.465861, 10.923465, 1664.667385},
            {30.9, 18.4, 8.11749, 11.886623, 1496.312421},
            {33.8, 17.4, 10.681958, 10.722575, 1837.46215},
            {27.1, 18.2, 8.291088, 9.750651, 959.397343},
            {29.2, 18.5, 8.305472, 11.815409, 1393.461774},
            {25.4, 16.9, 1.7522682, 0.0, 60.00335778},
            {24.3, 14.1, 0.5236707, 0.0, 29.74292727},
            {30.4, 13.1, 11.272991, 10.135212, 1728.753329},
            {31.5, 15.4, 11.86595, 9.530266, 1816.898452},
            {23.0, 12.1, 0.7362714, 0.0, 31.93244542},
        };

        std::unique_ptr<AirconSpec> model;
        try {
            model = AirconModelFactory::createModel("RAC", racStandard);
        } catch (const std::exception& e) {
            fail(std::string("RAC createModel failed (reference subset): ") + e.what());
            model.reset();
        }

        if (model) {
            double sumRelErr = 0.0;
            double maxRelErr = 0.0;
            int count = 0;

            for (size_t i = 0; i < refs.size(); ++i) {
                const auto& rp = refs[i];

                InputData in{};
                in.T_ex = rp.tExC;
                in.X_ex = rp.xExGkg * 1e-3;    // g/kg' -> kg/kg'
                in.Q_S = rp.lCsMjh / 0.0036;   // MJ/h -> W
                in.Q_L = rp.lClMjh / 0.0036;   // MJ/h -> W
                in.T_in = 27.0;
                in.X_in = 0.010;
                in.Q = in.Q_S + in.Q_L;
                in.V_inner = 0.0;
                in.V_outer = 0.0;

                COPResult out{};
                try {
                    out = model->estimateCOP("cooling", in);
                } catch (const std::exception& e) {
                    fail(std::string("RAC reference subset estimateCOP failed at index ") +
                         std::to_string(i) + ": " + e.what());
                    continue;
                }
                expectTrue(out.valid, "RAC reference subset: estimateCOP valid at index " + std::to_string(i));
                if (!out.valid) continue;

                const double actualW = out.power * 1000.0; // kW -> W
                const double expectedW = rp.powerW;
                const double absErr = std::abs(actualW - expectedW);
                const double relErr = absErr / std::max(1.0, expectedW);

                sumRelErr += relErr;
                maxRelErr = std::max(maxRelErr, relErr);
                ++count;

                // 代表点比較は「絶対+相対」の複合許容で評価する。
                // 低負荷域を過度に厳しくせず、高負荷域では相対誤差を制御する。
                const double perPointTolW = std::max(35.0, expectedW * 0.22);
                if (absErr > perPointTolW) {
                    std::ostringstream oss;
                    oss << "RAC reference subset out-of-tolerance at index " << i
                        << " (actualW=" << actualW
                        << ", expectedW=" << expectedW
                        << ", absErr=" << absErr
                        << ", relErr=" << relErr
                        << ", tolW=" << perPointTolW << ")";
                    fail(oss.str());
                }
            }

            expectTrue(count > 0, "RAC reference subset: at least one sample evaluated");
            if (count > 0) {
                const double mape = sumRelErr / static_cast<double>(count);
                expectTrue(mape <= 0.12, "RAC reference subset: MAPE <= 12%");
                expectTrue(maxRelErr <= 0.22, "RAC reference subset: max relative error <= 22%");
            }
        }
    }

    // -----------------------------
    // RAC: 参照ソフト出力（全件）との統計比較
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json racStandard = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.2432}}},
                   {"heating", {{"rtd", 6.68530 / 4.1573}}}}},
            {"dualcompressor", false}
        };

#ifndef RAC_REFERENCE_FULL_TSV_PATH
        fail("RAC full reference: RAC_REFERENCE_FULL_TSV_PATH is not defined");
#else
        std::ifstream ifs(RAC_REFERENCE_FULL_TSV_PATH);
        if (!ifs) {
            fail(std::string("RAC full reference: cannot open file: ") + RAC_REFERENCE_FULL_TSV_PATH);
        } else {
            std::unique_ptr<AirconSpec> model;
            try {
                model = AirconModelFactory::createModel("RAC", racStandard);
            } catch (const std::exception& e) {
                fail(std::string("RAC createModel failed (full reference): ") + e.what());
                model.reset();
            }

            if (model) {
                std::string line;
                // header skip
                if (!std::getline(ifs, line)) {
                    fail("RAC full reference: empty TSV");
                }

                int n = 0;
                double sumRelErr = 0.0;
                double sumSignedRelErr = 0.0;
                std::vector<double> relErrs;
                relErrs.reserve(1500);

                while (std::getline(ifs, line)) {
                    if (line.empty()) continue;
                    std::istringstream iss(line);
                    double tEx = 0.0, xExGkg = 0.0, lCsMjh = 0.0, lClMjh = 0.0, powerRefW = 0.0;
                    if (!(iss >> tEx >> xExGkg >> lCsMjh >> lClMjh >> powerRefW)) {
                        continue;
                    }

                    InputData in{};
                    in.T_ex = tEx;
                    in.X_ex = xExGkg * 1e-3;   // g/kg' -> kg/kg'
                    in.Q_S = lCsMjh / 0.0036;  // MJ/h -> W
                    in.Q_L = lClMjh / 0.0036;  // MJ/h -> W
                    in.T_in = 27.0;
                    in.X_in = 0.010;
                    in.Q = in.Q_S + in.Q_L;
                    in.V_inner = 0.0;
                    in.V_outer = 0.0;

                    COPResult out{};
                    try {
                        out = model->estimateCOP("cooling", in);
                    } catch (const std::exception&) {
                        continue;
                    }
                    if (!out.valid) continue;

                    const double powerModelW = out.power * 1000.0;
                    const double relErr = std::abs(powerModelW - powerRefW) / std::max(1.0, powerRefW);
                    const double signedRelErr = (powerModelW - powerRefW) / std::max(1.0, powerRefW);
                    relErrs.push_back(relErr);
                    sumRelErr += relErr;
                    sumSignedRelErr += signedRelErr;
                    ++n;
                }

                expectTrue(n >= 1000, "RAC full reference: enough samples (>=1000)");
                if (n > 0) {
                    const double mape = sumRelErr / static_cast<double>(n);
                    const double meanBias = sumSignedRelErr / static_cast<double>(n);
                    std::sort(relErrs.begin(), relErrs.end());
                    const int p95Index = static_cast<int>(0.95 * (relErrs.size() - 1));
                    const double p95 = relErrs[p95Index];
                    const double maxRel = relErrs.back();

                    // 全件比較は「監視」目的なので、代表点より緩い統計閾値で判定する。
                    expectTrue(mape <= 0.26, "RAC full reference: MAPE <= 26%");
                    expectTrue(p95 <= 0.45, "RAC full reference: P95 relative error <= 45%");
                    expectTrue(maxRel <= 0.85, "RAC full reference: max relative error <= 85%");
                    expectTrue(std::abs(meanBias) <= 0.20, "RAC full reference: mean bias within +-20%");
                }
            }
        }
#endif
    }

    // -----------------------------
    // RAC: い・ろ・は 3機種の回帰比較
    // - い: 既存基準
    // - ろ/は: ユーザー提示結果（いに対する定格消費電力差）との整合を確認
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json racI = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.2432}}},
                   {"heating", {{"rtd", 6.68530 / 4.157264}}}}},
            {"dualcompressor", false}
        };
        const nlohmann::json racRo = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.0576}}},
                   {"heating", {{"rtd", 6.68530 / 4.014352}}}}},
            {"dualcompressor", false}
        };
        const nlohmann::json racHa = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 2.8512}}},
                   {"heating", {{"rtd", 6.68530 / 3.855424}}}}},
            {"dualcompressor", false}
        };

        std::unique_ptr<AirconSpec> modelI, modelRo, modelHa;
        try {
            modelI = AirconModelFactory::createModel("RAC", racI);
            modelRo = AirconModelFactory::createModel("RAC", racRo);
            modelHa = AirconModelFactory::createModel("RAC", racHa);
        } catch (const std::exception& e) {
            fail(std::string("RAC iroha createModel failed: ") + e.what());
        }

        if (modelI && modelRo && modelHa) {
            struct RefPoint {
                double tExC;
                double xExGkg;
                double lCsMjh;
                double lClMjh;
                double pI_W;
                double pRo_W;
                double pHa_W;
            };

            // ユーザー提示値から代表点を抽出（い/ろ/は）
            const std::vector<RefPoint> points = {
                {23.8, 10.6, 1.5591197, 0.0, 50.44308058, 53.50503628, 57.37829648},
                {28.2, 15.8, 9.612888, 8.681185, 1053.493181, 1117.441486, 1198.333714},
                {31.0, 14.7, 10.896264, 10.505394, 1779.127167, 1887.122328, 2023.732193},
                {31.2, 14.9, 11.186873, 10.214659, 1796.336996, 1905.376814, 2043.308132},
                {25.6, 13.6, 2.930304, 0.0, 87.49059571, 92.80138017, 99.5193252},
                {25.4, 11.0, 5.1483645, 0.0, 137.9340836, 146.3068484862, 156.8980849928},
                {21.5, 11.9, 0.057832718, 0.0, 17.25106159, 18.2982217912, 19.6228405404},
                {31.0, 13.8, 8.497747, 6.964105, 837.2822338, 888.1062731097, 952.3967945637},
                {30.3, 8.9, 11.470021, 3.7860305, 788.0770358, 835.9142603698, 896.4265721474},
                {30.9, 18.4, 8.11749, 11.886623, 1496.312421, 1587.1403858540, 1702.0343868502},
                {23.0, 12.1, 0.7362714, 0.0, 31.93244542, 33.8707832896, 36.3227086792},
                {32.5, 12.1, 11.022491, 4.4615307, 897.9336348, 952.4392871479, 1021.3869123118},
            };

            const auto evalPowerW = [&](AirconSpec* model, const RefPoint& rp) -> double {
                InputData in{};
                in.T_ex = rp.tExC;
                in.X_ex = rp.xExGkg * 1e-3;   // g/kg' -> kg/kg'
                in.Q_S = rp.lCsMjh / 0.0036;  // MJ/h -> W
                in.Q_L = rp.lClMjh / 0.0036;  // MJ/h -> W
                in.T_in = 27.0;
                in.X_in = 0.010;
                in.Q = in.Q_S + in.Q_L;
                in.V_inner = 0.0;
                in.V_outer = 0.0;
                COPResult out = model->estimateCOP("cooling", in);
                expectTrue(out.valid, "RAC iroha representative: estimateCOP valid");
                return out.power * 1000.0; // kW -> W
            };

            for (size_t i = 0; i < points.size(); ++i) {
                const auto& rp = points[i];
                const double pI = evalPowerW(modelI.get(), rp);
                const double pRo = evalPowerW(modelRo.get(), rp);
                const double pHa = evalPowerW(modelHa.get(), rp);

                // 表示桁由来の丸めを吸収するため、厳しめの絶対許容 0.5W
                const double tolW = 0.5;
                expectNear(pI, rp.pI_W, tolW, "RAC iroha representative: i idx=" + std::to_string(i));
                expectNear(pRo, rp.pRo_W, tolW, "RAC iroha representative: ro idx=" + std::to_string(i));
                expectNear(pHa, rp.pHa_W, tolW, "RAC iroha representative: ha idx=" + std::to_string(i));
            }

#ifndef RAC_REFERENCE_FULL_TSV_PATH
            fail("RAC iroha full reference: RAC_REFERENCE_FULL_TSV_PATH is not defined");
#else
            // 既存「い」の全件TSVを基準に、ろ/はの期待電力比で全件回帰チェック
            // （ユーザー提示の全件結果と同じ関係を検証）
            std::ifstream ifs(RAC_REFERENCE_FULL_TSV_PATH);
            if (!ifs) {
                fail(std::string("RAC iroha full reference: cannot open file: ") + RAC_REFERENCE_FULL_TSV_PATH);
            } else {
                const double ratioRo = 3.2432 / 3.0576; // pRo = pI * ratioRo
                const double ratioHa = 3.2432 / 2.8512; // pHa = pI * ratioHa

                std::string line;
                if (!std::getline(ifs, line)) {
                    fail("RAC iroha full reference: empty TSV");
                } else {
                    int n = 0;
                    double maxRelRo = 0.0;
                    double maxRelHa = 0.0;
                    double sumRelRo = 0.0;
                    double sumRelHa = 0.0;

                    while (std::getline(ifs, line)) {
                        if (line.empty()) continue;
                        std::istringstream iss(line);
                        double tEx = 0.0, xExGkg = 0.0, lCsMjh = 0.0, lClMjh = 0.0, pIRefW = 0.0;
                        if (!(iss >> tEx >> xExGkg >> lCsMjh >> lClMjh >> pIRefW)) continue;

                        RefPoint rp{};
                        rp.tExC = tEx;
                        rp.xExGkg = xExGkg;
                        rp.lCsMjh = lCsMjh;
                        rp.lClMjh = lClMjh;

                        const double pRoExpW = pIRefW * ratioRo;
                        const double pHaExpW = pIRefW * ratioHa;
                        const double pRoActW = evalPowerW(modelRo.get(), rp);
                        const double pHaActW = evalPowerW(modelHa.get(), rp);

                        const double relRo = std::abs(pRoActW - pRoExpW) / std::max(1.0, pRoExpW);
                        const double relHa = std::abs(pHaActW - pHaExpW) / std::max(1.0, pHaExpW);
                        maxRelRo = std::max(maxRelRo, relRo);
                        maxRelHa = std::max(maxRelHa, relHa);
                        sumRelRo += relRo;
                        sumRelHa += relHa;
                        ++n;
                    }

                    expectTrue(n >= 1000, "RAC iroha full reference: enough samples (>=1000)");
                    if (n > 0) {
                        const double mapeRo = sumRelRo / static_cast<double>(n);
                        const double mapeHa = sumRelHa / static_cast<double>(n);
                        // 比率整合はほぼ一致のはずだが、浮動小数の安全マージンを持たせる
                        expectTrue(maxRelRo <= 1.0e-6, "RAC iroha full reference: ro max rel <= 1e-6");
                        expectTrue(maxRelHa <= 1.0e-6, "RAC iroha full reference: ha max rel <= 1e-6");
                        expectTrue(mapeRo <= 1.0e-7, "RAC iroha full reference: ro MAPE <= 1e-7");
                        expectTrue(mapeHa <= 1.0e-7, "RAC iroha full reference: ha MAPE <= 1e-7");
                    }
                }
            }
#endif
        }
    }

    // -----------------------------
    // RAC: 暖房 い・ろ・は 3機種の回帰比較（ユーザー提示データ）
    // -----------------------------
    {
        setLogger(nullptr);

        const nlohmann::json racI = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.2432}}},
                   {"heating", {{"rtd", 6.68530 / 4.157264}}}}},
            {"dualcompressor", false}
        };
        const nlohmann::json racRo = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 3.0576}}},
                   {"heating", {{"rtd", 6.68530 / 4.014352}}}}},
            {"dualcompressor", false}
        };
        const nlohmann::json racHa = {
            {"Q", {{"cooling", {{"rtd", 5.60000}, {"max", 5.94462}}},
                   {"heating", {{"rtd", 6.68530}, {"max", 10.04705}}}}},
            {"P", {{"cooling", {{"rtd", 5.60000 / 2.8512}}},
                   {"heating", {{"rtd", 6.68530 / 3.855424}}}}},
            {"dualcompressor", false}
        };

        std::unique_ptr<AirconSpec> modelI, modelRo, modelHa;
        try {
            modelI = AirconModelFactory::createModel("RAC", racI);
            modelRo = AirconModelFactory::createModel("RAC", racRo);
            modelHa = AirconModelFactory::createModel("RAC", racHa);
        } catch (const std::exception& e) {
            fail(std::string("RAC heating iroha createModel failed: ") + e.what());
        }

        if (modelI && modelRo && modelHa) {
            struct HeatRefPoint {
                double tExC;
                double xExGkg;
                double lHsMjh;
                double pI_W;
                double pRo_W;
                double pHa_W;
            };

            // ユーザー提示の暖房表から、運転域を広くカバーする代表点を抽出。
            const std::vector<HeatRefPoint> points = {
                {5.9, 2.9, 14.731312, 1090.98204, 1129.821291, 1176.3947},
                {6.2, 3.1, 5.2096434, 365.4912949, 378.5028823, 394.1054999},
                {5.3, 4.5, 10.052175, 690.3441318, 714.9205667, 744.3909688},
                {4.9, 4.4, 18.4325, 2405.609578, 2491.249919, 2593.944037},
                {9.7, 6.0, 3.345776, 229.1910454, 237.3503077, 247.1343443},
                {8.3, 3.2, 14.915359, 1078.539267, 1116.935552, 1162.977785},
                {4.7, 2.4, 18.735266, 1577.37137, 1633.52621, 1700.863306},
                {3.4, 2.3, 20.219301, 1813.100853, 1877.647726, 1955.047981},
                {1.4, 2.5, 24.453941, 2635.069253, 2728.87842, 2841.368043},
                {4.9, 2.0, 19.70531, 1708.474769, 1769.296926, 1842.230751},
                {2.9, 4.4, 20.858612, 3169.522103, 3282.357934, 3417.663047},
                {4.1, 4.1, 20.853079, 3100.380766, 3210.755146, 3343.108655},
            };

            const auto evalHeatingPowerW = [&](AirconSpec* model, const HeatRefPoint& rp) -> double {
                InputData in{};
                in.T_ex = rp.tExC;
                in.X_ex = rp.xExGkg * 1e-3;   // g/kg' -> kg/kg'
                in.Q_S = rp.lHsMjh / 0.0036;  // MJ/h -> W
                in.Q_L = 0.0;
                in.T_in = 20.0;
                in.X_in = 0.006;
                in.Q = in.Q_S;
                in.V_inner = 0.0;
                in.V_outer = 0.0;
                COPResult out = model->estimateCOP("heating", in);
                expectTrue(out.valid, "RAC heating iroha representative: estimateCOP valid");
                return out.power * 1000.0; // kW -> W
            };

            for (size_t i = 0; i < points.size(); ++i) {
                const auto& rp = points[i];
                const double pI = evalHeatingPowerW(modelI.get(), rp);
                const double pRo = evalHeatingPowerW(modelRo.get(), rp);
                const double pHa = evalHeatingPowerW(modelHa.get(), rp);

                // 表示桁の丸めを吸収しつつ回帰として十分厳しい許容幅。
                const double tolW = 1.0;
                expectNear(pI, rp.pI_W, tolW, "RAC heating iroha representative: i idx=" + std::to_string(i));
                expectNear(pRo, rp.pRo_W, tolW, "RAC heating iroha representative: ro idx=" + std::to_string(i));
                expectNear(pHa, rp.pHa_W, tolW, "RAC heating iroha representative: ha idx=" + std::to_string(i));

                // 3機種は同一負荷条件で概ね い < ろ < は の順になることを確認。
                expectTrue(pI < pRo && pRo < pHa,
                           "RAC heating iroha representative: power order idx=" + std::to_string(i));
            }
        }
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


