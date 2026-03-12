#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

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

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


