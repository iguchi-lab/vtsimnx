#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>

#include <nlohmann/json.hpp>

#include "parser/branches_parser.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
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

nlohmann::json opening(const std::string& key,
                       const std::string& src,
                       const std::string& tgt,
                       double alpha = 0.65,
                       double area = 1.0) {
    return nlohmann::json{
        {"key", key},
        {"type", "simple_opening"},
        {"source", src},
        {"target", tgt},
        {"alpha", alpha},
        {"area", area},
    };
}

} // namespace

int main() {
    using nlohmann::json;

    // -----------------------------
    // ventilation_branches: key 必須
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({{
                 {"type", "simple_opening"},
                 {"source", "A"},
                 {"target", "B"},
                 {"alpha", 0.65},
                 {"area", 1.0},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseVentilationBranches(cfg, logs, 0); },
                     "ventilation branch missing key should throw");
    }

    // -----------------------------
    // ventilation_branches: 重複 key はエラー
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({
                 opening("DUP", "A", "B"),
                 opening("DUP", "B", "C"),
             })},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseVentilationBranches(cfg, logs, 0); },
                     "duplicate ventilation key should throw");
    }

    // -----------------------------
    // enable: boolean / array の両対応 + デフォルト true
    // -----------------------------
    {
        auto e1 = opening("E1", "A", "B");
        e1["enable"] = true;
        auto e2 = opening("E2", "A", "B");
        e2["enable"] = json::array({true, false});
        auto e3 = opening("E3", "A", "B");
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches", json::array({e1, e2, e3})},
        };
        std::ostringstream logs;
        const auto b0 = parseVentilationBranches(cfg, logs, 0);
        const auto b1 = parseVentilationBranches(cfg, logs, 1);
        expectTrue(b0.size() == 3, "enable test: size == 3");
        if (b0.size() == 3 && b1.size() == 3) {
            expectTrue(b0[0].current_enabled == true, "enable boolean (t=0)");
            expectTrue(b0[1].current_enabled == true, "enable array (t=0)");
            expectTrue(b1[1].current_enabled == false, "enable array (t=1)");
            expectTrue(b0[2].current_enabled == true, "enable default true");
        }
    }

    // -----------------------------
    // unknown type / missing required params
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({{
                 {"key", "U1"},
                 {"type", "mystery"},
                 {"source", "A"},
                 {"target", "B"},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseVentilationBranches(cfg, logs, 0); },
                     "unknown ventilation type should throw");
    }
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({{
                 {"key", "O1"},
                 {"type", "simple_opening"},
                 {"source", "A"},
                 {"target", "B"},
                 // alpha/area missing
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseVentilationBranches(cfg, logs, 0); },
                     "simple_opening without alpha/area should throw");
    }
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({{
                 {"key", "G1"},
                 {"type", "gap"},
                 {"source", "A"},
                 {"target", "B"},
                 {"a", 0.0},
                 {"n", 1.5},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseVentilationBranches(cfg, logs, 0); },
                     "gap with a<=0 should throw");
    }
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({{
                 {"key", "F1"},
                 {"type", "fixed_flow"},
                 {"source", "A"},
                 {"target", "B"},
                 // vol missing
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseVentilationBranches(cfg, logs, 0); },
                     "fixed_flow without vol should throw");
    }

    // -----------------------------
    // thermal_branches: source/target 必須
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"thermal_branches",
             json::array({{
                 {"key", "T1"},
                 {"type", "conductance"},
                 {"conductance", 1.0},
                 {"area", 1.0},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseThermalBranches(cfg, logs, 0); },
                     "thermal branch missing source/target should throw");
    }

    // -----------------------------
    // ventilation_branches: pressure_loss パラメータ読込（k_total / lambda互換）
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({
                 {{"key", "PL1"}, {"type", "pressure_loss"}, {"source", "A"}, {"target", "B"}, {"area", 0.2}, {"k_total", 8.0}},
                 {{"key", "PL2"}, {"type", "pressure_loss"}, {"source", "B"}, {"target", "C"}, {"area", 0.2}, {"lambda", 0.02}, {"length", 10.0}, {"diameter", 0.2}, {"zeta_total", 1.0}},
             })},
        };
        std::ostringstream logs;
        const auto b = parseVentilationBranches(cfg, logs, 0);
        expectTrue(b.size() == 2, "pressure_loss branches should be parsed");
        if (b.size() == 2) {
            expectNear(b[0].k_total, 8.0, 1e-12, "pressure_loss: k_total parse");
            expectNear(b[1].friction_factor, 0.02, 1e-12, "pressure_loss: lambda alias parse");
            expectNear(b[1].length, 10.0, 1e-12, "pressure_loss: length parse");
            expectNear(b[1].diameter, 0.2, 1e-12, "pressure_loss: diameter parse");
        }
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}
