#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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
                 {{"key", "DUP"}, {"type", "simple_opening"}, {"source", "A"}, {"target", "B"}},
                 {{"key", "DUP"}, {"type", "simple_opening"}, {"source", "B"}, {"target", "C"}},
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
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"ventilation_branches",
             json::array({
                 {{"key", "E1"}, {"type", "simple_opening"}, {"source", "A"}, {"target", "B"}, {"enable", true}},
                 {{"key", "E2"}, {"type", "simple_opening"}, {"source", "A"}, {"target", "B"}, {"enable", json::array({true, false})}},
                 {{"key", "E3"}, {"type", "simple_opening"}, {"source", "A"}, {"target", "B"}}, // enable missing => true
             })},
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
    // thermal_branches: source/target 必須
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"thermal_branches",
             json::array({{
                 {"key", "T1"},
                 {"type", "conductance"},
                 // source/target missing
                 {"conductance", 1.0},
                 {"area", 1.0},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseThermalBranches(cfg, logs, 0); },
                     "thermal branch missing source/target should throw");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


