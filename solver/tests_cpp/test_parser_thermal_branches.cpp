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

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
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
    using nlohmann::json;

    // -----------------------------
    // response_conduction: area 必須
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"thermal_branches",
             json::array({{
                 {"key", "A->B"},
                 {"type", "response_conduction"},
                 // area missing
                 {"resp_a_src", json::array({5.0})},
                 {"resp_b_src", json::array({-5.0})},
                 {"resp_a_tgt", json::array({5.0})},
                 {"resp_b_tgt", json::array({-5.0})},
                 {"resp_c_src", json::array()},
                 {"resp_c_tgt", json::array()},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseThermalBranches(cfg, logs, 0); },
                     "response_conduction requires area");
    }

    // -----------------------------
    // response_conduction: a/b 長さ不一致はエラー
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"thermal_branches",
             json::array({{
                 {"key", "A->B"},
                 {"type", "response_conduction"},
                 {"area", 10.0},
                 {"resp_a_src", json::array({5.0, 1.0})},
                 {"resp_b_src", json::array({-5.0})}, // mismatch
                 {"resp_a_tgt", json::array({5.0})},
                 {"resp_b_tgt", json::array({-5.0})},
                 {"resp_c_src", json::array()},
                 {"resp_c_tgt", json::array()},
             }})},
        };
        std::ostringstream logs;
        expectThrows([&]() { (void)parseThermalBranches(cfg, logs, 0); },
                     "response_conduction rejects mismatched coefficient length");
    }

    // -----------------------------
    // response_conduction: 正常パース
    // -----------------------------
    {
        json cfg = {
            {"simulation", {{"log", {{"verbosity", 0}}}}},
            {"thermal_branches",
             json::array({{
                 {"key", "A->B"},
                 {"type", "response_conduction"},
                 {"area", 10.0},
                 {"resp_a_src", json::array({5.0})},
                 {"resp_b_src", json::array({-5.0})},
                 {"resp_a_tgt", json::array({5.0})},
                 {"resp_b_tgt", json::array({-5.0})},
                 {"resp_c_src", json::array()},
                 {"resp_c_tgt", json::array()},
             }})},
        };
        std::ostringstream logs;
        const auto branches = parseThermalBranches(cfg, logs, 0);
        expectTrue(branches.size() == 1, "response_conduction parses into one branch");
        if (branches.size() == 1) {
            const auto& b = branches[0];
            expectTrue(b.type == "response_conduction", "branch.type");
            expectTrue(b.key == "A->B", "branch.key");
            expectTrue(b.area == 10.0, "branch.area");
            expectTrue(b.resp_a_src.size() == 1 && b.resp_a_src[0] == 5.0, "resp_a_src[0]");
            expectTrue(b.resp_b_src.size() == 1 && b.resp_b_src[0] == -5.0, "resp_b_src[0]");
            expectTrue(b.resp_a_tgt.size() == 1 && b.resp_a_tgt[0] == 5.0, "resp_a_tgt[0]");
            expectTrue(b.resp_b_tgt.size() == 1 && b.resp_b_tgt[0] == -5.0, "resp_b_tgt[0]");
        }
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


