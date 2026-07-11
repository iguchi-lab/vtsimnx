#include <iostream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "parser/nodes_parser.h"

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

} // namespace

int main() {
    using nlohmann::json;

    // key 必須
    {
        json cfg;
        cfg["simulation"] = {{"log", {{"verbosity", 0}}}};
        cfg["nodes"] = json::array({
            {{"type", "normal"}, {"t", 20.0}},
        });
        std::ostringstream logs;
        expectThrows([&]() { (void)parseNodes(cfg, logs, 0); }, "nodes missing key should throw");
    }

    // type 必須
    {
        json cfg;
        cfg["simulation"] = {{"log", {{"verbosity", 0}}}};
        cfg["nodes"] = json::array({
            {{"key", "A"}, {"t", 20.0}},
        });
        std::ostringstream logs;
        expectThrows([&]() { (void)parseNodes(cfg, logs, 0); }, "nodes missing type should throw");
    }

    // ノードキー重複
    {
        json cfg;
        cfg["simulation"] = {{"log", {{"verbosity", 0}}}};
        cfg["nodes"] = json::array({
            {{"key", "A"}, {"type", "normal"}, {"t", 20.0}},
            {{"key", "A"}, {"type", "normal"}, {"t", 21.0}},
        });
        std::ostringstream logs;
        expectThrows([&]() { (void)parseNodes(cfg, logs, 0); }, "duplicate node key should throw");
    }

    // 空キー
    {
        json cfg;
        cfg["simulation"] = {{"log", {{"verbosity", 0}}}};
        cfg["nodes"] = json::array({
            {{"key", ""}, {"type", "normal"}, {"t", 20.0}},
        });
        std::ostringstream logs;
        expectThrows([&]() { (void)parseNodes(cfg, logs, 0); }, "empty node key should throw");
    }

    // 参照ノード欠落
    {
        json cfg;
        cfg["simulation"] = {{"log", {{"verbosity", 0}}}};
        cfg["nodes"] = json::array({
            {{"key", "AC"}, {"type", "aircon"}, {"set_node", "MISSING"}, {"t", 20.0}},
        });
        std::ostringstream logs;
        expectThrows([&]() { (void)parseNodes(cfg, logs, 0); }, "missing set_node should throw");
    }

    // 負の濃度初期値
    {
        json cfg;
        cfg["simulation"] = {{"log", {{"verbosity", 0}}}};
        cfg["nodes"] = json::array({
            {{"key", "A"}, {"type", "normal"}, {"c", -1.0}},
        });
        std::ostringstream logs;
        expectThrows([&]() { (void)parseNodes(cfg, logs, 0); }, "negative c should throw");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


