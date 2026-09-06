#include "parser/nodes_parser.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <iostream>

using nlohmann::json;

static void expect(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    {
        // numeric scalar
        json cfg;
        cfg["verbosity"] = 0;
        cfg["nodes"] = json::array({
            {{"key","LD"}, {"type","aircon"}, {"mode", 1}},
        });
        std::ostringstream logs;
        auto nodes = parseNodes(cfg, logs, 0);
        expect(nodes.size() == 1, "nodes size should be 1");
        expect(nodes[0].current_mode == "HEATING", "mode 1 should map to HEATING");
    }
    {
        // numeric array
        json cfg;
        cfg["verbosity"] = 0;
        cfg["nodes"] = json::array({
            {{"key","LD"}, {"type","aircon"}, {"mode", json::array({0,1,2,3})}},
        });
        std::ostringstream logs;
        auto nodes = parseNodes(cfg, logs, 2);
        expect(nodes.size() == 1, "nodes size should be 1");
        expect(nodes[0].current_mode == "COOLING", "mode[2] should map to COOLING");
    }
    {
        // mixed array
        json cfg;
        cfg["verbosity"] = 0;
        cfg["nodes"] = json::array({
            {{"key","LD"}, {"type","aircon"}, {"mode", json::array({"OFF", 3})}},
        });
        std::ostringstream logs;
        auto nodes = parseNodes(cfg, logs, 1);
        expect(nodes[0].current_mode == "AUTO", "mixed mode should parse");
    }
    {
        // invalid code
        json cfg;
        cfg["verbosity"] = 0;
        cfg["nodes"] = json::array({
            {{"key","LD"}, {"type","aircon"}, {"mode", 9}},
        });
        std::ostringstream logs;
        bool threw = false;
        try {
            (void)parseNodes(cfg, logs, 0);
        } catch (...) {
            threw = true;
        }
        expect(threw, "invalid mode should throw");
    }
    {
        // aircon calc_x=true は吹出湿度境界のため false へ強制
        json cfg;
        cfg["verbosity"] = 0;
        cfg["nodes"] = json::array({
            {{"key", "ROOM"}, {"type", "normal"}, {"calc_x", true}, {"t", 25.0}},
            {{"key", "AC"}, {"type", "aircon"}, {"set_node", "ROOM"}, {"calc_x", true}, {"mode", 2}},
        });
        std::ostringstream logs;
        auto nodes = parseNodes(cfg, logs, 0);
        expect(nodes.size() == 2, "nodes size should be 2");
        bool foundAc = false;
        for (const auto& n : nodes) {
            if (n.key != "AC") continue;
            foundAc = true;
            expect(!n.calc_x, "aircon calc_x must be forced false");
        }
        expect(foundAc, "AC node missing");
    }

    std::cout << "OK\n";
    return 0;
}


