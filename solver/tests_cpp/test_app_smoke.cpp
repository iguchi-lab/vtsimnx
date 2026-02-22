#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "app/vtsimnx_app.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static std::string readAll(const std::filesystem::path& p) {
    std::ifstream ifs(p, std::ios::in | std::ios::binary);
    if (!ifs) throw std::runtime_error("failed to open: " + p.string());
    std::string s;
    ifs.seekg(0, std::ios::end);
    const std::streampos n = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    s.resize(static_cast<size_t>(n));
    if (n > 0) ifs.read(&s[0], n);
    return s;
}

static size_t countOccurrences(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return 0;
    size_t count = 0;
    size_t pos = 0;
    while (true) {
        pos = haystack.find(needle, pos);
        if (pos == std::string::npos) break;
        ++count;
        pos += needle.size();
    }
    return count;
}

} // namespace

int main() {
    using nlohmann::json;
    namespace fs = std::filesystem;

    const fs::path base = fs::temp_directory_path() / ("vtsimnx_app_smoke_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base, ec);
    if (ec) {
        std::cerr << "[FAIL] failed to create temp dir: " << base.string() << " (" << ec.message() << ")\n";
        return 1;
    }

    const fs::path inputPath = base / "input.json";
    const fs::path outputPath = base / "output.json";

    // 最小入力（calc_flag は false でOK。ノード/ブランチは空配列でOK）
    json cfg = {
        {"simulation",
         {{"log", {{"verbosity", 0}}},
          {"index",
           {{"start", "2020-01-01T00:00:00"},
            {"end", "2020-01-01T00:00:01"},
            {"timestep", 1},
            {"length", 1}}},
          {"tolerance", {{"ventilation", 1e-3}, {"thermal", 1e-3}, {"convergence", 1e-3}}},
          {"calc_flag", {{"p", false}, {"t", false}, {"x", false}, {"c", false}}}}},
        {"nodes", json::array()},
        {"ventilation_branches", json::array()},
        {"thermal_branches", json::array()},
    };

    {
        std::ofstream ofs(inputPath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!ofs) {
            fail("failed to open inputPath");
        } else {
            ofs << cfg.dump(2) << "\n";
        }
    }

    const int rc = runVtsimnxSolverApp(inputPath.c_str(), outputPath.c_str());
    expectTrue(rc == 0, "runVtsimnxSolverApp returns 0");

    expectTrue(fs::exists(outputPath), "output.json exists");
    if (fs::exists(outputPath)) {
        const json out = json::parse(readAll(outputPath));
        expectTrue(out.contains("status") && out["status"].is_string(), "output.status exists");
        expectTrue(out["status"] == "ok", "output.status == ok");
        expectTrue(out.contains("artifact_dir") && out["artifact_dir"].is_string(), "output.artifact_dir exists");
        expectTrue(out.contains("result_files") && out["result_files"].is_object(), "output.result_files exists");
        expectTrue(out["result_files"].contains("schema"), "output.result_files.schema exists");

        if (out.contains("artifact_dir") && out["artifact_dir"].is_string()) {
            const fs::path artifactDir = base / out["artifact_dir"].get<std::string>();
            expectTrue(fs::exists(artifactDir), "artifact_dir exists");
            const fs::path schemaPath = artifactDir / "schema.json";
            expectTrue(fs::exists(schemaPath), "schema.json exists");
        }
    }

    // 各タイムステップの先頭で applyPreset() が走ることを確認する。
    // 期待: 長さ2ステップなら「エアコン設定（初期化）」ログが2回出る。
    const fs::path inputPath2 = base / "input_aircon_each_step.json";
    const fs::path outputPath2 = base / "output_aircon_each_step.json";
    json cfg2 = {
        {"simulation",
         {{"log", {{"verbosity", 1}}},
          {"index",
           {{"start", "2020-01-01T00:00:00"},
            {"end", "2020-01-01T00:00:01"},
            {"timestep", 1},
            {"length", 2}}},
          {"tolerance", {{"ventilation", 1e-3}, {"thermal", 1e-3}, {"convergence", 1e-3}}},
          {"calc_flag", {{"p", false}, {"t", true}, {"x", false}, {"c", false}}}}},
        {"nodes",
         json::array({
             {
                 {"key", "outside"},
                 {"type", "normal"},
                 {"t", 5.0},
             },
             {
                 {"key", "room"},
                 {"type", "normal"},
                 {"calc_t", true},
                 {"t", 22.0},
             },
             {
                 {"key", "ac1"},
                 {"type", "aircon"},
                 {"in_node", "room"},
                 {"set_node", "room"},
                 {"outside_node", "outside"},
                 {"model", "CRIEPI"},
                 {"mode", json::array({"HEATING", "HEATING"})},
                 {"pre_temp", 20.0},
                 {"ac_spec",
                  {
                      {"Q",
                       {{"cooling", {{"min", 0.7}, {"rtd", 2.2}, {"max", 3.3}}},
                        {"heating", {{"min", 0.7}, {"rtd", 2.5}, {"max", 5.4}}}}},
                      {"P",
                       {{"cooling", {{"min", 0.095}, {"rtd", 0.395}, {"max", 0.78}}},
                        {"heating", {{"min", 0.095}, {"rtd", 0.39}, {"max", 1.36}}}}},
                      {"V_inner",
                       {{"cooling", {{"rtd", 0.2016666666667}}},
                        {"heating", {{"rtd", 0.2016666666667}}}}},
                      {"V_outer",
                       {{"cooling", {{"rtd", 0.47}}},
                        {"heating", {{"rtd", 0.47}}}}},
                  }},
             },
         })},
        {"ventilation_branches", json::array()},
        {"thermal_branches",
         json::array({
             {
                 {"key", "outside_to_room"},
                 {"type", "conductance"},
                 {"source", "outside"},
                 {"target", "room"},
                 {"conductance", 1.0},
             },
             {
                 {"key", "room_to_ac1"},
                 {"type", "conductance"},
                 {"source", "room"},
                 {"target", "ac1"},
                 {"conductance", 1.0},
             },
         })},
    };
    {
        std::ofstream ofs(inputPath2, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!ofs) {
            fail("failed to open inputPath2");
        } else {
            ofs << cfg2.dump(2) << "\n";
        }
    }
    const int rc2 = runVtsimnxSolverApp(inputPath2.c_str(), outputPath2.c_str());
    expectTrue(rc2 == 0, "runVtsimnxSolverApp(aircon each step) returns 0");
    expectTrue(fs::exists(outputPath2), "output_aircon_each_step.json exists");
    if (fs::exists(outputPath2)) {
        const json out2 = json::parse(readAll(outputPath2));
        expectTrue(out2.contains("status") && out2["status"] == "ok", "output2.status == ok");
        if (out2.contains("artifact_dir") && out2["artifact_dir"].is_string()) {
            const fs::path artifactDir2 = base / out2["artifact_dir"].get<std::string>();
            const fs::path logPath2 = artifactDir2 / "solver.log";
            expectTrue(fs::exists(logPath2), "solver.log exists for output2");
            if (fs::exists(logPath2)) {
                const std::string logText = readAll(logPath2);
                const size_t modelInitCount = countOccurrences(logText, "エアコンモデル初期化完了");
                const size_t initCount = countOccurrences(logText, "エアコン設定（初期化）");
                expectTrue(modelInitCount == 1, "aircon model initializes once");
                expectTrue(initCount == 2, "applyPreset log appears once per timestep (2 times)");
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


