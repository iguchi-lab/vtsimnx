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

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


