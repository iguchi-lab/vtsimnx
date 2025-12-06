#include "sim_constants_parser.h"
#include <stdexcept>
#include <type_traits>
#include "parser_utils.h"

SimulationConstants parseSimulationConstants(const nlohmann::json& config,
                                             std::ostream& logs)
{
    SimulationConstants outConstants;
    logs << "シミュレーション定数を解析中...\n";
    // ルート検証
    if (!config.contains("simulation") || !config["simulation"].is_object()) {
        throw std::runtime_error("Missing or invalid 'simulation' object");
    }
    const auto& sim = config["simulation"];
    // インデックス情報
    if (!sim.contains("index") || !sim["index"].is_object()) {
        throw std::runtime_error("Missing or invalid 'simulation.index' object");
    }
    const auto& idx = sim["index"];
    if (!idx.contains("start") || !idx["start"].is_string()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.start' (string required)");
    }
    outConstants.startTime = idx["start"];
    logs << "-シミュレーション開始時刻: " << outConstants.startTime << "\n";
    if (!idx.contains("end") || !idx["end"].is_string()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.end' (string required)");
    }
    outConstants.endTime   = idx["end"];
    logs << "-シミュレーション終了時刻: " << outConstants.endTime << "\n";
    if (!idx.contains("timestep") || !idx["timestep"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.timestep' (number required)");
    }
    outConstants.timestep  = idx["timestep"];
    logs << "-シミュレーション時間ステップ: " << outConstants.timestep << "\n";
    if (!idx.contains("length") || !idx["length"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.length' (number required)");
    }
    outConstants.length    = idx["length"];
    logs << "-シミュレーション長さ: " << outConstants.length << "\n";

    // 許容誤差
    if (!sim.contains("tolerance") || !sim["tolerance"].is_object()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance' object");
    }
    const auto& tol = sim["tolerance"];
    if (!tol.contains("ventilation") || !tol["ventilation"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance.ventilation' (number required)");
    }
    outConstants.ventilationTolerance = tol["ventilation"];
    logs << "-圧力許容誤差: " << outConstants.ventilationTolerance << "\n";
    if (!tol.contains("thermal") || !tol["thermal"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance.thermal' (number required)");
    }
    outConstants.thermalTolerance = tol["thermal"];
    logs << "-温度許容誤差: " << outConstants.thermalTolerance << "\n";
    if (!tol.contains("convergence") || !tol["convergence"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance.convergence' (number required)");
    }
    outConstants.convergenceTolerance = tol["convergence"];
    logs << "-収束許容誤差: " << outConstants.convergenceTolerance << "\n";
    bool customMaxInner = false;
    outConstants.maxInnerIteration = 100;
    if (sim.contains("iteration")) {
        if (!sim["iteration"].is_object()) {
            throw std::runtime_error("Missing or invalid 'simulation.iteration' object");
        }
        const auto& iteration = sim["iteration"];
        if (iteration.contains("max_inner")) {
            if (!iteration["max_inner"].is_number()) {
                throw std::runtime_error("Missing or invalid 'simulation.iteration.max_inner' (number required)");
            }
            outConstants.maxInnerIteration = iteration["max_inner"];
            customMaxInner = true;
        }
    } else if (sim.contains("max_inner_iteration")) {
        if (!sim["max_inner_iteration"].is_number()) {
            throw std::runtime_error("Missing or invalid 'simulation.max_inner_iteration' (number required)");
        }
        outConstants.maxInnerIteration = sim["max_inner_iteration"];
        customMaxInner = true;
    }
    logs << "-最大内部反復回数"
         << (customMaxInner ? "（設定値）: " : "（デフォルト値）: ")
         << outConstants.maxInnerIteration << "\n";

    // 計算フラグ
    if (!sim.contains("calc_flag") || !sim["calc_flag"].is_object()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag' object");
    }
    const auto& cf = sim["calc_flag"];
    if (!cf.contains("p") || !cf["p"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.p' (boolean required)");
    }
    outConstants.pressureCalc = cf["p"];
    logs << "-圧力計算フラグ: " << parser_utils::boolToString(outConstants.pressureCalc) << "\n";
    if (!cf.contains("t") || !cf["t"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.t' (boolean required)");
    }
    outConstants.temperatureCalc = cf["t"];
    logs << "-温度計算フラグ: " << parser_utils::boolToString(outConstants.temperatureCalc) << "\n";
    if (!cf.contains("x") || !cf["x"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.x' (boolean required)");
    }
    outConstants.humidityCalc = cf["x"];
    logs << "-湿度計算フラグ: " << parser_utils::boolToString(outConstants.humidityCalc) << "\n";
    if (!cf.contains("c") || !cf["c"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.c' (boolean required)");
    }
    outConstants.concentrationCalc = cf["c"];
    logs << "-濃度計算フラグ: " << parser_utils::boolToString(outConstants.concentrationCalc) << "\n";

    logs << "設定ファイルを解析しました。\n";

    return outConstants;
}


