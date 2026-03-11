#include "sim_constants_parser.h"
#include <stdexcept>
#include <type_traits>
#include <sstream>
#include "parser_utils.h"
#include "utils/utils.h"

SimulationConstants parseSimulationConstants(const nlohmann::json& config,
                                             std::ostream& logs)
{
    SimulationConstants outConstants;
    // ログ冗長度（任意, 既定 1）
    outConstants.logVerbosity = parser_utils::readVerbosity(config);
    const bool logEnabled = (outConstants.logVerbosity > 0);
    if (logEnabled) {
        writeLog(logs, "シミュレーション定数を解析中...");
    }
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
    auto logLine = [&](const auto& formatter) {
        if (!logEnabled) return;
        std::ostringstream oss;
        formatter(oss);
        writeLog(logs, oss.str());
    };

    logLine([&](std::ostringstream& oss) {
        oss << "  ログ冗長度(verbosity): " << outConstants.logVerbosity;
    });

    logLine([&](std::ostringstream& oss) {
        oss << "  シミュレーション開始時刻: " << outConstants.startTime;
    });
    if (!idx.contains("end") || !idx["end"].is_string()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.end' (string required)");
    }
    outConstants.endTime   = idx["end"];
    logLine([&](std::ostringstream& oss) {
        oss << "  シミュレーション終了時刻: " << outConstants.endTime;
    });
    if (!idx.contains("timestep") || !idx["timestep"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.timestep' (number required)");
    }
    outConstants.timestep  = idx["timestep"];
    logLine([&](std::ostringstream& oss) {
        oss << "  シミュレーション時間ステップ: " << outConstants.timestep;
    });
    if (!idx.contains("length") || !idx["length"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.index.length' (number required)");
    }
    outConstants.length    = idx["length"];
    logLine([&](std::ostringstream& oss) {
        oss << "  シミュレーション長さ: " << outConstants.length;
    });

    // 許容誤差
    if (!sim.contains("tolerance") || !sim["tolerance"].is_object()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance' object");
    }
    const auto& tol = sim["tolerance"];
    if (!tol.contains("ventilation") || !tol["ventilation"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance.ventilation' (number required)");
    }
    outConstants.ventilationTolerance = tol["ventilation"];
    logLine([&](std::ostringstream& oss) {
        oss << "  圧力許容誤差: " << outConstants.ventilationTolerance;
    });
    if (!tol.contains("thermal") || !tol["thermal"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance.thermal' (number required)");
    }
    outConstants.thermalTolerance = tol["thermal"];
    logLine([&](std::ostringstream& oss) {
        oss << "  温度許容誤差: " << outConstants.thermalTolerance;
    });
    if (!tol.contains("convergence") || !tol["convergence"].is_number()) {
        throw std::runtime_error("Missing or invalid 'simulation.tolerance.convergence' (number required)");
    }
    outConstants.convergenceTolerance = tol["convergence"];
    logLine([&](std::ostringstream& oss) {
        oss << "  収束許容誤差: " << outConstants.convergenceTolerance;
    });

    // 連成反復の停止判定（任意）
    if (tol.contains("coupling_pressure")) {
        if (!tol["coupling_pressure"].is_number()) {
            throw std::runtime_error("Missing or invalid 'simulation.tolerance.coupling_pressure' (number required)");
        }
        outConstants.couplingPressureTolerance = tol["coupling_pressure"];
        logLine([&](std::ostringstream& oss) {
            oss << "  連成(圧力)許容誤差: " << outConstants.couplingPressureTolerance;
        });
    }
    if (tol.contains("coupling_temperature")) {
        if (!tol["coupling_temperature"].is_number()) {
            throw std::runtime_error("Missing or invalid 'simulation.tolerance.coupling_temperature' (number required)");
        }
        outConstants.couplingTemperatureTolerance = tol["coupling_temperature"];
        logLine([&](std::ostringstream& oss) {
            oss << "  連成(温度)許容誤差: " << outConstants.couplingTemperatureTolerance;
        });
    }
    if (tol.contains("coupling_humidity")) {
        if (!tol["coupling_humidity"].is_number()) {
            throw std::runtime_error("Missing or invalid 'simulation.tolerance.coupling_humidity' (number required)");
        }
        outConstants.couplingHumidityTolerance = tol["coupling_humidity"];
        logLine([&](std::ostringstream& oss) {
            oss << "  連成(湿気)許容誤差: " << outConstants.couplingHumidityTolerance;
        });
    }
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
    logLine([&](std::ostringstream& oss) {
        oss << "  最大内部反復回数"
            << (customMaxInner ? "（設定値）: " : "（デフォルト値）: ")
            << outConstants.maxInnerIteration;
    });

    // 3ネットワーク連成制御（任意）
    if (sim.contains("coupling")) {
        if (!sim["coupling"].is_object()) {
            throw std::runtime_error("Missing or invalid 'simulation.coupling' object");
        }
        const auto& cp = sim["coupling"];
        if (cp.contains("moisture_enabled")) {
            if (!cp["moisture_enabled"].is_boolean()) {
                throw std::runtime_error("Missing or invalid 'simulation.coupling.moisture_enabled' (boolean required)");
            }
            outConstants.moistureCouplingEnabled = cp["moisture_enabled"];
        }
        if (cp.contains("humidity_relaxation")) {
            if (!cp["humidity_relaxation"].is_number()) {
                throw std::runtime_error("Missing or invalid 'simulation.coupling.humidity_relaxation' (number required)");
            }
            outConstants.humidityRelaxation = cp["humidity_relaxation"];
        }
        if (cp.contains("latent_relaxation")) {
            if (!cp["latent_relaxation"].is_number()) {
                throw std::runtime_error("Missing or invalid 'simulation.coupling.latent_relaxation' (number required)");
            }
            outConstants.latentRelaxation = cp["latent_relaxation"];
        }
        if (cp.contains("humidity_solver_max_iter")) {
            if (!cp["humidity_solver_max_iter"].is_number_integer()) {
                throw std::runtime_error("Missing or invalid 'simulation.coupling.humidity_solver_max_iter' (integer required)");
            }
            logLine([&](std::ostringstream& oss) {
                oss << "  [WARN] simulation.coupling.humidity_solver_max_iter は廃止予定です（直接法のため無視されます）";
            });
        }
        if (cp.contains("humidity_solver_tolerance")) {
            if (!cp["humidity_solver_tolerance"].is_number()) {
                throw std::runtime_error("Missing or invalid 'simulation.coupling.humidity_solver_tolerance' (number required)");
            }
            outConstants.humiditySolverTolerance = cp["humidity_solver_tolerance"];
        }
    }
    if (!(outConstants.humidityRelaxation > 0.0 && outConstants.humidityRelaxation <= 1.0)) {
        throw std::runtime_error("'simulation.coupling.humidity_relaxation' must be in (0, 1]");
    }
    if (!(outConstants.latentRelaxation > 0.0 && outConstants.latentRelaxation <= 1.0)) {
        throw std::runtime_error("'simulation.coupling.latent_relaxation' must be in (0, 1]");
    }
    if (!(outConstants.humiditySolverTolerance > 0.0)) {
        throw std::runtime_error("'simulation.coupling.humidity_solver_tolerance' must be > 0");
    }
    logLine([&](std::ostringstream& oss) {
        oss << "  3ネットワーク連成: " << parser_utils::boolToString(outConstants.moistureCouplingEnabled)
            << ", humidity_relaxation=" << outConstants.humidityRelaxation
            << ", latent_relaxation=" << outConstants.latentRelaxation
            << ", humidity_solver_tolerance=" << outConstants.humiditySolverTolerance;
    });

    // 計算フラグ
    if (!sim.contains("calc_flag") || !sim["calc_flag"].is_object()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag' object");
    }
    const auto& cf = sim["calc_flag"];
    if (!cf.contains("p") || !cf["p"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.p' (boolean required)");
    }
    outConstants.pressureCalc = cf["p"];
    logLine([&](std::ostringstream& oss) {
        oss << "  圧力計算フラグ: " << parser_utils::boolToString(outConstants.pressureCalc);
    });
    if (!cf.contains("t") || !cf["t"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.t' (boolean required)");
    }
    outConstants.temperatureCalc = cf["t"];
    logLine([&](std::ostringstream& oss) {
        oss << "  温度計算フラグ: " << parser_utils::boolToString(outConstants.temperatureCalc);
    });
    if (!cf.contains("x") || !cf["x"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.x' (boolean required)");
    }
    outConstants.humidityCalc = cf["x"];
    logLine([&](std::ostringstream& oss) {
        oss << "  湿度計算フラグ: " << parser_utils::boolToString(outConstants.humidityCalc);
    });
    if (!cf.contains("c") || !cf["c"].is_boolean()) {
        throw std::runtime_error("Missing or invalid 'simulation.calc_flag.c' (boolean required)");
    }
    outConstants.concentrationCalc = cf["c"];
    logLine([&](std::ostringstream& oss) {
        oss << "  濃度計算フラグ: " << parser_utils::boolToString(outConstants.concentrationCalc);
    });

    if (logEnabled) {
        writeLog(logs, "  設定ファイルを解析しました。");
    }


    return outConstants;
}


