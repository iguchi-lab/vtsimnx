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
    // 用途別の正本。個別キーが無ければ thermal を共通初期値として埋める（後方互換）。
    outConstants.airconTemperatureToleranceK = outConstants.thermalTolerance;
    outConstants.thermalBalanceToleranceW = outConstants.thermalTolerance;
    outConstants.thermalLinearResidualRelativeTolerance = outConstants.thermalTolerance;
    logLine([&](std::ostringstream& oss) {
        oss << "  温度許容誤差(thermal/互換・空調[K]): " << outConstants.thermalTolerance;
    });
    if (tol.contains("aircon_temperature")) {
        if (!tol["aircon_temperature"].is_number()) {
            throw std::runtime_error(
                "Missing or invalid 'simulation.tolerance.aircon_temperature' (number required)");
        }
        outConstants.airconTemperatureToleranceK = tol["aircon_temperature"];
        logLine([&](std::ostringstream& oss) {
            oss << "  空調温度許容誤差[K]: " << outConstants.airconTemperatureToleranceK;
        });
    }
    if (tol.contains("thermal_balance")) {
        if (!tol["thermal_balance"].is_number()) {
            throw std::runtime_error(
                "Missing or invalid 'simulation.tolerance.thermal_balance' (number required)");
        }
        outConstants.thermalBalanceToleranceW = tol["thermal_balance"];
        logLine([&](std::ostringstream& oss) {
            oss << "  熱収支許容誤差[W]: " << outConstants.thermalBalanceToleranceW;
        });
    }
    if (tol.contains("thermal_linear_residual")) {
        if (!tol["thermal_linear_residual"].is_number()) {
            throw std::runtime_error(
                "Missing or invalid 'simulation.tolerance.thermal_linear_residual' (number required)");
        }
        outConstants.thermalLinearResidualRelativeTolerance = tol["thermal_linear_residual"];
        logLine([&](std::ostringstream& oss) {
            oss << "  熱線形残差相対許容: " << outConstants.thermalLinearResidualRelativeTolerance;
        });
    }
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
    outConstants.maxInnerIterations = 100;
    auto readPositiveSizeT = [](const nlohmann::json& value, const char* path) -> std::size_t {
        if (!value.is_number_integer() || value.get<int>() <= 0) {
            throw std::runtime_error(std::string("Missing or invalid '") + path +
                                     "' (positive integer required)");
        }
        return static_cast<std::size_t>(value.get<int>());
    };
    if (sim.contains("iteration")) {
        if (!sim["iteration"].is_object()) {
            throw std::runtime_error("Missing or invalid 'simulation.iteration' object");
        }
        const auto& iteration = sim["iteration"];
        if (iteration.contains("max_inner")) {
            outConstants.maxInnerIterations =
                readPositiveSizeT(iteration["max_inner"], "simulation.iteration.max_inner");
            customMaxInner = true;
        }
    } else if (sim.contains("max_inner_iteration")) {
        outConstants.maxInnerIterations =
            readPositiveSizeT(sim["max_inner_iteration"], "simulation.max_inner_iteration");
        customMaxInner = true;
    }
    logLine([&](std::ostringstream& oss) {
        oss << "  最大内部反復回数"
            << (customMaxInner ? "（設定値）: " : "（デフォルト値）: ")
            << outConstants.maxInnerIterations;
    });

    // 意味分離: 既定は maxInnerIterations と同じ。任意キーで上書き可。
    outConstants.maxCouplingIterations = outConstants.maxInnerIterations;
    outConstants.maxAirconControlIterations = outConstants.maxInnerIterations;
    if (sim.contains("iteration") && sim["iteration"].is_object()) {
        const auto& iteration = sim["iteration"];
        if (iteration.contains("max_coupling")) {
            outConstants.maxCouplingIterations =
                readPositiveSizeT(iteration["max_coupling"], "simulation.iteration.max_coupling");
        }
        if (iteration.contains("max_aircon_control")) {
            outConstants.maxAirconControlIterations = readPositiveSizeT(
                iteration["max_aircon_control"], "simulation.iteration.max_aircon_control");
        }
    }
    if (sim.contains("max_coupling_iteration")) {
        outConstants.maxCouplingIterations =
            readPositiveSizeT(sim["max_coupling_iteration"], "simulation.max_coupling_iteration");
    }
    if (sim.contains("max_aircon_control_iteration")) {
        outConstants.maxAirconControlIterations = readPositiveSizeT(
            sim["max_aircon_control_iteration"], "simulation.max_aircon_control_iteration");
    }
    logLine([&](std::ostringstream& oss) {
        oss << "  最大連成反復回数: " << outConstants.maxCouplingIterations
            << ", 最大空調制御反復回数: " << outConstants.maxAirconControlIterations;
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
        if (cp.contains("latent_coupling_mode")) {
            if (cp["latent_coupling_mode"].is_string()) {
                const std::string mode = cp["latent_coupling_mode"].get<std::string>();
                if (mode == "disabled" || mode == "Disabled" || mode == "DISABLED") {
                    outConstants.latentCouplingMode = 0;
                } else if (mode == "from_humidity_change" || mode == "FromHumidityChange" ||
                           mode == "feedback_to_thermal" || mode == "FeedbackToThermal") {
                    // 実験用・非推奨（換気由来の Δx を相変化と誤認し得る）。将来削除予定。
                    outConstants.latentCouplingMode = 1;
                    logLine([&](std::ostringstream& oss) {
                        oss << "  [WARN] simulation.coupling.latent_coupling_mode="
                               "from_humidity_change は非推奨です。"
                               "材料相変化には from_phase_change、"
                               "換気潜熱には moist_enthalpy_enabled を使ってください。";
                    });
                } else if (mode == "from_phase_change" || mode == "FromPhaseChange") {
                    outConstants.latentCouplingMode = 2;
                } else {
                    throw std::runtime_error(
                        "Invalid 'simulation.coupling.latent_coupling_mode' "
                        "(disabled|from_phase_change|from_humidity_change)");
                }
            } else if (cp["latent_coupling_mode"].is_number_integer()) {
                outConstants.latentCouplingMode = cp["latent_coupling_mode"].get<int>();
                if (outConstants.latentCouplingMode < 0 || outConstants.latentCouplingMode > 2) {
                    throw std::runtime_error(
                        "Invalid 'simulation.coupling.latent_coupling_mode' (0|1|2)");
                }
                if (outConstants.latentCouplingMode == 1) {
                    logLine([&](std::ostringstream& oss) {
                        oss << "  [WARN] simulation.coupling.latent_coupling_mode=1 "
                               "(from_humidity_change) は非推奨です。"
                               "材料相変化には from_phase_change、"
                               "換気潜熱には moist_enthalpy_enabled を使ってください。";
                    });
                }
            } else {
                throw std::runtime_error(
                    "Invalid 'simulation.coupling.latent_coupling_mode' (string|int)");
            }
        }
        if (cp.contains("latent_absolute_tolerance_w")) {
            if (!cp["latent_absolute_tolerance_w"].is_number()) {
                throw std::runtime_error(
                    "Invalid 'simulation.coupling.latent_absolute_tolerance_w' (number required)");
            }
            outConstants.couplingLatentAbsoluteToleranceW = cp["latent_absolute_tolerance_w"];
        }
        if (cp.contains("latent_relative_tolerance")) {
            if (!cp["latent_relative_tolerance"].is_number()) {
                throw std::runtime_error(
                    "Invalid 'simulation.coupling.latent_relative_tolerance' (number required)");
            }
            outConstants.couplingLatentRelativeTolerance = cp["latent_relative_tolerance"];
        }
        if (cp.contains("moist_enthalpy_enabled")) {
            if (!cp["moist_enthalpy_enabled"].is_boolean()) {
                throw std::runtime_error(
                    "Missing or invalid 'simulation.coupling.moist_enthalpy_enabled' (boolean required)");
            }
            outConstants.moistEnthalpyEnabled = cp["moist_enthalpy_enabled"];
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

    // 潜熱フィードバックは moisture 連成＋温度計算が必須（非連成だと熱へ未反映になる）
    if (outConstants.latentCouplingMode != 0) {
        if (!outConstants.moistureCouplingEnabled) {
            throw std::runtime_error(
                "Invalid combination: simulation.coupling.latent_coupling_mode requires "
                "moisture_enabled=true (decoupled humidity cannot apply latent to the same "
                "timestep thermal solve)");
        }
        if (outConstants.humidityCalc && !outConstants.temperatureCalc) {
            throw std::runtime_error(
                "Invalid combination: simulation.coupling.latent_coupling_mode requires "
                "calc_flag.t=true when humidityCalc is enabled");
        }
    }

    if (outConstants.moistEnthalpyEnabled) {
        if (!outConstants.humidityCalc) {
            throw std::runtime_error(
                "Invalid combination: simulation.coupling.moist_enthalpy_enabled requires "
                "calc_flag.x=true");
        }
        if (outConstants.latentCouplingMode == 1) {
            throw std::runtime_error(
                "Invalid combination: simulation.coupling.moist_enthalpy_enabled cannot be used "
                "with latent_coupling_mode=from_humidity_change (double-counting risk)");
        }
    }

    if (logEnabled) {
        writeLog(logs, "  設定ファイルを解析しました。");
    }


    return outConstants;
}


