#include "app/vtsimnx_app.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include <filesystem>
#include <system_error>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <streambuf>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"
#include "vtsimnx_solver_timing.h"
#include "parser/sim_constants_parser.h"
#include "parser/nodes_parser.h"
#include "parser/branches_parser.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "aircon/aircon_controller.h"
#include "simulation_runner.h"
#include "utils/utils.h"
#include "output/artifact_io.h"

namespace {

// 出力JSONの形式バージョン（API側での互換性判断用）
static constexpr int kOutputFormatVersion = 5; // binary artifacts

// verbosity を入力JSONから読む（無ければ 1）
static int readRequestedVerbosity(const json& inputJson) {
    try {
        if (inputJson.contains("simulation") && inputJson["simulation"].is_object()) {
            const auto& sim = inputJson["simulation"];
            if (sim.contains("log") && sim["log"].is_object()) {
                const auto& lg = sim["log"];
                if (lg.contains("verbosity") && lg["verbosity"].is_number_integer()) {
                    return lg["verbosity"].get<int>();
                }
            }
        }
    } catch (...) {}
    return 1;
}

// 書き捨て用 ostream（verbosity=0 のときに使う）
struct NullBuffer final : std::streambuf {
    int overflow(int c) override { return c; }
};
static NullBuffer g_null_buf;
static std::ostream g_null_stream(&g_null_buf);

// ヘルパー: ファイル全体を読み込む（失敗時は false, err に理由）
static bool readFileToString(const char* path, std::string& out, std::string& err) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        err = std::string("エラー: 入力ファイルを開けません: ") + path;
        return false;
    }

    // ファイルサイズを取得
    ifs.seekg(0, std::ios::end);
    std::streampos fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // ファイルサイズが0の場合はエラー
    if (fileSize <= 0) {
        err = std::string("エラー: 入力ファイルが空です: ") + path;
        return false;
    }

    // ファイル全体を読み込む
    out.resize(static_cast<std::size_t>(fileSize));
    ifs.read(&out[0], fileSize);

    // 読み込みが正常に完了したか確認
    if (ifs.gcount() != fileSize) {
        err = std::string("エラー: ファイルの読み込みが不完全です。期待サイズ: ") +
              std::to_string(fileSize) + ", 実際に読み込んだ: " + std::to_string(ifs.gcount());
        return false;
    }

    // ストリームの状態を確認
    if (ifs.fail() && !ifs.eof()) {
        err = std::string("エラー: ファイルの読み込み中にエラーが発生しました: ") + path;
        return false;
    }

    return true;
}

// 入力データ構造体
struct InputData {
    json inputJson;
    SimulationConstants simConstants;
    std::vector<VertexProperties> allNodes;
    std::vector<EdgeProperties> allVentilationBranches;
    std::vector<EdgeProperties> allThermalBranches;
    std::string inputContent;
};

// ファイル入力処理: 入力ファイルを読み込み、JSONをパースしてデータを準備
static bool loadInputData(const char* inputPath, InputData& inputData, std::string& err) {
    // 入力ファイルを読み込む
    if (!readFileToString(inputPath, inputData.inputContent, err)) {
        return false;
    }

    // JSONをパース
    try {
        inputData.inputJson = json::parse(inputData.inputContent);
    } catch (const json::parse_error& e) {
        err = std::string("JSONの解析に失敗しました: ") + e.what();
        return false;
    }

    return true;
}

// シミュレーション実行: ネットワークを構築し、タイムステップループを実行
static bool runSimulationLoop(const InputData& inputData,
                              std::ostream& logs,
                              std::ofstream& ventPressureFile,
                              std::ofstream& ventFlowRateFile,
                              std::ofstream& thermalTemperatureFile,
                              std::ofstream& thermalHeatRateFile,
                              std::ofstream& airconSensibleHeatFile,
                              std::ofstream& airconLatentHeatFile,
                              std::ofstream& airconPowerFile,
                              std::ofstream& airconCOPFile,
                              const std::filesystem::path& schemaPath,
                              std::string& err,
                              TimingList& timings) {
    try {
        ScopedTimer loopTimer(timings, "runSimulationLoop");
        // 設定ファイルを解析（ログも蓄積）
        SimulationConstants simConstants;
        {
            ScopedTimer timer(timings, "parse_simulation_constants");
            simConstants = parseSimulationConstants(inputData.inputJson, logs);
        }

        // 詳細タイミング（各タイムステップ等）はオーバーヘッドになりやすいので、
        // 通常は抑制し、必要なときだけ有効化する。
        // - logVerbosity>=2 のときは有効
        // - それ以外は環境変数 VTSIMNX_TIMINGS が設定されていれば有効
        const bool enableDetailedTimings =
            (simConstants.logVerbosity >= 2) || (std::getenv("VTSIMNX_TIMINGS") != nullptr);
        setTimingsEnabled(enableDetailedTimings);

        // lengthがタイムステップの回数、timestepが1タイムステップの秒数
        writeLog(logs, "タイムステップループ開始: length=" +
                              std::to_string(simConstants.length) +
                              ", timestep=" + std::to_string(simConstants.timestep) + "秒");

        // 最適化: ネットワークをループ外で一度だけ作成
        // 最初のタイムステップで全データを読み込んでトポロジーを構築
        std::vector<VertexProperties> allNodes;
        std::vector<EdgeProperties>   allVentilationBranches;
        std::vector<EdgeProperties>   allThermalBranches;
        VentilationNetwork ventNetwork;
        ThermalNetwork     thermalNetwork;
        AirconController   airconController;

        {
            ScopedLogSection initScope(logs, "初期化: ネットワークトポロジー構築中...");
            writeLog(logs, " トポロジ構築中...");
            auto topoStart = std::chrono::steady_clock::now();

            {
                ScopedTimer timer(timings, "initial_data_parse");
                allNodes               = parseNodes(inputData.inputJson, logs, 0);
                allVentilationBranches = parseVentilationBranches(inputData.inputJson, logs, 0);
                allThermalBranches     = parseThermalBranches(inputData.inputJson, logs, 0);
            }

            {
                ScopedTimer timer(timings, "build_networks");
                ventNetwork.buildFromData(allNodes, allVentilationBranches, simConstants, logs);
                thermalNetwork.buildFromData(allNodes, allThermalBranches, allVentilationBranches, simConstants, logs);
            }

            if (simConstants.temperatureCalc) {
                airconController.initializeModels(thermalNetwork, logs, simConstants.logVerbosity);
            }
            auto topoEnd = std::chrono::steady_clock::now();
            double topoSec = std::chrono::duration_cast<std::chrono::duration<double>>(topoEnd - topoStart).count();
            std::ostringstream oss;
            oss << "初期化完了: ネットワークトポロジー構築済み (所要時間: "
                << std::fixed << std::setprecision(3) << topoSec << "秒)";
            writeLog(logs, oss.str());
        }

        // 前のタイムステップの温度を保存（熱容量ノード用）
        // string map を避け、vertex index ベースで保持する
        std::vector<Vertex> capacityVertices;
        std::vector<Vertex> capacityRefVertices;
        std::vector<double> prevTempByVertex;
        bool havePrevTemps = false;
        if (simConstants.temperatureCalc) {
            auto& thermalGraph = thermalNetwork.getGraph();
            const size_t vCount = static_cast<size_t>(boost::num_vertices(thermalGraph));
            prevTempByVertex.assign(vCount, 0.0);
            capacityVertices.reserve(vCount / 8 + 1);
            capacityRefVertices.reserve(vCount / 8 + 1);
            const auto& keyToVertex = thermalNetwork.getKeyToVertex();
            for (auto v : boost::make_iterator_range(boost::vertices(thermalGraph))) {
                auto& nd = thermalGraph[v];
                if (nd.getTypeCode() == VertexProperties::TypeCode::Capacity && !nd.ref_node.empty()) {
                    auto it = keyToVertex.find(nd.ref_node);
                    if (it != keyToVertex.end()) {
                        capacityVertices.push_back(v);
                        capacityRefVertices.push_back(it->second);
                    }
                }
            }
        }

        constexpr size_t kResultsFlushIntervalLines = 2000;
        size_t resultsLinesWritten = 0;
        ArtifactIO::OutputSchema schema;
        bool schemasWritten = false;

        for (long timestepIndex = 0; timestepIndex < simConstants.length; timestepIndex++) {
            std::string stepMeta;
            if (timingsEnabled()) {
                stepMeta = "timestep=" + std::to_string(timestepIndex + 1);
            }
            ScopedTimer timestepTimer(timings, "timestep", stepMeta);
            setLogTimestepMeta(logs, static_cast<int>(timestepIndex + 1));
            const bool verboseStepLog = (simConstants.logVerbosity >= 2);
            std::unique_ptr<ScopedLogSection> timestepScope;
            if (verboseStepLog) {
                timestepScope = std::make_unique<ScopedLogSection>(
                    logs,
                    "タイムステップ " + std::to_string(timestepIndex + 1) + " を実行中...",
                    true);
            }

            if (timestepIndex > 0) {
                if (verboseStepLog) writeLog(logs, " 時変プロパティ更新中...");
                ScopedTimer timer(timings, "parse_timestep_data", stepMeta);
                const auto topoStart =
                    verboseStepLog ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

                ventNetwork.updatePropertiesForTimestep(allNodes, allVentilationBranches, timestepIndex);
                thermalNetwork.updatePropertiesForTimestep(allNodes, allThermalBranches, allVentilationBranches, timestepIndex);

                if (simConstants.temperatureCalc && havePrevTemps && !capacityVertices.empty()) {
                    auto& thermalGraph = thermalNetwork.getGraph();
                    for (size_t i = 0; i < capacityVertices.size(); ++i) {
                        thermalGraph[capacityVertices[i]].current_t =
                            prevTempByVertex[static_cast<size_t>(capacityRefVertices[i])];
                    }
                    if (verboseStepLog) {
                        writeLog(logs, "熱容量ノード温度を更新しました: " + std::to_string(capacityVertices.size()) + "個");
                    }
                }

                if (verboseStepLog) {
                    auto topoEnd = std::chrono::steady_clock::now();
                    double topoSec = std::chrono::duration_cast<std::chrono::duration<double>>(topoEnd - topoStart).count();
                    std::ostringstream oss;
                    oss << " トポロジ更新完了 (所要時間: " << std::fixed << std::setprecision(3) << topoSec << "秒)";
                    writeLog(logs, oss.str());
                }
            }

            TimestepResult timestepResult;
            if (simConstants.pressureCalc || simConstants.temperatureCalc) {
                ScopedTimer timer(timings, "runSimulation", stepMeta);
                runSimulation(ventNetwork, thermalNetwork, airconController, simConstants, timestepResult, logs, timings, stepMeta);
            }

            if (simConstants.temperatureCalc) {
                auto& thermalGraph = thermalNetwork.getGraph();
                const size_t vCount = static_cast<size_t>(boost::num_vertices(thermalGraph));
                if (prevTempByVertex.size() != vCount) prevTempByVertex.assign(vCount, 0.0);
                for (auto v : boost::make_iterator_range(boost::vertices(thermalGraph))) {
                    prevTempByVertex[static_cast<size_t>(v)] = thermalGraph[v].current_t;
                }
                havePrevTemps = true;
            }

            {
                ScopedTimer timer(timings, "write_timestep_results", stepMeta);
                if (!schema.initialized) {
                    schema.pressureKeys = ventNetwork.getPressureKeys();
                    schema.flowRateKeys = ventNetwork.getFlowRateKeys();
                    schema.temperatureKeys = thermalNetwork.getTemperatureKeys();
                    schema.heatRateKeys = thermalNetwork.getHeatRateKeys();
                    const auto& airconKeys = airconController.getAirconKeys();
                    schema.airconSensibleHeatKeys = airconKeys;
                    schema.airconLatentHeatKeys = airconKeys;
                    schema.airconPowerKeys = airconKeys;
                    schema.airconCOPKeys = airconKeys;
                    schema.initialized = true;
                }

                if (!schemasWritten) {
                    std::string schemaErr;
                    const std::string schemaPathStr = schemaPath.string();
                    if (!ArtifactIO::writeJsonToFile(schemaPathStr.c_str(),
                                                     ArtifactIO::schemaToJson(simConstants.length, simConstants.timestep, schema),
                                                     schemaErr)) {
                        err = schemaErr;
                        return false;
                    }
                    resultsLinesWritten += 8;
                    schemasWritten = true;
                }

                ArtifactIO::writeFloat32ArrayBinary(ventPressureFile, timestepResult.pressure, schema.pressureKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(ventFlowRateFile, timestepResult.flowRate, schema.flowRateKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(thermalTemperatureFile, timestepResult.temperature, schema.temperatureKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(thermalHeatRateFile, timestepResult.heatRate, schema.heatRateKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(airconSensibleHeatFile, timestepResult.airconSensibleHeat, schema.airconSensibleHeatKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(airconLatentHeatFile, timestepResult.airconLatentHeat, schema.airconLatentHeatKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(airconPowerFile, timestepResult.airconPower, schema.airconPowerKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(airconCOPFile, timestepResult.airconCOP, schema.airconCOPKeys.size());

                resultsLinesWritten += 8;
                if (resultsLinesWritten % kResultsFlushIntervalLines == 0) {
                    ventPressureFile.flush();
                    ventFlowRateFile.flush();
                    thermalTemperatureFile.flush();
                    thermalHeatRateFile.flush();
                    airconSensibleHeatFile.flush();
                    airconLatentHeatFile.flush();
                    airconPowerFile.flush();
                    airconCOPFile.flush();
                }
            }
        }

        ventPressureFile.flush();
        ventFlowRateFile.flush();
        thermalTemperatureFile.flush();
        thermalHeatRateFile.flush();
        airconSensibleHeatFile.flush();
        airconLatentHeatFile.flush();
        airconPowerFile.flush();
        airconCOPFile.flush();

        clearLogTimestepMeta(logs);
        writeLog(logs, "タイムステップループ終了");
        return true;
    } catch (const std::exception& e) {
        err = std::string("シミュレーション実行中にエラーが発生しました: ") + e.what();
        writeLog(logs, "[ERROR] " + err);
        return false;
    }
}

// ファイル出力処理: ログと結果を読み込んで出力JSONを構築し、ファイルに書き込む
static bool writeOutputData(const char* outputPath,
                            const std::string& artifactDirName,
                            const std::string& logFileName,
                            const std::map<std::string, std::string>& resultFiles,
                            const std::string& inputContent,
                            const TimingList& timings,
                            std::string& err) {
    json out;
    out["status"] = "ok";
    out["format_version"] = kOutputFormatVersion;
    out["input_length"] = inputContent.size();
    out["artifact_dir"] = artifactDirName;
    out["log_file"] = logFileName;
    out["result_files"] = resultFiles;
    if (!timings.empty()) {
        json timingArray = json::array();
        for (const auto& entry : timings) {
            json t;
            t["name"] = entry.name;
            t["duration_ms"] = entry.durationMs;
            if (!entry.metadata.empty()) {
                t["meta"] = entry.metadata;
            }
            timingArray.push_back(t);
        }
        out["timings"] = timingArray;
    }

    if (!ArtifactIO::writeJsonToFile(outputPath, out, err)) {
        return false;
    }
    return true;
}

// エラー出力処理: エラー情報をログファイルと出力JSONに書き込む
static bool writeErrorOutput(const char* outputPath,
                             const std::string& artifactDirName,
                             const std::string& logFileName,
                             const std::string& inputContent,
                             const std::string& errorMessage,
                             const TimingList& timings,
                             std::string& err) {
    json out = {
        {"status", "error"},
        {"format_version", kOutputFormatVersion},
        {"error", errorMessage},
        {"input_length", inputContent.size()},
        {"artifact_dir", artifactDirName},
        {"log_file", logFileName},
    };
    if (!timings.empty()) {
        json timingArray = json::array();
        for (const auto& entry : timings) {
            json t;
            t["name"] = entry.name;
            t["duration_ms"] = entry.durationMs;
            if (!entry.metadata.empty()) {
                t["meta"] = entry.metadata;
            }
            timingArray.push_back(t);
        }
        out["timings"] = timingArray;
    }

    if (!ArtifactIO::writeJsonToFile(outputPath, out, err)) {
        return false;
    }
    return true;
}

} // namespace

int runVtsimnxSolverApp(const char* inputPath, const char* outputPath) {
    const std::filesystem::path outputJsonPath = std::filesystem::path(outputPath);
    const std::filesystem::path outputParent = outputJsonPath.has_parent_path() ? outputJsonPath.parent_path()
                                                                                : std::filesystem::current_path();
    const std::string outputStem = outputJsonPath.stem().string().empty() ? "output" : outputJsonPath.stem().string();
    const std::string artifactDirName = outputStem + ".artifacts." + std::to_string(ArtifactIO::epochMillis());
    const std::filesystem::path artifactDirPath = outputParent / artifactDirName;

    std::error_code fsErr;
    std::filesystem::create_directories(artifactDirPath, fsErr);
    if (fsErr) {
        std::cerr << "エラー: 出力フォルダを作成できません: " << artifactDirPath.string() << " (" << fsErr.message()
                  << ")\n";
        return 1;
    }

    const std::string logFileName = "solver.log";
    const std::filesystem::path logPath = artifactDirPath / logFileName;

    const std::string schemaFileName = "schema.json";
    const std::filesystem::path schemaPath = artifactDirPath / schemaFileName;

    // 結果ファイル（バイナリ）
    const std::string ventPressureBinName = "vent.pressure.f32.bin";
    const std::string ventFlowRateBinName = "vent.flow_rate.f32.bin";
    const std::string thermalTemperatureBinName = "thermal.temperature.f32.bin";
    const std::string thermalHeatRateBinName = "thermal.heat_rate.f32.bin";
    const std::string airconSensibleHeatBinName = "aircon.sensible_heat.f32.bin";
    const std::string airconLatentHeatBinName = "aircon.latent_heat.f32.bin";
    const std::string airconPowerBinName = "aircon.power.f32.bin";
    const std::string airconCOPBinName = "aircon.cop.f32.bin";

    const std::filesystem::path ventPressureBinPath = artifactDirPath / ventPressureBinName;
    const std::filesystem::path ventFlowRateBinPath = artifactDirPath / ventFlowRateBinName;
    const std::filesystem::path thermalTemperatureBinPath = artifactDirPath / thermalTemperatureBinName;
    const std::filesystem::path thermalHeatRateBinPath = artifactDirPath / thermalHeatRateBinName;
    const std::filesystem::path airconSensibleHeatBinPath = artifactDirPath / airconSensibleHeatBinName;
    const std::filesystem::path airconLatentHeatBinPath = artifactDirPath / airconLatentHeatBinName;
    const std::filesystem::path airconPowerBinPath = artifactDirPath / airconPowerBinName;
    const std::filesystem::path airconCOPBinPath = artifactDirPath / airconCOPBinName;

    std::ofstream logFile(logPath, std::ios::out | std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "エラー: ログファイルを開けません: " << logPath << "\n";
        return 1;
    }

    std::ofstream ventPressureFile(ventPressureBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ventPressureFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << ventPressureBinPath << "\n";
        return 1;
    }
    std::ofstream ventFlowRateFile(ventFlowRateBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ventFlowRateFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << ventFlowRateBinPath << "\n";
        return 1;
    }
    std::ofstream thermalTemperatureFile(thermalTemperatureBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalTemperatureFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalTemperatureBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateFile(thermalHeatRateBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateBinPath << "\n";
        return 1;
    }
    std::ofstream airconSensibleHeatFile(airconSensibleHeatBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconSensibleHeatFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconSensibleHeatBinPath << "\n";
        return 1;
    }
    std::ofstream airconLatentHeatFile(airconLatentHeatBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconLatentHeatFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconLatentHeatBinPath << "\n";
        return 1;
    }
    std::ofstream airconPowerFile(airconPowerBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconPowerFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconPowerBinPath << "\n";
        return 1;
    }
    std::ofstream airconCOPFile(airconCOPBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconCOPFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconCOPBinPath << "\n";
        return 1;
    }

    // 出力ファイルのバッファを大きくしてI/Oオーバーヘッドを低減
    constexpr size_t kFileBufferBytes = 1u << 20; // 1MiB
    std::vector<char> logBuf(kFileBufferBytes);
    std::vector<char> ventPressureBuf(kFileBufferBytes);
    std::vector<char> ventFlowRateBuf(kFileBufferBytes);
    std::vector<char> thermalTemperatureBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateBuf(kFileBufferBytes);
    std::vector<char> airconSensibleHeatBuf(kFileBufferBytes);
    std::vector<char> airconLatentHeatBuf(kFileBufferBytes);
    std::vector<char> airconPowerBuf(kFileBufferBytes);
    std::vector<char> airconCOPBuf(kFileBufferBytes);
    logFile.rdbuf()->pubsetbuf(logBuf.data(), static_cast<std::streamsize>(logBuf.size()));
    ventPressureFile.rdbuf()->pubsetbuf(ventPressureBuf.data(), static_cast<std::streamsize>(ventPressureBuf.size()));
    ventFlowRateFile.rdbuf()->pubsetbuf(ventFlowRateBuf.data(), static_cast<std::streamsize>(ventFlowRateBuf.size()));
    thermalTemperatureFile.rdbuf()->pubsetbuf(thermalTemperatureBuf.data(), static_cast<std::streamsize>(thermalTemperatureBuf.size()));
    thermalHeatRateFile.rdbuf()->pubsetbuf(thermalHeatRateBuf.data(), static_cast<std::streamsize>(thermalHeatRateBuf.size()));
    airconSensibleHeatFile.rdbuf()->pubsetbuf(airconSensibleHeatBuf.data(), static_cast<std::streamsize>(airconSensibleHeatBuf.size()));
    airconLatentHeatFile.rdbuf()->pubsetbuf(airconLatentHeatBuf.data(), static_cast<std::streamsize>(airconLatentHeatBuf.size()));
    airconPowerFile.rdbuf()->pubsetbuf(airconPowerBuf.data(), static_cast<std::streamsize>(airconPowerBuf.size()));
    airconCOPFile.rdbuf()->pubsetbuf(airconCOPBuf.data(), static_cast<std::streamsize>(airconCOPBuf.size()));

    InputData inputData;
    std::string err;
    TimingList timings;

    auto loadStart = std::chrono::steady_clock::now();
    bool loadOk = loadInputData(inputPath, inputData, err);
    auto loadEnd = std::chrono::steady_clock::now();
    double loadMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(loadEnd - loadStart).count();
    timings.push_back({"load_input", loadMs, ""});
    if (!loadOk) {
        std::cerr << err << "\n";
        std::string writeErr;
        writeErrorOutput(outputPath, artifactDirName, logFileName, "", err, timings, writeErr);
        return 1;
    }

    const int requestedVerbosity = readRequestedVerbosity(inputData.inputJson);
    std::ostream& logs = (requestedVerbosity <= 0) ? g_null_stream : static_cast<std::ostream&>(logFile);
    writeLog(logs, "入力JSONを読み込みました。");

    auto simStart = std::chrono::steady_clock::now();
    bool simulationSuccess = runSimulationLoop(
        inputData,
        logs,
        ventPressureFile,
        ventFlowRateFile,
        thermalTemperatureFile,
        thermalHeatRateFile,
        airconSensibleHeatFile,
        airconLatentHeatFile,
        airconPowerFile,
        airconCOPFile,
        schemaPath,
        err,
        timings);
    auto simEnd = std::chrono::steady_clock::now();
    double simMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(simEnd - simStart).count();
    timings.push_back({"simulation_total", simMs, ""});

    airconCOPFile.close();
    airconPowerFile.close();
    airconLatentHeatFile.close();
    airconSensibleHeatFile.close();
    thermalHeatRateFile.close();
    thermalTemperatureFile.close();
    ventFlowRateFile.close();
    ventPressureFile.close();
    logFile.close();

    if (simulationSuccess) {
        std::map<std::string, std::string> resultFiles = {
            {"schema", schemaFileName},
            {"vent_pressure", ventPressureBinName},
            {"vent_flow_rate", ventFlowRateBinName},
            {"thermal_temperature", thermalTemperatureBinName},
            {"thermal_heat_rate", thermalHeatRateBinName},
            {"aircon_sensible_heat", airconSensibleHeatBinName},
            {"aircon_latent_heat", airconLatentHeatBinName},
            {"aircon_power", airconPowerBinName},
            {"aircon_cop", airconCOPBinName},
        };
        if (!writeOutputData(outputPath, artifactDirName, logFileName, resultFiles, inputData.inputContent, timings, err)) {
            std::cerr << err << "\n";
            return 1;
        }
    } else {
        std::string writeErr;
        if (!writeErrorOutput(outputPath, artifactDirName, logFileName, inputData.inputContent, err, timings, writeErr)) {
            std::cerr << writeErr << "\n";
            return 1;
        }
        return 0;
    }

    return 0;
}


