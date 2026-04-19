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
#include "parser/sim_constants_parser.h"
#include "parser/nodes_parser.h"
#include "parser/branches_parser.h"
#include "network/humidity_network.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "network/contaminant_network.h"
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

struct OutputFiles {
    std::ofstream& ventPressureFile;
    std::ofstream& ventFlowRateFile;
    std::ofstream& thermalTemperatureFile;
    std::ofstream& thermalTemperatureCapacityFile;
    std::ofstream& thermalTemperatureLayerFile;
    std::ofstream& humidityXFile;
    std::ofstream& humidityFluxFile;
    std::ofstream& concentrationCFile;
    std::ofstream& concentrationFluxFile;
    std::ofstream& thermalHeatRateAdvectionFile;
    std::ofstream& thermalHeatRateHeatGenerationFile;
    std::ofstream& thermalHeatRateSolarGainFile;
    std::ofstream& thermalHeatRateNocturnalLossFile;
    std::ofstream& thermalHeatRateConvectionFile;
    std::ofstream& thermalHeatRateConductionFile;
    std::ofstream& thermalHeatRateRadiationFile;
    std::ofstream& thermalHeatRateCapacityFile;
    std::ofstream& airconSensibleHeatFile;
    std::ofstream& airconLatentHeatFile;
    std::ofstream& airconPowerFile;
    std::ofstream& airconCOPFile;
};

static void initializeSchemaIfNeeded(ArtifactIO::OutputSchema& schema,
                                     VentilationNetwork& ventNetwork,
                                     ThermalNetwork& thermalNetwork,
                                     HumidityNetwork& humidityNetwork,
                                     ContaminantNetwork& contaminantNetwork,
                                     AirconController& airconController) {
    if (schema.initialized) return;
    schema.pressureKeys = ventNetwork.getPressureKeys();
    schema.flowRateKeys = ventNetwork.getFlowRateKeys();
    schema.temperatureKeys = thermalNetwork.getTemperatureKeys();
    schema.temperatureKeysCapacity = thermalNetwork.getTemperatureKeysCapacity();
    schema.temperatureKeysLayer = thermalNetwork.getTemperatureKeysLayer();
    schema.humidityKeys = humidityNetwork.getOutputKeys(static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView());
    schema.humidityFluxKeys = schema.flowRateKeys;
    schema.concentrationKeys = contaminantNetwork.getOutputKeys(static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView());
    schema.concentrationFluxKeys = schema.flowRateKeys;
    schema.heatRateKeysAdvection = thermalNetwork.getHeatRateKeysAdvection();
    schema.heatRateKeysHeatGeneration = thermalNetwork.getHeatRateKeysHeatGeneration();
    schema.heatRateKeysSolarGain = thermalNetwork.getHeatRateKeysSolarGain();
    schema.heatRateKeysNocturnalLoss = thermalNetwork.getHeatRateKeysNocturnalLoss();
    schema.heatRateKeysConvection = thermalNetwork.getHeatRateKeysConvection();
    schema.heatRateKeysConduction = thermalNetwork.getHeatRateKeysConduction();
    schema.heatRateKeysRadiation = thermalNetwork.getHeatRateKeysRadiation();
    schema.heatRateKeysCapacity = thermalNetwork.getHeatRateKeysCapacity();

    const auto& airconKeys = airconController.getAirconKeys();
    schema.airconSensibleHeatKeys = airconKeys;
    schema.airconLatentHeatKeys = airconKeys;
    schema.airconPowerKeys = airconKeys;
    schema.airconCOPKeys = airconKeys;
    schema.initialized = true;
}

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
                              OutputFiles& outFiles,
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
        HumidityNetwork    humidityNetwork;
        ContaminantNetwork contaminantNetwork;
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
                humidityNetwork.invalidateCaches();
                contaminantNetwork.invalidateCaches();
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

            // 各タイムステップの先頭では、まずエアコンをOFFで開始する。
            // これにより「非運転時に設定温度を超過していれば暖房しない /
            // 非運転時に設定温度未満なら暖房する」という判定を毎ステップで行える。
            airconController.applyPreset(thermalNetwork, logs);

            {
                // 各ステップの先頭で、当該インデックスの時変プロパティを反映する
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
            if (simConstants.pressureCalc || simConstants.temperatureCalc || simConstants.humidityCalc || simConstants.concentrationCalc) {
                ScopedTimer timer(timings, "runSimulation", stepMeta);
                runSimulation(ventNetwork, thermalNetwork, humidityNetwork, contaminantNetwork, airconController, simConstants, timestepResult, logs, timings, stepMeta);
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
                initializeSchemaIfNeeded(schema, ventNetwork, thermalNetwork, humidityNetwork, contaminantNetwork, airconController);

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

                ArtifactIO::writeFloat32ArrayBinary(outFiles.ventPressureFile, timestepResult.pressure, schema.pressureKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.ventFlowRateFile, timestepResult.flowRate, schema.flowRateKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalTemperatureFile, timestepResult.temperature, schema.temperatureKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalTemperatureCapacityFile, timestepResult.temperatureCapacity, schema.temperatureKeysCapacity.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalTemperatureLayerFile, timestepResult.temperatureLayer, schema.temperatureKeysLayer.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.humidityXFile, timestepResult.humidityX, schema.humidityKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.humidityFluxFile, timestepResult.humidityFlux, schema.humidityFluxKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.concentrationCFile, timestepResult.concentrationC, schema.concentrationKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.concentrationFluxFile, timestepResult.concentrationFlux, schema.concentrationFluxKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateAdvectionFile, timestepResult.heatRateAdvection, schema.heatRateKeysAdvection.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateHeatGenerationFile, timestepResult.heatRateHeatGeneration, schema.heatRateKeysHeatGeneration.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateSolarGainFile, timestepResult.heatRateSolarGain, schema.heatRateKeysSolarGain.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateNocturnalLossFile, timestepResult.heatRateNocturnalLoss, schema.heatRateKeysNocturnalLoss.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateConvectionFile, timestepResult.heatRateConvection, schema.heatRateKeysConvection.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateConductionFile, timestepResult.heatRateConduction, schema.heatRateKeysConduction.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateRadiationFile, timestepResult.heatRateRadiation, schema.heatRateKeysRadiation.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.thermalHeatRateCapacityFile, timestepResult.heatRateCapacity, schema.heatRateKeysCapacity.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.airconSensibleHeatFile, timestepResult.airconSensibleHeat, schema.airconSensibleHeatKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.airconLatentHeatFile, timestepResult.airconLatentHeat, schema.airconLatentHeatKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.airconPowerFile, timestepResult.airconPower, schema.airconPowerKeys.size());
                ArtifactIO::writeFloat32ArrayBinary(outFiles.airconCOPFile, timestepResult.airconCOP, schema.airconCOPKeys.size());

                resultsLinesWritten += 8;
                if (resultsLinesWritten % kResultsFlushIntervalLines == 0) {
                    outFiles.ventPressureFile.flush();
                    outFiles.ventFlowRateFile.flush();
                    outFiles.thermalTemperatureFile.flush();
                    outFiles.thermalTemperatureCapacityFile.flush();
                    outFiles.thermalTemperatureLayerFile.flush();
                    outFiles.humidityXFile.flush();
                    outFiles.humidityFluxFile.flush();
                    outFiles.concentrationCFile.flush();
                    outFiles.concentrationFluxFile.flush();
                    outFiles.thermalHeatRateAdvectionFile.flush();
                    outFiles.thermalHeatRateHeatGenerationFile.flush();
                    outFiles.thermalHeatRateSolarGainFile.flush();
                    outFiles.thermalHeatRateNocturnalLossFile.flush();
                    outFiles.thermalHeatRateConvectionFile.flush();
                    outFiles.thermalHeatRateConductionFile.flush();
                    outFiles.thermalHeatRateRadiationFile.flush();
                    outFiles.thermalHeatRateCapacityFile.flush();
                    outFiles.airconSensibleHeatFile.flush();
                    outFiles.airconLatentHeatFile.flush();
                    outFiles.airconPowerFile.flush();
                    outFiles.airconCOPFile.flush();
                }
            }
        }

        outFiles.ventPressureFile.flush();
        outFiles.ventFlowRateFile.flush();
        outFiles.thermalTemperatureFile.flush();
        outFiles.thermalTemperatureCapacityFile.flush();
        outFiles.thermalTemperatureLayerFile.flush();
        outFiles.humidityXFile.flush();
        outFiles.humidityFluxFile.flush();
        outFiles.concentrationCFile.flush();
        outFiles.concentrationFluxFile.flush();
        outFiles.thermalHeatRateAdvectionFile.flush();
        outFiles.thermalHeatRateHeatGenerationFile.flush();
        outFiles.thermalHeatRateSolarGainFile.flush();
        outFiles.thermalHeatRateNocturnalLossFile.flush();
        outFiles.thermalHeatRateConvectionFile.flush();
        outFiles.thermalHeatRateConductionFile.flush();
        outFiles.thermalHeatRateRadiationFile.flush();
        outFiles.thermalHeatRateCapacityFile.flush();
        outFiles.airconSensibleHeatFile.flush();
        outFiles.airconLatentHeatFile.flush();
        outFiles.airconPowerFile.flush();
        outFiles.airconCOPFile.flush();

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
                            const json& inputJson,
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
    // index 情報（クライアント側で時間インデックスを再構成できるようにする）
    // 入力JSONは parseSimulationConstants で検証済みだが、念のため型チェックしてから入れる
    try {
        if (inputJson.contains("simulation") && inputJson["simulation"].is_object()) {
            const auto& sim = inputJson["simulation"];
            if (sim.contains("index") && sim["index"].is_object()) {
                const auto& idx = sim["index"];
                if (idx.contains("start") && idx["start"].is_string() &&
                    idx.contains("end") && idx["end"].is_string() &&
                    idx.contains("timestep") && idx["timestep"].is_number() &&
                    idx.contains("length") && idx["length"].is_number()) {
                    json outIdx;
                    outIdx["start"] = idx["start"];
                    outIdx["end"] = idx["end"];
                    outIdx["timestep"] = idx["timestep"];
                    outIdx["length"] = idx["length"];
                    out["index"] = outIdx;
                }
            }
        }
    } catch (...) {
        // index の追記失敗は致命ではないので無視
    }
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
                             const json& inputJson,
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
    // index 情報（可能なら出す）
    try {
        if (inputJson.contains("simulation") && inputJson["simulation"].is_object()) {
            const auto& sim = inputJson["simulation"];
            if (sim.contains("index") && sim["index"].is_object()) {
                const auto& idx = sim["index"];
                if (idx.contains("start") && idx["start"].is_string() &&
                    idx.contains("end") && idx["end"].is_string() &&
                    idx.contains("timestep") && idx["timestep"].is_number() &&
                    idx.contains("length") && idx["length"].is_number()) {
                    json outIdx;
                    outIdx["start"] = idx["start"];
                    outIdx["end"] = idx["end"];
                    outIdx["timestep"] = idx["timestep"];
                    outIdx["length"] = idx["length"];
                    out["index"] = outIdx;
                }
            }
        }
    } catch (...) {
    }
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
    // 時系列で追いやすいよう、artifact 名は timestamp を先頭に置く。
    // 例: artifacts.1776566754847.output.<run_id>
    const std::string artifactDirName = "artifacts." + std::to_string(ArtifactIO::epochMillis()) + "." + outputStem;
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
    const std::string thermalTemperatureCapacityBinName = "thermal.temperature.capacity.f32.bin";
    const std::string thermalTemperatureLayerBinName = "thermal.temperature.layer.f32.bin";
    const std::string humidityXBinName = "humidity.x.f32.bin";
    const std::string humidityFluxBinName = "humidity.flux.f32.bin";
    const std::string concentrationCBinName = "concentration.c.f32.bin";
    const std::string concentrationFluxBinName = "concentration.flux.f32.bin";
    const std::string thermalHeatRateAdvectionBinName = "thermal.heat_rate.advection.f32.bin";
    const std::string thermalHeatRateHeatGenerationBinName = "thermal.heat_rate.heat_generation.f32.bin";
    const std::string thermalHeatRateSolarGainBinName = "thermal.heat_rate.solar_gain.f32.bin";
    const std::string thermalHeatRateNocturnalLossBinName = "thermal.heat_rate.nocturnal_loss.f32.bin";
    const std::string thermalHeatRateConvectionBinName = "thermal.heat_rate.convection.f32.bin";
    const std::string thermalHeatRateConductionBinName = "thermal.heat_rate.conduction.f32.bin";
    const std::string thermalHeatRateRadiationBinName = "thermal.heat_rate.radiation.f32.bin";
    const std::string thermalHeatRateCapacityBinName = "thermal.heat_rate.capacity.f32.bin";
    const std::string airconSensibleHeatBinName = "aircon.sensible_heat.f32.bin";
    const std::string airconLatentHeatBinName = "aircon.latent_heat.f32.bin";
    const std::string airconPowerBinName = "aircon.power.f32.bin";
    const std::string airconCOPBinName = "aircon.cop.f32.bin";

    const std::filesystem::path ventPressureBinPath = artifactDirPath / ventPressureBinName;
    const std::filesystem::path ventFlowRateBinPath = artifactDirPath / ventFlowRateBinName;
    const std::filesystem::path thermalTemperatureBinPath = artifactDirPath / thermalTemperatureBinName;
    const std::filesystem::path thermalTemperatureCapacityBinPath = artifactDirPath / thermalTemperatureCapacityBinName;
    const std::filesystem::path thermalTemperatureLayerBinPath = artifactDirPath / thermalTemperatureLayerBinName;
    const std::filesystem::path humidityXBinPath = artifactDirPath / humidityXBinName;
    const std::filesystem::path humidityFluxBinPath = artifactDirPath / humidityFluxBinName;
    const std::filesystem::path concentrationCBinPath = artifactDirPath / concentrationCBinName;
    const std::filesystem::path concentrationFluxBinPath = artifactDirPath / concentrationFluxBinName;
    const std::filesystem::path thermalHeatRateAdvectionBinPath = artifactDirPath / thermalHeatRateAdvectionBinName;
    const std::filesystem::path thermalHeatRateHeatGenerationBinPath = artifactDirPath / thermalHeatRateHeatGenerationBinName;
    const std::filesystem::path thermalHeatRateSolarGainBinPath = artifactDirPath / thermalHeatRateSolarGainBinName;
    const std::filesystem::path thermalHeatRateNocturnalLossBinPath = artifactDirPath / thermalHeatRateNocturnalLossBinName;
    const std::filesystem::path thermalHeatRateConvectionBinPath = artifactDirPath / thermalHeatRateConvectionBinName;
    const std::filesystem::path thermalHeatRateConductionBinPath = artifactDirPath / thermalHeatRateConductionBinName;
    const std::filesystem::path thermalHeatRateRadiationBinPath = artifactDirPath / thermalHeatRateRadiationBinName;
    const std::filesystem::path thermalHeatRateCapacityBinPath = artifactDirPath / thermalHeatRateCapacityBinName;
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
    std::ofstream thermalTemperatureCapacityFile(thermalTemperatureCapacityBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalTemperatureCapacityFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalTemperatureCapacityBinPath << "\n";
        return 1;
    }
    std::ofstream thermalTemperatureLayerFile(thermalTemperatureLayerBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalTemperatureLayerFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalTemperatureLayerBinPath << "\n";
        return 1;
    }
    std::ofstream humidityXFile(humidityXBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!humidityXFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << humidityXBinPath << "\n";
        return 1;
    }
    std::ofstream humidityFluxFile(humidityFluxBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!humidityFluxFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << humidityFluxBinPath << "\n";
        return 1;
    }
    std::ofstream concentrationCFile(concentrationCBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!concentrationCFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << concentrationCBinPath << "\n";
        return 1;
    }
    std::ofstream concentrationFluxFile(concentrationFluxBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!concentrationFluxFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << concentrationFluxBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateAdvectionFile(thermalHeatRateAdvectionBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateAdvectionFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateAdvectionBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateHeatGenerationFile(thermalHeatRateHeatGenerationBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateHeatGenerationFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateHeatGenerationBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateSolarGainFile(thermalHeatRateSolarGainBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateSolarGainFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateSolarGainBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateNocturnalLossFile(thermalHeatRateNocturnalLossBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateNocturnalLossFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateNocturnalLossBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateConvectionFile(thermalHeatRateConvectionBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateConvectionFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateConvectionBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateConductionFile(thermalHeatRateConductionBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateConductionFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateConductionBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateRadiationFile(thermalHeatRateRadiationBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateRadiationFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateRadiationBinPath << "\n";
        return 1;
    }
    std::ofstream thermalHeatRateCapacityFile(thermalHeatRateCapacityBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateCapacityFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateCapacityBinPath << "\n";
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
    std::vector<char> thermalTemperatureCapacityBuf(kFileBufferBytes);
    std::vector<char> thermalTemperatureLayerBuf(kFileBufferBytes);
    std::vector<char> humidityXBuf(kFileBufferBytes);
    std::vector<char> humidityFluxBuf(kFileBufferBytes);
    std::vector<char> concentrationCBuf(kFileBufferBytes);
    std::vector<char> concentrationFluxBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateAdvectionBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateHeatGenerationBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateSolarGainBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateNocturnalLossBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateConvectionBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateConductionBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateRadiationBuf(kFileBufferBytes);
    std::vector<char> thermalHeatRateCapacityBuf(kFileBufferBytes);
    std::vector<char> airconSensibleHeatBuf(kFileBufferBytes);
    std::vector<char> airconLatentHeatBuf(kFileBufferBytes);
    std::vector<char> airconPowerBuf(kFileBufferBytes);
    std::vector<char> airconCOPBuf(kFileBufferBytes);
    logFile.rdbuf()->pubsetbuf(logBuf.data(), static_cast<std::streamsize>(logBuf.size()));
    ventPressureFile.rdbuf()->pubsetbuf(ventPressureBuf.data(), static_cast<std::streamsize>(ventPressureBuf.size()));
    ventFlowRateFile.rdbuf()->pubsetbuf(ventFlowRateBuf.data(), static_cast<std::streamsize>(ventFlowRateBuf.size()));
    thermalTemperatureFile.rdbuf()->pubsetbuf(thermalTemperatureBuf.data(), static_cast<std::streamsize>(thermalTemperatureBuf.size()));
    thermalTemperatureCapacityFile.rdbuf()->pubsetbuf(thermalTemperatureCapacityBuf.data(), static_cast<std::streamsize>(thermalTemperatureCapacityBuf.size()));
    thermalTemperatureLayerFile.rdbuf()->pubsetbuf(thermalTemperatureLayerBuf.data(), static_cast<std::streamsize>(thermalTemperatureLayerBuf.size()));
    humidityXFile.rdbuf()->pubsetbuf(humidityXBuf.data(), static_cast<std::streamsize>(humidityXBuf.size()));
    humidityFluxFile.rdbuf()->pubsetbuf(humidityFluxBuf.data(), static_cast<std::streamsize>(humidityFluxBuf.size()));
    concentrationCFile.rdbuf()->pubsetbuf(concentrationCBuf.data(), static_cast<std::streamsize>(concentrationCBuf.size()));
    concentrationFluxFile.rdbuf()->pubsetbuf(concentrationFluxBuf.data(), static_cast<std::streamsize>(concentrationFluxBuf.size()));
    thermalHeatRateAdvectionFile.rdbuf()->pubsetbuf(thermalHeatRateAdvectionBuf.data(), static_cast<std::streamsize>(thermalHeatRateAdvectionBuf.size()));
    thermalHeatRateHeatGenerationFile.rdbuf()->pubsetbuf(thermalHeatRateHeatGenerationBuf.data(), static_cast<std::streamsize>(thermalHeatRateHeatGenerationBuf.size()));
    thermalHeatRateSolarGainFile.rdbuf()->pubsetbuf(thermalHeatRateSolarGainBuf.data(), static_cast<std::streamsize>(thermalHeatRateSolarGainBuf.size()));
    thermalHeatRateNocturnalLossFile.rdbuf()->pubsetbuf(thermalHeatRateNocturnalLossBuf.data(), static_cast<std::streamsize>(thermalHeatRateNocturnalLossBuf.size()));
    thermalHeatRateConvectionFile.rdbuf()->pubsetbuf(thermalHeatRateConvectionBuf.data(), static_cast<std::streamsize>(thermalHeatRateConvectionBuf.size()));
    thermalHeatRateConductionFile.rdbuf()->pubsetbuf(thermalHeatRateConductionBuf.data(), static_cast<std::streamsize>(thermalHeatRateConductionBuf.size()));
    thermalHeatRateRadiationFile.rdbuf()->pubsetbuf(thermalHeatRateRadiationBuf.data(), static_cast<std::streamsize>(thermalHeatRateRadiationBuf.size()));
    thermalHeatRateCapacityFile.rdbuf()->pubsetbuf(thermalHeatRateCapacityBuf.data(), static_cast<std::streamsize>(thermalHeatRateCapacityBuf.size()));
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
        // 入力読み込み/パースに失敗しているので index は出せない（空objectを渡す）
        writeErrorOutput(outputPath, artifactDirName, logFileName, json::object(), "", err, timings, writeErr);
        return 1;
    }

    const int requestedVerbosity = readRequestedVerbosity(inputData.inputJson);
    std::ostream& logs = (requestedVerbosity <= 0) ? g_null_stream : static_cast<std::ostream&>(logFile);
    writeLog(logs, "入力JSONを読み込みました。");

    OutputFiles outFiles{
        ventPressureFile,
        ventFlowRateFile,
        thermalTemperatureFile,
        thermalTemperatureCapacityFile,
        thermalTemperatureLayerFile,
        humidityXFile,
        humidityFluxFile,
        concentrationCFile,
        concentrationFluxFile,
        thermalHeatRateAdvectionFile,
        thermalHeatRateHeatGenerationFile,
        thermalHeatRateSolarGainFile,
        thermalHeatRateNocturnalLossFile,
        thermalHeatRateConvectionFile,
        thermalHeatRateConductionFile,
        thermalHeatRateRadiationFile,
        thermalHeatRateCapacityFile,
        airconSensibleHeatFile,
        airconLatentHeatFile,
        airconPowerFile,
        airconCOPFile,
    };

    auto simStart = std::chrono::steady_clock::now();
    bool simulationSuccess = runSimulationLoop(
        inputData,
        logs,
        outFiles,
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
    thermalHeatRateCapacityFile.close();
    thermalHeatRateRadiationFile.close();
    thermalHeatRateConductionFile.close();
    thermalHeatRateConvectionFile.close();
    thermalHeatRateNocturnalLossFile.close();
    thermalHeatRateSolarGainFile.close();
    thermalHeatRateHeatGenerationFile.close();
    thermalHeatRateAdvectionFile.close();
    thermalTemperatureFile.close();
    thermalTemperatureLayerFile.close();
    thermalTemperatureCapacityFile.close();
    humidityXFile.close();
    humidityFluxFile.close();
    concentrationCFile.close();
    concentrationFluxFile.close();
    ventFlowRateFile.close();
    ventPressureFile.close();
    logFile.close();

    if (simulationSuccess) {
        std::map<std::string, std::string> resultFiles = {
            {"schema", schemaFileName},
            {"vent_pressure", ventPressureBinName},
            {"vent_flow_rate", ventFlowRateBinName},
            {"thermal_temperature", thermalTemperatureBinName},
            {"thermal_temperature_capacity", thermalTemperatureCapacityBinName},
            {"thermal_temperature_layer", thermalTemperatureLayerBinName},
            {"humidity_x", humidityXBinName},
            {"humidity_flux", humidityFluxBinName},
            {"concentration_c", concentrationCBinName},
            {"concentration_flux", concentrationFluxBinName},
            {"thermal_heat_rate_advection", thermalHeatRateAdvectionBinName},
            {"thermal_heat_rate_heat_generation", thermalHeatRateHeatGenerationBinName},
            {"thermal_heat_rate_solar_gain", thermalHeatRateSolarGainBinName},
            {"thermal_heat_rate_nocturnal_loss", thermalHeatRateNocturnalLossBinName},
            {"thermal_heat_rate_convection", thermalHeatRateConvectionBinName},
            {"thermal_heat_rate_conduction", thermalHeatRateConductionBinName},
            {"thermal_heat_rate_radiation", thermalHeatRateRadiationBinName},
            {"thermal_heat_rate_capacity", thermalHeatRateCapacityBinName},
            {"aircon_sensible_heat", airconSensibleHeatBinName},
            {"aircon_latent_heat", airconLatentHeatBinName},
            {"aircon_power", airconPowerBinName},
            {"aircon_cop", airconCOPBinName},
        };
        if (!writeOutputData(outputPath, artifactDirName, logFileName, resultFiles, inputData.inputJson, inputData.inputContent, timings, err)) {
            std::cerr << err << "\n";
            return 1;
        }
    } else {
        std::string writeErr;
        if (!writeErrorOutput(outputPath, artifactDirName, logFileName, inputData.inputJson, inputData.inputContent, err, timings, writeErr)) {
            std::cerr << writeErr << "\n";
            return 1;
        }
        return 0;
    }

    return 0;
}


