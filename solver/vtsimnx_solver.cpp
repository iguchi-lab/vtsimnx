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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"
#include "parser/sim_constants_parser.h"
#include "parser/nodes_parser.h"
#include "parser/branches_parser.h"
#include "network/ventilation_network.h"
#include "network/thermal_network.h"
#include "aircon/aircon_controller.h"
#include "simulation_runner.h"
#include "utils/utils.h"

// 出力JSONの形式バージョン（API側での互換性判断用）
static constexpr int kOutputFormatVersion = 5; // binary artifacts

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

// ヘルパー: float32 little-endian の配列をバイナリで書き出す
// 1 timestep = expectedSize 個を固定長で出力（不足分は 0.0f で埋める）
static inline void writeFloat32ArrayBinary(std::ofstream& ofs,
                                           const std::vector<float>& v,
                                           size_t expectedSize) {
    if (expectedSize == 0) return;
    const size_t n = std::min(expectedSize, v.size());
    if (n > 0) {
        ofs.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(n * sizeof(float)));
    }
    if (n < expectedSize) {
        // ゼロ埋め（不足分）
        thread_local std::vector<float> zeros;
        const size_t need = expectedSize - n;
        if (zeros.size() < need) zeros.assign(need, 0.0f);
        ofs.write(reinterpret_cast<const char*>(zeros.data()),
                  static_cast<std::streamsize>(need * sizeof(float)));
    }
}

static inline long long epochMillis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// ヘルパー: JSON をファイルに書き出す（失敗時は false, err に理由）
static bool writeJsonToFile(const char* path, const nlohmann::json& j, std::string& err) {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    if (!ofs) {
        err = std::string("エラー: 出力ファイルを開けません: ") + path;
        return false;
    }
    ofs << j.dump(2) << "\n";
    ofs.close();
    return true;
}

// 出力スキーマ（キー配列）: 計算結果はMap→配列で圧縮して出す
struct OutputSchema {
    std::vector<std::string> pressureKeys;
    std::vector<std::string> flowRateKeys;
    std::vector<std::string> temperatureKeys;
    std::vector<std::string> heatRateKeys;

    std::vector<std::string> airconSensibleHeatKeys;
    std::vector<std::string> airconLatentHeatKeys;
    std::vector<std::string> airconPowerKeys;
    std::vector<std::string> airconCOPKeys;

    bool initialized = false;
};

// （旧）map→配列変換用ヘルパー。出力を vector 化したため不要になった。

// schema.json（バイナリ出力のメタ情報）
static json schemaToJson(long length, long timestepSec, const OutputSchema& s) {
    json j;
    j["length"] = length;
    j["timestep_sec"] = timestepSec;
    j["dtype"] = "f32le";
    j["layout"] = "timestep-major";
    j["series"] = {
        {"vent_pressure", {{"keys", s.pressureKeys}}},
        {"vent_flow_rate", {{"keys", s.flowRateKeys}}},
        {"thermal_temperature", {{"keys", s.temperatureKeys}}},
        {"thermal_heat_rate", {{"keys", s.heatRateKeys}}},
        {"aircon_sensible_heat", {{"keys", s.airconSensibleHeatKeys}}},
        {"aircon_latent_heat", {{"keys", s.airconLatentHeatKeys}}},
        {"aircon_power", {{"keys", s.airconPowerKeys}}},
        {"aircon_cop", {{"keys", s.airconCOPKeys}}},
    };
    return j;
}

// --- timestep レコード（バイナリ） ---

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
                              std::ofstream& logFile,
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
            simConstants = parseSimulationConstants(inputData.inputJson, logFile);
        }

        // 詳細タイミング（各タイムステップ等）はオーバーヘッドになりやすいので、
        // 通常は抑制し、必要なときだけ有効化する。
        // - logVerbosity>=2 のときは有効
        // - それ以外は環境変数 VTSIMNX_TIMINGS が設定されていれば有効
        const bool enableDetailedTimings =
            (simConstants.logVerbosity >= 2) || (std::getenv("VTSIMNX_TIMINGS") != nullptr);
        setTimingsEnabled(enableDetailedTimings);

        // lengthがタイムステップの回数、timestepが1タイムステップの秒数
        writeLog(logFile, "タイムステップループ開始: length=" +
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
            ScopedLogSection initScope(logFile, "初期化: ネットワークトポロジー構築中...");
            writeLog(logFile, " トポロジ構築中...");
            auto topoStart = std::chrono::steady_clock::now();

            {
                ScopedTimer timer(timings, "initial_data_parse");
                allNodes               = parseNodes(inputData.inputJson, logFile, 0);
                allVentilationBranches = parseVentilationBranches(inputData.inputJson, logFile, 0);
                allThermalBranches     = parseThermalBranches(inputData.inputJson, logFile, 0);
            }
            
            {
                ScopedTimer timer(timings, "build_networks");
                ventNetwork.buildFromData(allNodes, allVentilationBranches, simConstants, logFile);
                thermalNetwork.buildFromData(allNodes, allThermalBranches, allVentilationBranches, simConstants, logFile);
            }
            
            if (simConstants.temperatureCalc) {
                airconController.initializeModels(thermalNetwork, logFile, simConstants.logVerbosity);
            }
            auto topoEnd = std::chrono::steady_clock::now();
            double topoSec = std::chrono::duration_cast<std::chrono::duration<double>>(topoEnd - topoStart).count();
            std::ostringstream oss;
            oss << "初期化完了: ネットワークトポロジー構築済み (所要時間: "
                << std::fixed << std::setprecision(3) << topoSec << "秒)";
            writeLog(logFile, oss.str());
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
        
        // タイムステップループ: 各ステップでプロパティのみ更新
        // 結果JSONLのflushは頻繁だとI/Oが支配的になりやすいので、一定行数ごとに間引く
        // （クラッシュ時のデータ損失は最大でこの間隔分）
        // 8ファイルに対して flush が走るため、間隔を大きめにして flush 回数を抑える。
        constexpr size_t kResultsFlushIntervalLines = 2000;
        size_t resultsLinesWritten = 0;
        OutputSchema schema;
        bool schemasWritten = false;
        for (long timestepIndex = 0; timestepIndex < simConstants.length; timestepIndex++) {  //タイムステップループ
            // 詳細タイミングが有効な場合のみ、メタ文字列を生成する
            std::string stepMeta;
            if (g_enable_timings) {
                stepMeta = "timestep=" + std::to_string(timestepIndex + 1);
            }
            ScopedTimer timestepTimer(timings, "timestep", stepMeta);
            setLogTimestepMeta(logFile, static_cast<int>(timestepIndex + 1));
            const bool verboseStepLog = (simConstants.logVerbosity >= 2);
            std::unique_ptr<ScopedLogSection> timestepScope;
            if (verboseStepLog) {
                timestepScope = std::make_unique<ScopedLogSection>(
                    logFile,
                    "タイムステップ " + std::to_string(timestepIndex + 1) + " を実行中...",
                    true);
            }

            // 最適化: タイムステップごとに全データを再読み込みする代わりに、
            // 既に読み込んだ全データから、Graph内のプロパティのみを更新
            // ただし、最初のタイムステップでは既に読み込んでいるので、再度読み込む必要はない
            if (timestepIndex > 0) {
                if (verboseStepLog) writeLog(logFile, " 時変プロパティ更新中...");
                // トポロジ更新（時系列データ読込 + 熱容量ノード初期化までを計測）
                ScopedTimer timer(timings, "parse_timestep_data", stepMeta);
                const auto topoStart = verboseStepLog ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                // グラフのプロパティ更新（初回に読み込んだ時系列を使い、再パースは行わない）
                ventNetwork.updatePropertiesForTimestep(allNodes, allVentilationBranches, timestepIndex);
                thermalNetwork.updatePropertiesForTimestep(allNodes, allThermalBranches, allVentilationBranches, timestepIndex);

                // 熱容量ノードの温度を親ノードの前タイムステップ温度に設定（計測に含める）
                if (simConstants.temperatureCalc && havePrevTemps && !capacityVertices.empty()) {
                    auto& thermalGraph = thermalNetwork.getGraph();
                    for (size_t i = 0; i < capacityVertices.size(); ++i) {
                        thermalGraph[capacityVertices[i]].current_t =
                            prevTempByVertex[static_cast<size_t>(capacityRefVertices[i])];
                    }
                    if (verboseStepLog) {
                        writeLog(logFile, "熱容量ノード温度を更新しました: " + std::to_string(capacityVertices.size()) + "個");
                    }
                }

                // 計測対象に含まれるため、ここでログ出力
                if (verboseStepLog) {
                    auto topoEnd = std::chrono::steady_clock::now();
                    double topoSec = std::chrono::duration_cast<std::chrono::duration<double>>(topoEnd - topoStart).count();
                    std::ostringstream oss;
                    oss << " トポロジ更新完了 (所要時間: " << std::fixed << std::setprecision(3) << topoSec << "秒)";
                    writeLog(logFile, oss.str());
                }
            }
            
            // シミュレーション実行
            TimestepResult timestepResult;
            if (simConstants.pressureCalc || simConstants.temperatureCalc) {
                ScopedTimer timer(timings, "runSimulation", stepMeta);
                runSimulation(ventNetwork, thermalNetwork, airconController, simConstants, timestepResult, logFile, timings, stepMeta);
            }
            
            // 次のタイムステップのために、現在のタイムステップの温度を保存（vertex index ベース）
            if (simConstants.temperatureCalc) {
                auto& thermalGraph = thermalNetwork.getGraph();
                const size_t vCount = static_cast<size_t>(boost::num_vertices(thermalGraph));
                if (prevTempByVertex.size() != vCount) prevTempByVertex.assign(vCount, 0.0);
                for (auto v : boost::make_iterator_range(boost::vertices(thermalGraph))) {
                    prevTempByVertex[static_cast<size_t>(v)] = thermalGraph[v].current_t;
                }
                havePrevTemps = true;
            }
            
            // タイムステップごとに結果をJSON Lines形式でファイルに書き込む
            {
                ScopedTimer timer(timings, "write_timestep_results", stepMeta);
                if (!schema.initialized) {
                    schema.pressureKeys = ventNetwork.getPressureKeys();
                    // flow_rate は VentilationNetwork 側でキー順を固定して提供する
                    schema.flowRateKeys = ventNetwork.getFlowRateKeys();
                    schema.temperatureKeys = thermalNetwork.getTemperatureKeys();
                    // heat_rate は ThermalNetwork 側でキー順を固定して提供する
                    schema.heatRateKeys = thermalNetwork.getHeatRateKeys();
                    // aircon は AirconController 側でキー順を固定して提供する
                    const auto& airconKeys = airconController.getAirconKeys();
                    schema.airconSensibleHeatKeys = airconKeys;
                    schema.airconLatentHeatKeys = airconKeys;
                    schema.airconPowerKeys = airconKeys;
                    schema.airconCOPKeys = airconKeys;
                    schema.initialized = true;
                }
                // バイナリ出力: schema は別ファイルへ1回だけ書く（呼び出し側で実施）
                if (!schemasWritten) {
                    std::string schemaErr;
                    const std::string schemaPathStr = schemaPath.string();
                    if (!writeJsonToFile(schemaPathStr.c_str(),
                                         schemaToJson(simConstants.length, simConstants.timestep, schema),
                                         schemaErr)) {
                        err = schemaErr;
                        return false;
                    }
                    resultsLinesWritten += 8;
                    schemasWritten = true;
                }

                writeFloat32ArrayBinary(ventPressureFile, timestepResult.pressure, schema.pressureKeys.size());
                writeFloat32ArrayBinary(ventFlowRateFile, timestepResult.flowRate, schema.flowRateKeys.size());
                writeFloat32ArrayBinary(thermalTemperatureFile, timestepResult.temperature, schema.temperatureKeys.size());
                writeFloat32ArrayBinary(thermalHeatRateFile, timestepResult.heatRate, schema.heatRateKeys.size());
                writeFloat32ArrayBinary(airconSensibleHeatFile, timestepResult.airconSensibleHeat, schema.airconSensibleHeatKeys.size());
                writeFloat32ArrayBinary(airconLatentHeatFile, timestepResult.airconLatentHeat, schema.airconLatentHeatKeys.size());
                writeFloat32ArrayBinary(airconPowerFile, timestepResult.airconPower, schema.airconPowerKeys.size());
                writeFloat32ArrayBinary(airconCOPFile, timestepResult.airconCOP, schema.airconCOPKeys.size());
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
        // ループ終了時に最終flush（プロセス正常終了ならcloseでもflushされるが明示しておく）
        ventPressureFile.flush();
        ventFlowRateFile.flush();
        thermalTemperatureFile.flush();
        thermalHeatRateFile.flush();
        airconSensibleHeatFile.flush();
        airconLatentHeatFile.flush();
        airconPowerFile.flush();
        airconCOPFile.flush();
        
        clearLogTimestepMeta(logFile);
        writeLog(logFile, "タイムステップループ終了");
        
        return true;
    } catch (const std::exception& e) {
        err = std::string("シミュレーション実行中にエラーが発生しました: ") + e.what();
        writeLog(logFile, "[ERROR] " + err);
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
    // 出力 JSON を構築
    json out;
    out["status"] = "ok";
    out["format_version"] = kOutputFormatVersion;
    out["input_length"] = inputContent.size();
    // ログ本文は output.json に含めない（巨大化防止）
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

    if (!writeJsonToFile(outputPath, out, err)) {
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
    // エラー出力JSONを構築
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
    
    if (!writeJsonToFile(outputPath, out, err)) {
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[])
{
    // 引数チェック
    // 使い方: vtsimnx_solver <input.json> <output.json>
    if (argc < 3) {
        std::cerr << "使い方: " << argv[0] << " <input.json> <output.json>\n";
        return 1;
    }

    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];

    // ファイルパスの決定
    // 出力JSON（argv[2]）は固定ファイル。実データは都度フォルダを作ってそこへ出す。
    const std::filesystem::path outputJsonPath = std::filesystem::path(outputPath);
    const std::filesystem::path outputParent = outputJsonPath.has_parent_path() ? outputJsonPath.parent_path()
                                                                                : std::filesystem::current_path();
    const std::string outputStem = outputJsonPath.stem().string().empty() ? "output" : outputJsonPath.stem().string();
    const std::string artifactDirName = outputStem + ".artifacts." + std::to_string(epochMillis());
    const std::filesystem::path artifactDirPath = outputParent / artifactDirName;

    std::error_code fsErr;
    std::filesystem::create_directories(artifactDirPath, fsErr);
    if (fsErr) {
        std::cerr << "エラー: 出力フォルダを作成できません: " << artifactDirPath.string() << " (" << fsErr.message()
                  << ")\n";
        return 1;
    }

    // ここから先の出力は全て artifactDirPath 配下へ
    const std::string logFileName = "solver.log";
    const std::filesystem::path logPath = artifactDirPath / logFileName;

    const std::string ventPressureName = "vent.pressure.jsonl";
    const std::string ventFlowRateName = "vent.flow_rate.jsonl";
    const std::string thermalTemperatureName = "thermal.temperature.jsonl";
    const std::string thermalHeatRateName = "thermal.heat_rate.jsonl";
    const std::string airconSensibleHeatName = "aircon.sensible_heat.jsonl";
    const std::string airconLatentHeatName = "aircon.latent_heat.jsonl";
    const std::string airconPowerName = "aircon.power.jsonl";
    const std::string airconCOPName = "aircon.cop.jsonl";
    const std::string schemaFileName = "schema.json";

    const std::filesystem::path ventPressurePath = artifactDirPath / ventPressureName;
    const std::filesystem::path ventFlowRatePath = artifactDirPath / ventFlowRateName;
    const std::filesystem::path thermalTemperaturePath = artifactDirPath / thermalTemperatureName;
    const std::filesystem::path thermalHeatRatePath = artifactDirPath / thermalHeatRateName;
    const std::filesystem::path airconSensibleHeatPath = artifactDirPath / airconSensibleHeatName;
    const std::filesystem::path airconLatentHeatPath = artifactDirPath / airconLatentHeatName;
    const std::filesystem::path airconPowerPath = artifactDirPath / airconPowerName;
    const std::filesystem::path airconCOPPath = artifactDirPath / airconCOPName;
    const std::filesystem::path schemaPath = artifactDirPath / schemaFileName;

    // ログファイルを開く（追記モードで開き、既存の内容をクリア）
    std::ofstream logFile(logPath, std::ios::out | std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "エラー: ログファイルを開けません: " << logPath << "\n";
        return 1;
    }
    
    // 結果ファイルを開く（既存の内容をクリア、バイナリ）
    // ※互換性は不要との方針により JSONL ではなく float32 little-endian の raw binary を出す
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

    std::ofstream ventPressureFile(ventPressureBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ventPressureFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << ventPressureBinPath << "\n";
        logFile.close();
        return 1;
    }
    std::ofstream ventFlowRateFile(ventFlowRateBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ventFlowRateFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << ventFlowRateBinPath << "\n";
        ventPressureFile.close();
        logFile.close();
        return 1;
    }
    std::ofstream thermalTemperatureFile(thermalTemperatureBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalTemperatureFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalTemperatureBinPath << "\n";
        ventFlowRateFile.close();
        ventPressureFile.close();
        logFile.close();
        return 1;
    }
    std::ofstream thermalHeatRateFile(thermalHeatRateBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!thermalHeatRateFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << thermalHeatRateBinPath << "\n";
        thermalTemperatureFile.close();
        ventFlowRateFile.close();
        ventPressureFile.close();
        logFile.close();
        return 1;
    }
    std::ofstream airconSensibleHeatFile(airconSensibleHeatBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconSensibleHeatFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconSensibleHeatBinPath << "\n";
        thermalHeatRateFile.close();
        thermalTemperatureFile.close();
        ventFlowRateFile.close();
        ventPressureFile.close();
        logFile.close();
        return 1;
    }
    std::ofstream airconLatentHeatFile(airconLatentHeatBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconLatentHeatFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconLatentHeatBinPath << "\n";
        airconSensibleHeatFile.close();
        thermalHeatRateFile.close();
        thermalTemperatureFile.close();
        ventFlowRateFile.close();
        ventPressureFile.close();
        logFile.close();
        return 1;
    }
    std::ofstream airconPowerFile(airconPowerBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconPowerFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconPowerBinPath << "\n";
        airconLatentHeatFile.close();
        airconSensibleHeatFile.close();
        thermalHeatRateFile.close();
        thermalTemperatureFile.close();
        ventFlowRateFile.close();
        ventPressureFile.close();
        logFile.close();
        return 1;
    }
    std::ofstream airconCOPFile(airconCOPBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!airconCOPFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << airconCOPBinPath << "\n";
        airconPowerFile.close();
        airconLatentHeatFile.close();
        airconSensibleHeatFile.close();
        thermalHeatRateFile.close();
        thermalTemperatureFile.close();
        ventFlowRateFile.close();
        ventPressureFile.close();
        logFile.close();
        return 1;
    }

    // 出力ファイルのバッファを大きくしてI/Oオーバーヘッドを低減
    // ※バッファは ofstream が生存している間ずっと有効である必要があるため、このスコープで保持する
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
    
    // ファイル入力処理
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
        logFile.close();
        ventPressureFile.close();
        ventFlowRateFile.close();
        thermalTemperatureFile.close();
        thermalHeatRateFile.close();
        airconSensibleHeatFile.close();
        airconLatentHeatFile.close();
        airconPowerFile.close();
        airconCOPFile.close();
        std::string writeErr;
        writeErrorOutput(outputPath, artifactDirName, logFileName, "", err, timings, writeErr);
        return 1;
    }
    
    writeLog(logFile, "入力JSONを読み込みました。");

    // シミュレーション実行
    auto simStart = std::chrono::steady_clock::now();
    bool simulationSuccess = runSimulationLoop(
        inputData,
        logFile,
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
    
    // ファイルを閉じる
    airconCOPFile.close();
    airconPowerFile.close();
    airconLatentHeatFile.close();
    airconSensibleHeatFile.close();
    thermalHeatRateFile.close();
    thermalTemperatureFile.close();
    ventFlowRateFile.close();
    ventPressureFile.close();
    logFile.close();
    
    // ファイル出力処理
    if (simulationSuccess) {
        // 成功時: output.json には「フォルダ名＋ファイル名」のみを書く（本文は入れない）
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
        // 失敗時: エラー情報を出力JSONに書き込む
        std::string writeErr;
        if (!writeErrorOutput(outputPath, artifactDirName, logFileName, inputData.inputContent, err, timings, writeErr)) {
            std::cerr << writeErr << "\n";
            return 1;
        }
        return 0;
    }

    return 0;
}
