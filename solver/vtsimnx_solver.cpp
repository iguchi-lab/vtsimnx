#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <utility>

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

// ヘルパー: TimestepResultをJSONに変換
static json timestepResultToJson(const TimestepResult& result) {
    json j;
    
    // 圧力関連
    json pmapJson;
    for (const auto& [key, value] : result.pressureMap) {
        pmapJson[key] = value;
    }
    j["pressureMap"] = pmapJson;
    
    j["flowRateMap"] = result.flowRateMap;
    
    json fbmJson;
    for (const auto& [key, value] : result.flowBalanceMap) {
        fbmJson[key] = value;
    }
    j["flowBalanceMap"] = fbmJson;
    
    // 温度関連
    json tmapJson;
    for (const auto& [key, value] : result.temperatureMap) {
        tmapJson[key] = value;
    }
    j["temperatureMap"] = tmapJson;
    
    j["heatRateMap"] = result.heatRateMap;
    
    json hbmJson;
    for (const auto& [key, value] : result.heatBalanceMap) {
        hbmJson[key] = value;
    }
    j["heatBalanceMap"] = hbmJson;
    
    // エアコン関連
    j["airconInletTempMap"] = result.airconInletTempMap;
    j["airconOutletTempMap"] = result.airconOutletTempMap;
    j["airconFlowMap"] = result.airconFlowMap;
    j["airconSensibleHeatMap"] = result.airconSensibleHeatMap;
    j["airconLatentHeatMap"] = result.airconLatentHeatMap;
    j["airconPowerMap"] = result.airconPowerMap;
    j["airconCOPMap"] = result.airconCOPMap;
    
    return j;
}

// ヘルパー: JSON Linesファイルを読み込んで配列形式のJSONに変換
static json readJsonLinesToArray(const std::string& filePath, std::string& err) {
    std::ifstream ifs(filePath);
    if (!ifs) {
        err = std::string("エラー: 結果ファイルを開けません: ") + filePath;
        return json::array();
    }
    
    json resultArray = json::array();
    std::string line;
    int lineNum = 0;
    while (std::getline(ifs, line)) {
        lineNum++;
        if (line.empty()) continue;
        try {
            json lineJson = json::parse(line);
            resultArray.push_back(lineJson);
        } catch (const json::parse_error& e) {
            err = std::string("エラー: 結果ファイルの") + std::to_string(lineNum) + "行目をパースできません: " + e.what();
            // エラーがあっても続行（既に読み込んだ分は有効）
        }
    }
    return resultArray;
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
                              std::ofstream& logFile,
                              std::ofstream& resultsFile,
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
        TemperatureMap previousTimestepTemperatures;
        
        // タイムステップループ: 各ステップでプロパティのみ更新
        for (long timestepIndex = 0; timestepIndex < simConstants.length; timestepIndex++) {  //タイムステップループ
            std::ostringstream stepMeta;
            stepMeta << "timestep=" << (timestepIndex + 1);
            ScopedTimer timestepTimer(timings, "timestep", stepMeta.str());
            setLogTimestepMeta(logFile, static_cast<int>(timestepIndex + 1));
            ScopedLogSection timestepScope(
                logFile,
                "タイムステップ " + std::to_string(timestepIndex + 1) + " を実行中...",
                true);

            // 最適化: タイムステップごとに全データを再読み込みする代わりに、
            // 既に読み込んだ全データから、Graph内のプロパティのみを更新
            // ただし、最初のタイムステップでは既に読み込んでいるので、再度読み込む必要はない
            if (timestepIndex > 0) {
                writeLog(logFile, " トポロジ構築中...");
                auto topoStart = std::chrono::steady_clock::now();
                // トポロジ更新（時系列データ読込 + 熱容量ノード初期化までを計測）
                ScopedTimer timer(timings, "parse_timestep_data", stepMeta.str());
                // 2回目以降のタイムステップでは、全データを再読み込んで時系列データを更新
                // （JSONから読み込む必要があるため、パーサー関数を使用）
                allNodes               = parseNodes(inputData.inputJson, logFile, timestepIndex);
                allVentilationBranches = parseVentilationBranches(inputData.inputJson, logFile, timestepIndex);
                allThermalBranches     = parseThermalBranches(inputData.inputJson, logFile, timestepIndex);
                
                // グラフのプロパティ更新
                ventNetwork.updatePropertiesForTimestep(allNodes, allVentilationBranches, timestepIndex);
                thermalNetwork.updatePropertiesForTimestep(allNodes, allThermalBranches, allVentilationBranches, timestepIndex);

                // 熱容量ノードの温度を親ノードの前タイムステップ温度に設定（計測に含める）
                if (!previousTimestepTemperatures.empty()) {
                    auto& thermalGraph = thermalNetwork.getGraph();
                    auto vertex_range = boost::vertices(thermalGraph);
                    int capacityNodeCount = 0;
                    for (auto vertex : boost::make_iterator_range(vertex_range)) {
                        auto& nodeData = thermalGraph[vertex];
                        if (nodeData.type == "capacity" && !nodeData.ref_node.empty()) {
                            auto prevTempIt = previousTimestepTemperatures.find(nodeData.ref_node);
                            if (prevTempIt != previousTimestepTemperatures.end()) {
                                nodeData.current_t = prevTempIt->second;
                                capacityNodeCount++;
                                if (simConstants.logVerbosity >= 2) {
                                    writeLog(
                                        logFile,
                                        "熱容量ノード " + nodeData.key +
                                            " の初期温度を親ノード " + nodeData.ref_node +
                                            " の前温度 " + std::to_string(prevTempIt->second) + "℃ に設定");
                                }
                            }
                        }
                    }
                    if (capacityNodeCount > 0 && simConstants.logVerbosity >= 1) {
                        writeLog(
                            logFile,
                            "熱容量ノード温度を更新しました: " + std::to_string(capacityNodeCount) + "個");
                    }
                }

                // 計測対象に含まれるため、ここでログ出力
                auto topoEnd = std::chrono::steady_clock::now();
                double topoSec = std::chrono::duration_cast<std::chrono::duration<double>>(topoEnd - topoStart).count();
                std::ostringstream oss;
                oss << " トポロジ更新完了 (所要時間: " << std::fixed << std::setprecision(3) << topoSec << "秒)";
                writeLog(logFile, oss.str());
            }
            
            // シミュレーション実行
            SimulationResults results;
            if (simConstants.pressureCalc || simConstants.temperatureCalc) {
                ScopedTimer timer(timings, "runSimulation", stepMeta.str());
                runSimulation(ventNetwork, thermalNetwork, airconController, simConstants, results, logFile, timings, stepMeta.str());
            }
            
            // 次のタイムステップのために、現在のタイムステップの温度を保存
            if (simConstants.temperatureCalc && !results.timestepHistory.empty()) {
                previousTimestepTemperatures = results.timestepHistory.back().temperatureMap;
            }
            
            // タイムステップごとに結果をJSON Lines形式でファイルに書き込む
            {
                ScopedTimer timer(timings, "write_timestep_results", stepMeta.str());
                for (const auto& timestepResult : results.timestepHistory) {
                    json timestepJson = timestepResultToJson(timestepResult);
                    resultsFile << timestepJson.dump() << "\n";
                    resultsFile.flush(); // 即座にファイルに書き込む
                }
            }
        }
        
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
                            const std::string& logPath,
                            const std::string& resultsPath,
                            const std::string& inputContent,
                            const TimingList& timings,
                            std::string& err) {
    // ログファイルの内容を読み込む
    std::string logContent;
    std::string ioErr;
    if (!readFileToString(logPath.c_str(), logContent, ioErr)) {
        std::cerr << "警告: ログファイルを読み込めません: " << logPath << "\n";
        logContent = "ログファイルの読み込みに失敗しました。";
    }
    
    // JSON Linesファイルから結果を読み込んで配列形式に変換
    std::string resultsErr;
    json resultsArray = readJsonLinesToArray(resultsPath, resultsErr);
    if (!resultsErr.empty()) {
        logContent += "\n警告: " + resultsErr;
    }
    
    // 出力 JSON を構築
    json out;
    out["status"] = "success";
    out["input_length"] = inputContent.size();
    out["logs"] = logContent;
    out["results"] = resultsArray;
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
                             const std::string& logPath,
                             const std::string& resultsPath,
                             const std::string& inputContent,
                             const std::string& errorMessage,
                             const TimingList& timings,
                             std::string& err) {
    // ログファイルの内容を読み込む
    std::string logContent;
    std::string ioErr;
    if (!readFileToString(logPath.c_str(), logContent, ioErr)) {
        logContent = "ログファイルの読み込みに失敗しました。";
    }
    
    // エラー出力JSONを構築
    json out = {
        {"status", "error"},
        {"error", errorMessage},
        {"input_length", inputContent.size()},
        {"logs", logContent},
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
    // ログファイルのパスを決定（output.json -> output.log）
    std::string logPath = outputPath;
    size_t lastDot = logPath.find_last_of('.');
    if (lastDot != std::string::npos) {
        logPath = logPath.substr(0, lastDot) + ".log";
    } else {
        logPath += ".log";
    }

    // 結果ファイルのパスを決定（output.json -> output.results.jsonl）
    std::string resultsPath = outputPath;
    lastDot = resultsPath.find_last_of('.');
    if (lastDot != std::string::npos) {
        resultsPath = resultsPath.substr(0, lastDot) + ".results.jsonl";
    } else {
        resultsPath += ".results.jsonl";
    }

    // ログファイルを開く（追記モードで開き、既存の内容をクリア）
    std::ofstream logFile(logPath, std::ios::out | std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "エラー: ログファイルを開けません: " << logPath << "\n";
        return 1;
    }
    
    // 結果ファイルを開く（追記モードで開き、既存の内容をクリア）
    std::ofstream resultsFile(resultsPath, std::ios::out | std::ios::trunc);
    if (!resultsFile.is_open()) {
        std::cerr << "エラー: 結果ファイルを開けません: " << resultsPath << "\n";
        logFile.close();
        return 1;
    }
    
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
        resultsFile.close();
        std::string writeErr;
        writeErrorOutput(outputPath, logPath, resultsPath, "", err, timings, writeErr);
        return 1;
    }
    
    writeLog(logFile, "入力JSONを読み込みました。");

    // シミュレーション実行
    auto simStart = std::chrono::steady_clock::now();
    bool simulationSuccess = runSimulationLoop(inputData, logFile, resultsFile, err, timings);
    auto simEnd = std::chrono::steady_clock::now();
    double simMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(simEnd - simStart).count();
    timings.push_back({"simulation_total", simMs, ""});
    
    // ファイルを閉じる
    resultsFile.close();
    logFile.close();
    
    // ファイル出力処理
    if (simulationSuccess) {
        // 成功時: ログと結果を読み込んで出力JSONを構築
        if (!writeOutputData(outputPath, logPath, resultsPath, inputData.inputContent, timings, err)) {
            std::cerr << err << "\n";
            return 1;
        }
    } else {
        // 失敗時: エラー情報を出力JSONに書き込む
        std::string writeErr;
        if (!writeErrorOutput(outputPath, logPath, resultsPath, inputData.inputContent, err, timings, writeErr)) {
            std::cerr << writeErr << "\n";
            return 1;
        }
        return 0;
    }

    return 0;
}
