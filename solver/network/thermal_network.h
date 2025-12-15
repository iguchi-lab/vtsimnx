#pragma once

#include "vtsim_solver.h"
#include <vector>
#include <fstream>
#include <ostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <tuple>
#include <map>

using json = nlohmann::json;

class VentilationNetwork; // forward declaration

// 熱回路網クラス
class ThermalNetwork {
private:
    Graph graph;
    std::unordered_map<std::string, Vertex> keyToVertex;  // キーから頂点への高速マッピング
    // 出力用（temperature）キャッシュ：キー順を固定して値配列で回す
    mutable bool temperatureCacheInitialized = false;
    mutable std::vector<Vertex> temperatureVerticesOrdered;
    mutable std::vector<std::string> temperatureKeysOrdered;
    // 出力用（heat_rate）キャッシュ：キー順を固定して値配列で回す
    mutable bool heatRateCacheInitialized = false;
    mutable std::vector<Edge> heatRateEdgesOrdered;
    mutable std::vector<std::string> heatRateKeysOrdered;

public:
    // ノード・エッジ操作
    Vertex addNode(const VertexProperties& node);
    void addEdge(const EdgeProperties& edge);

    // ネットワーク情報
    int getNodeCount() const;
    int getEdgeCount() const;

    // ノード・頂点アクセス
    VertexProperties& getNode(const std::string& key);
    const VertexProperties& getNode(const std::string& key) const;

    // グラフアクセス
    const Graph& getGraph() const { return graph; }
    Graph& getGraph() { return graph; }
    const std::unordered_map<std::string, Vertex>& getKeyToVertex() const { return keyToVertex; }

    // データ構築（換気ブランチは別途同期）
    void buildFromData(const std::vector<VertexProperties>& allNodes,
                       const std::vector<EdgeProperties>& thermalBranches,
                       const std::vector<EdgeProperties>& ventilationBranches,
                       const SimulationConstants& simConstants,
                       std::ostream& logs);

    // 換気回路網から風量を同期
    void syncFlowRatesFromVentilationNetwork(const VentilationNetwork& ventNetwork);

    // 計算（宣言のみ。実装は別途）
    void calculateTemperature(const SimulationConstants& constants, std::ostream& logs);

    // 更新操作（互換性不要のため、計算結果は graph 内に反映される前提）
    
    // タイムステップに応じてノードとエッジの時変プロパティを更新
    void updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                      const std::vector<EdgeProperties>& thermalBranches,
                                      const std::vector<EdgeProperties>& ventilationBranches,
                                      long timestep);

    // 熱流量データ収集（個別ブランチの熱流量データを返す）
    const std::vector<std::string>& getTemperatureKeys() const;
    std::vector<double> collectTemperatureValues() const;
    const std::vector<std::string>& getHeatRateKeys() const;
    std::vector<double> collectHeatRateValues() const;
};


