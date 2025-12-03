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
    std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> calculateTemperature(
        const SimulationConstants& constants, std::ostream& logs);

    // 更新操作
    void updateNodeTemperatures(const TemperatureMap& tempMap);
    void updateHeatRatesInGraph(const HeatRateMap& heatRates);
    void updateCalculationResults(const TemperatureMap& tempMap, const HeatRateMap& heatRates);
    
    // タイムステップに応じてノードとエッジの時変プロパティを更新
    void updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                      const std::vector<EdgeProperties>& thermalBranches,
                                      const std::vector<EdgeProperties>& ventilationBranches,
                                      long timestep);

    // 熱流量データ収集（個別ブランチの熱流量データを返す）
    std::map<std::string, double> collectHeatRates() const;
};


