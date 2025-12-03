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

// 換気回路網クラス
class VentilationNetwork {
private:
    Graph graph;
    std::unordered_map<std::string, Vertex> keyToVertex;  // キーから頂点への高速マッピング

    // スーパーノード検出のキャッシュ（1タイムステップ内で固定）
    bool supernodeCacheValid = false;
    std::map<Vertex, int> supernodeGroupCache; // vertex -> groupId (-1: 非属)

    // 直近の圧力計算が収束したかどうか
    bool lastPressureConverged = false;

public:
    // ノード・エッジ操作
    Vertex addNode(const VertexProperties& node);
    void addEdge(const EdgeProperties& edge);

    // ネットワーク情報
    int getNodeCount() const;
    int getEdgeCount() const;

    // ノード・頂点アクセス
    VertexProperties& getNode(const std::string& key);

    // グラフアクセス
    const Graph& getGraph() const { return graph; }
    Graph& getGraph() { return graph; }

    // キーマッピングアクセス
    const std::unordered_map<std::string, Vertex>& getKeyToVertex() const { return keyToVertex; }

    // 収束フラグ操作
    void setLastPressureConverged(bool v) { lastPressureConverged = v; }
    bool getLastPressureConverged() const { return lastPressureConverged; }

    // スーパーノードキャッシュ操作
    void invalidateSupernodeCache() { supernodeCacheValid = false; supernodeGroupCache.clear(); }
    bool hasSupernodeCache() const { return supernodeCacheValid; }
    const std::map<Vertex, int>& getSupernodeGroupCache() const { return supernodeGroupCache; }
    void setSupernodeGroupCache(const std::map<Vertex, int>& cache) { supernodeGroupCache = cache; supernodeCacheValid = true; }

    // データ構築
    void buildFromData(const std::vector<VertexProperties>& allNodes,
                       const std::vector<EdgeProperties>& ventilationBranches,
                       const SimulationConstants& simConstants,
                       std::ostream& logs);

    // 圧力計算メソッド（宣言のみ。実装は別途）
    std::tuple<PressureMap, std::map<std::pair<std::string, std::string>, double>, FlowBalanceMap> calculatePressure(
        const SimulationConstants& constants, std::ostream& logs);

    // 更新操作
    void updateNodePressures(const PressureMap& pressureMap);
    void updateCalculationResults(const PressureMap& pressureMap, const FlowRateMap& flowRates);
    void updateFlowRatesInGraph(const FlowRateMap& flowRates);
    void updateNodeTemperatures(const TemperatureMap& tempMap);
    
    // タイムステップに応じてノードとエッジの時変プロパティを更新
    void updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                     const std::vector<EdgeProperties>& ventilationBranches,
                                     long timestep);

    // 風量データ収集（個別ブランチの風量データを返す）
    std::map<std::string, double> collectFlowRates() const;

};


