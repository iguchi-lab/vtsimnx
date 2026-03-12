#pragma once

#include "network/node_state_view.h"
#include "vtsim_solver.h"
#include <vector>
#include <fstream>
#include <ostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <tuple>
#include <map>

using json = nlohmann::json;

class ThermalNetwork; // forward declaration

// 換気回路網クラス
class VentilationNetwork {
private:
    Graph graph;
    std::unordered_map<std::string, Vertex> keyToVertex;  // キーから頂点への高速マッピング

    // 出力用（pressure）キャッシュ：キー順を固定して値配列で回す
    mutable bool pressureCacheInitialized = false;
    mutable std::vector<Vertex> pressureVerticesOrdered;
    mutable std::vector<std::string> pressureKeysOrdered;

    // 出力用（flow_rate）キャッシュ：キー順を固定して値配列で回す
    mutable bool flowRateCacheInitialized = false;
    mutable std::vector<Edge> flowRateEdgesOrdered;
    mutable std::vector<std::string> flowRateKeysOrdered;

    // スーパーノード検出のキャッシュ（1タイムステップ内で固定）
    bool supernodeCacheValid = false;
    std::map<Vertex, int> supernodeGroupCache; // vertex -> groupId (-1: 非属)

    // 直近の圧力計算が収束したかどうか
    bool lastPressureConverged = false;

public:
    // 1) Node/Graph access
    const Graph& getGraph() const { return graph; }
    Graph& getGraph() { return graph; }
    const std::unordered_map<std::string, Vertex>& getKeyToVertex() const { return keyToVertex; }
    ConstNodeStateView nodeStateView() const { return ConstNodeStateView{graph, keyToVertex}; }
    NodeStateView nodeStateView() { return NodeStateView{graph, keyToVertex}; }

    // ノード・エッジ操作
    Vertex addNode(const VertexProperties& node);
    void addEdge(const EdgeProperties& edge);
    int getNodeCount() const;
    int getEdgeCount() const;
    VertexProperties& getNode(const std::string& key);

    // 2) Build / Update / Sync
    void buildFromData(const std::vector<VertexProperties>& allNodes,
                       const std::vector<EdgeProperties>& ventilationBranches,
                       const SimulationConstants& simConstants,
                       std::ostream& logs);

    void updateNodePressures(const PressureMap& pressureMap);
    void updateCalculationResults(const PressureMap& pressureMap, const FlowRateMap& flowRates);
    void updateFlowRatesInGraph(const FlowRateMap& flowRates);
    void updateNodeTemperatures(const TemperatureMap& tempMap);
    // 温度マップを作らずに ThermalNetwork の graph から反映する
    void syncTemperaturesFromThermalNetwork(const ThermalNetwork& thermalNetwork);
    void updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                     const std::vector<EdgeProperties>& ventilationBranches,
                                     long timestep);

    // 3) Solve
    std::tuple<PressureMap, std::map<std::pair<std::string, std::string>, double>, FlowBalanceMap> solvePressure(
        const SimulationConstants& constants, std::ostream& logs);

    // 4) Output APIs
    const std::vector<std::string>& getPressureKeys() const;
    std::vector<double> collectPressureValues() const;
    const std::vector<std::string>& getFlowRateKeys() const;
    std::vector<double> collectFlowRateValues() const;

    // pressureCalc=false（固定流量など）の場合でも、aircon制御等が参照できるように
    // (sourceKey,targetKey)->flow_rate の map を生成する。
    FlowRateMap collectFlowRateMap() const;

    // 5) Diagnostics / cache controls
    void invalidateCaches();
    void setLastPressureConverged(bool v) { lastPressureConverged = v; }
    bool getLastPressureConverged() const { return lastPressureConverged; }
    void invalidateSupernodeCache() { supernodeCacheValid = false; supernodeGroupCache.clear(); }
    bool hasSupernodeCache() const { return supernodeCacheValid; }
    const std::map<Vertex, int>& getSupernodeGroupCache() const { return supernodeGroupCache; }
    void setSupernodeGroupCache(const std::map<Vertex, int>& cache) { supernodeGroupCache = cache; supernodeCacheValid = true; }

};


