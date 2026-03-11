#pragma once

#include "vtsim_solver.h"
#include "network/humidity_network.h"
#include <vector>
#include <fstream>
#include <ostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <tuple>
#include <map>
#include <string>

using json = nlohmann::json;

class VentilationNetwork; // forward declaration

// 熱回路網クラス
class ThermalNetwork {
private:
    Graph graph;
    std::unordered_map<std::string, Vertex> keyToVertex;  // キーから頂点への高速マッピング
    // 直近の熱計算の収束情報（ログだけだと上位ロジックが判断できないため保持する）
    bool lastThermalConverged = true;
    double lastThermalRmseBalance = 0.0;
    double lastThermalMaxBalance = 0.0;
    std::string lastThermalMethod;
    // 出力用（temperature）キャッシュ：キー順を固定して値配列で回す
    // 温度は 3 系列に分けて出力する:
    // - main   : normal + aircon + unknown
    // - capacity: capacity
    // - layer  : layer
    mutable bool temperatureCacheInitialized = false;
    mutable std::vector<Vertex> temperatureVerticesOrdered;          // main
    mutable std::vector<std::string> temperatureKeysOrdered;         // main
    mutable std::vector<Vertex> temperatureVerticesOrderedCapacity;  // capacity
    mutable std::vector<std::string> temperatureKeysOrderedCapacity; // capacity
    mutable std::vector<Vertex> temperatureVerticesOrderedLayer;     // layer
    mutable std::vector<std::string> temperatureKeysOrderedLayer;    // layer

    // 出力用（humidity x）キャッシュ：calc_x ノードをキー順で固定
    mutable bool humidityCacheInitialized = false;
    mutable std::vector<Vertex> humidityVerticesOrdered;
    mutable std::vector<std::string> humidityKeysOrdered;
    // 出力用（concentration c）キャッシュ：calc_c ノードをキー順で固定
    mutable bool concentrationCacheInitialized = false;
    mutable std::vector<Vertex> concentrationVerticesOrdered;
    mutable std::vector<std::string> concentrationKeysOrdered;
    // 出力用（heat_rate）キャッシュ：キー順を固定して値配列で回す
    mutable bool heatRateCacheInitialized = false;
    mutable std::vector<Edge> heatRateEdgesOrderedAdvection;
    mutable std::vector<std::string> heatRateKeysOrderedAdvection;
    mutable std::vector<Edge> heatRateEdgesOrderedHeatGeneration;
    mutable std::vector<std::string> heatRateKeysOrderedHeatGeneration;
    mutable std::vector<Edge> heatRateEdgesOrderedSolarGain;
    mutable std::vector<std::string> heatRateKeysOrderedSolarGain;
    mutable std::vector<Edge> heatRateEdgesOrderedNocturnalLoss;
    mutable std::vector<std::string> heatRateKeysOrderedNocturnalLoss;
    mutable std::vector<Edge> heatRateEdgesOrderedConvection;
    mutable std::vector<std::string> heatRateKeysOrderedConvection;
    mutable std::vector<Edge> heatRateEdgesOrderedConduction;
    mutable std::vector<std::string> heatRateKeysOrderedConduction;
    mutable std::vector<Edge> heatRateEdgesOrderedRadiation;
    mutable std::vector<std::string> heatRateKeysOrderedRadiation;
    mutable std::vector<Edge> heatRateEdgesOrderedCapacity;
    mutable std::vector<std::string> heatRateKeysOrderedCapacity;

    // 換気→熱の移流エッジ同期用キャッシュ（graph が不変な前提で構築は1回）
    // key: (sourceVertex<<32 | targetVertex)
    // value: 同一 source/target を持つ移流エッジ群（重複ペア対応）
    mutable bool advectionEdgeCacheInitialized = false;
    mutable std::unordered_map<std::uint64_t, std::vector<Edge>> advectionEdgesByVertexPair;
    // 換気枝 unique_id → 熱側移流エッジ（"advection_" + vent.unique_id で一意対応、順序に依存しない）
    mutable std::unordered_map<std::string, Edge> advectionEdgeByVentUniqueId;

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

    // 湿気計算用のネットワーク項を組み立てる（core/humidity から呼び出し）
    void buildHumidityNetworkTerms(const VentilationNetwork& ventNetwork,
                                   HumidityNetworkTerms& terms) const;

    // 換気回路網から風量を同期
    void syncFlowRatesFromVentilationNetwork(const VentilationNetwork& ventNetwork);

    // 計算（宣言のみ。実装は別途）
    void calculateTemperature(const SimulationConstants& constants, std::ostream& logs);

    // 直近の熱計算の収束情報（solver内部から set される）
    void setLastThermalConvergence(bool ok, double rmseBalance, double maxBalance, const std::string& method);
    bool getLastThermalConverged() const { return lastThermalConverged; }
    double getLastThermalRmseBalance() const { return lastThermalRmseBalance; }
    double getLastThermalMaxBalance() const { return lastThermalMaxBalance; }
    const std::string& getLastThermalMethod() const { return lastThermalMethod; }

    // 更新操作（互換性不要のため、計算結果は graph 内に反映される前提）
    
    // タイムステップに応じてノードとエッジの時変プロパティを更新
    void updatePropertiesForTimestep(const std::vector<VertexProperties>& allNodes,
                                      const std::vector<EdgeProperties>& thermalBranches,
                                      const std::vector<EdgeProperties>& ventilationBranches,
                                      long timestep);

    // 熱流量データ収集（個別ブランチの熱流量データを返す）
    // 温度（3系列）
    const std::vector<std::string>& getTemperatureKeys() const;
    std::vector<double> collectTemperatureValues() const;
    const std::vector<std::string>& getTemperatureKeysCapacity() const;
    std::vector<double> collectTemperatureValuesCapacity() const;
    const std::vector<std::string>& getTemperatureKeysLayer() const;
    std::vector<double> collectTemperatureValuesLayer() const;

    // 湿度（絶対湿度 x）
    const std::vector<std::string>& getHumidityKeys() const;
    std::vector<double> collectHumidityValues() const;

    // 濃度（c）
    const std::vector<std::string>& getConcentrationKeys() const;
    std::vector<double> collectConcentrationValues() const;

    // heat_rate（カテゴリ別）
    const std::vector<std::string>& getHeatRateKeysAdvection() const;
    std::vector<double> collectHeatRateValuesAdvection() const;
    const std::vector<std::string>& getHeatRateKeysHeatGeneration() const;
    std::vector<double> collectHeatRateValuesHeatGeneration() const;
    const std::vector<std::string>& getHeatRateKeysSolarGain() const;
    std::vector<double> collectHeatRateValuesSolarGain() const;
    const std::vector<std::string>& getHeatRateKeysNocturnalLoss() const;
    std::vector<double> collectHeatRateValuesNocturnalLoss() const;
    const std::vector<std::string>& getHeatRateKeysConvection() const;
    std::vector<double> collectHeatRateValuesConvection() const;
    const std::vector<std::string>& getHeatRateKeysConduction() const;
    std::vector<double> collectHeatRateValuesConduction() const;
    const std::vector<std::string>& getHeatRateKeysRadiation() const;
    std::vector<double> collectHeatRateValuesRadiation() const;
    const std::vector<std::string>& getHeatRateKeysCapacity() const;
    std::vector<double> collectHeatRateValuesCapacity() const;
};


