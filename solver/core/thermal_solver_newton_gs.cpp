#include "core/thermal_solver_newton_gs.h"
#include "core/heat_calculation.h"
#include "utils/utils.h"
#include "../network/thermal_network.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace ThermalSolverNewtonGS {

namespace {

// 線形システム Ax = b を構築
struct LinearSystem {
    std::vector<std::vector<double>> A;  // 係数行列（疎行列として実装）
    std::vector<double> b;               // 右辺ベクトル
    std::vector<std::vector<size_t>> colIndices; // 各行の非ゼロ列インデックス
    
    void resize(size_t n) {
        A.resize(n);
        b.resize(n, 0.0);
        colIndices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            A[i].clear();
            colIndices[i].clear();
        }
    }
    
    void addCoefficient(size_t row, size_t col, double value) {
        if (std::abs(value) < 1e-15) return; // ゼロは無視
        
        auto& rowA = A[row];
        auto& rowCols = colIndices[row];
        
        // 既存の列かチェック
        auto it = std::find(rowCols.begin(), rowCols.end(), col);
        if (it != rowCols.end()) {
            size_t idx = std::distance(rowCols.begin(), it);
            rowA[idx] += value;
        } else {
            rowCols.push_back(col);
            rowA.push_back(value);
        }
    }
    
    double getCoefficient(size_t row, size_t col) const {
        const auto& rowCols = colIndices[row];
        auto it = std::find(rowCols.begin(), rowCols.end(), col);
        if (it != rowCols.end()) {
            size_t idx = std::distance(rowCols.begin(), it);
            return A[row][idx];
        }
        return 0.0;
    }
};

// 線形システムを構築
void buildLinearSystem(
    const Graph& graph,
    const std::vector<std::string>& nodeNames,
    const std::map<Vertex, size_t>& vertexToParameterIndex,
    const std::unordered_map<Vertex, std::vector<Edge>>& incidentEdges,
    const std::unordered_map<std::string, std::vector<Vertex>>& airconBySetNode,
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
    const std::vector<double>* tempsOverride,
    LinearSystem& system) {
    
    size_t n = nodeNames.size();
    system.resize(n);

    auto getTemp = [&](Vertex v) -> double {
        if (tempsOverride) {
            auto it = vertexToParameterIndex.find(v);
            if (it != vertexToParameterIndex.end()) {
                size_t idx = it->second;
                if (idx < tempsOverride->size()) return (*tempsOverride)[idx];
            }
        }
        return graph[v].current_t;
    };
    
    for (size_t i = 0; i < n; ++i) {
        const std::string& nodeName = nodeNames[i];
        auto nodeIt = nodeKeyToVertex.find(nodeName);
        if (nodeIt == nodeKeyToVertex.end()) continue;
        
        Vertex nodeVertex = nodeIt->second;
        const auto& nodeData = graph[nodeVertex];
        
        // エアコンのset_nodeがONの場合は固定温度（制約として扱う）
        auto itSetAc = airconBySetNode.find(nodeName);
        bool isFixed = false;
        if (itSetAc != airconBySetNode.end()) {
            for (auto v_ac : itSetAc->second) {
                const auto& nd = graph[v_ac];
                if (nd.type == "aircon" && nd.on && nd.set_node == nodeName) {
                    isFixed = true;
                    // 固定温度: T[i] = current_pre_temp（設定温度）
                    system.addCoefficient(i, i, 1.0);
                    system.b[i] = nd.current_pre_temp;
                    break;
                }
            }
        }
        
        if (isFixed) continue;
        
        // 熱バランス方程式: Σ(inflow) - Σ(outflow) = 0
        const auto& nodeEdgesIt = incidentEdges.find(nodeVertex);
        if (nodeEdgesIt == incidentEdges.end()) continue;
        
        double inflow = 0.0;
        double outflow = 0.0;
        
        for (auto edge : nodeEdgesIt->second) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& eprop = graph[edge];
            
            auto itS = vertexToParameterIndex.find(sv);
            auto itT = vertexToParameterIndex.find(tv);
            bool sIsVariable = (itS != vertexToParameterIndex.end());
            bool tIsVariable = (itT != vertexToParameterIndex.end());
            
            double dQdTs = 0.0, dQdTt = 0.0;
            
            if (eprop.type == "conductance") {
                dQdTs = eprop.conductance;
                dQdTt = -eprop.conductance;
            } else if (eprop.type == "advection") {
                double flowRate = eprop.flow_rate;
                if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
                    double mDotCpAbs = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
                    dQdTs = mDotCpAbs;
                    dQdTt = -mDotCpAbs;
                }
            } else if (eprop.type == "heat_generation") {
                // heat_generation は Q=constant なので係数0、導関数0
            }
            
            double Ts = getTemp(sv);
            double Tt = getTemp(tv);
            double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);
            
            if (sv == nodeVertex) {
                outflow += Q;
                if (sIsVariable) system.addCoefficient(i, itS->second, -dQdTs);
                if (tIsVariable) system.addCoefficient(i, itT->second, -dQdTt);
            } else if (tv == nodeVertex) {
                inflow += Q;
                if (sIsVariable) system.addCoefficient(i, itS->second, dQdTs);
                if (tIsVariable) system.addCoefficient(i, itT->second, dQdTt);
            }
        }
        
        // エアコンノードのset_nodeの熱バランスを肩代わり
        if (nodeData.type == "aircon" && nodeData.on && !nodeData.set_node.empty()) {
            auto itSet = nodeKeyToVertex.find(nodeData.set_node);
            if (itSet != nodeKeyToVertex.end()) {
                Vertex setV = itSet->second;
                const auto& setEdges = incidentEdges.find(setV);
                if (setEdges != incidentEdges.end()) {
                    double setIn = 0.0, setOut = 0.0;
                    for (auto e2 : setEdges->second) {
                        Vertex sv = boost::source(e2, graph);
                        Vertex tv = boost::target(e2, graph);
                        const auto& ep = graph[e2];
                        
                        auto is = vertexToParameterIndex.find(sv);
                        auto it = vertexToParameterIndex.find(tv);
                        bool sIsVar = (is != vertexToParameterIndex.end());
                        bool tIsVar = (it != vertexToParameterIndex.end());
                        
                        double dQdTs = 0.0, dQdTt = 0.0;
                        if (ep.type == "conductance") {
                            dQdTs = ep.conductance;
                            dQdTt = -ep.conductance;
                        } else if (ep.type == "advection") {
                            double flowRate = ep.flow_rate;
                            if (std::abs(flowRate) >= archenv::FLOW_RATE_MIN) {
                                double mDotCpAbs = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(flowRate);
                                dQdTs = mDotCpAbs;
                                dQdTt = -mDotCpAbs;
                            }
                        }
                        
                        double Ts = getTemp(sv);
                        double Tt = getTemp(tv);
                        double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep);
                        
                        if (sv == setV) {
                            // set_nodeから出る熱
                            setOut += Q;
                            if (sIsVar) system.addCoefficient(i, is->second, dQdTs);
                            if (tIsVar) system.addCoefficient(i, it->second, dQdTt);
                        } else if (tv == setV) {
                            // set_nodeに入る熱
                            setIn += Q;
                            if (sIsVar) system.addCoefficient(i, is->second, -dQdTs);
                            if (tIsVar) system.addCoefficient(i, it->second, -dQdTt);
                        }
                    }
                    // inflow - outflow に対して −(setIn - setOut) を加える → inflow += setOut - setIn
                    inflow += (setOut - setIn);
                }
            }
        }
        
        double net = inflow - outflow;
        system.b[i] = -net; // J * delta = -residual
    }
}

// Gauss-Seidel + SOR 反復法
bool solveGaussSeidel(
    const LinearSystem& system,
    std::vector<double>& x,
    double tolerance,
    int maxIterations,
    double omega,
    std::ostream& logFile) {
    
    size_t n = x.size();
    std::vector<double> xNew(n);
    
    // omega=1.0 が標準、1<omega<2で加速。異常値なら1.0に戻す
    if (!(omega > 0.0 && omega < 2.0)) omega = 1.0;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        double maxChange = 0.0;
        double maxResidual = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            double sum = system.b[i];
            const auto& rowA = system.A[i];
            const auto& rowCols = system.colIndices[i];
            
            // Σ(A[i][j] * x[j]) を計算（j != i）
            for (size_t k = 0; k < rowCols.size(); ++k) {
                size_t j = rowCols[k];
                if (j != i) {
                    sum -= rowA[k] * x[j];
                }
            }
            
            // 対角要素で割る
            double diag = system.getCoefficient(i, i);
            if (std::abs(diag) < 1e-15) {
                // 対角要素がゼロの場合はスキップ
                xNew[i] = x[i];
                continue;
            }
            
            double gsValue = sum / diag;
            // SOR 更新
            xNew[i] = x[i] + omega * (gsValue - x[i]);
            
            // 変数の変化量を計算
            double change = std::abs(xNew[i] - x[i]);
            if (change > maxChange) {
                maxChange = change;
            }
            
            // 即時更新（Gauss-Seidel）
            x[i] = xNew[i];
        }
        
        // 実際の残差を計算（Ax - b）
        for (size_t i = 0; i < n; ++i) {
            double residual = system.b[i];
            const auto& rowA = system.A[i];
            const auto& rowCols = system.colIndices[i];
            
            for (size_t k = 0; k < rowCols.size(); ++k) {
                size_t j = rowCols[k];
                residual -= rowA[k] * x[j];
            }
            
            double absResidual = std::abs(residual);
            if (absResidual > maxResidual) {
                maxResidual = absResidual;
            }
        }
        
        // 収束判定（実際の残差を使用）
        if (maxResidual < tolerance) {
            return true;
        }
        
    }
    
    // 最終残差をログに出力
    double finalResidual = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double residual = system.b[i];
        const auto& rowA = system.A[i];
        const auto& rowCols = system.colIndices[i];
        
        for (size_t k = 0; k < rowCols.size(); ++k) {
            size_t j = rowCols[k];
            residual -= rowA[k] * x[j];
        }
        
        double absResidual = std::abs(residual);
        if (absResidual > finalResidual) {
            finalResidual = absResidual;
        }
    }
    
    std::ostringstream oss;
    oss << "--------警告: Newton-GS+SORが最大反復回数(" << maxIterations 
        << ")に達しました | 最終残差=" << std::scientific << std::setprecision(6) << finalResidual
        << " | 必要許容誤差=" << tolerance
        << " | omega=" << omega;
    writeLog(logFile, oss.str());
    return false;
}

} // namespace

std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> solveTemperaturesNewtonGS(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    double omega,
    std::ostream& logFile) {
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    const auto& graph = network.getGraph();
    std::unordered_map<Vertex, std::vector<Edge>> incidentEdges;
    incidentEdges.reserve(boost::num_vertices(graph));
    auto incident_edge_range = boost::edges(graph);
    for (auto edge : boost::make_iterator_range(incident_edge_range)) {
        Vertex sv = boost::source(edge, graph);
        Vertex tv = boost::target(edge, graph);
        incidentEdges[sv].push_back(edge);
        incidentEdges[tv].push_back(edge);
    }
    
    std::unordered_map<std::string, std::vector<Vertex>> airconBySetNode;
    auto vertex_range_aircon = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range_aircon)) {
        const auto& properties = graph[vertex];
        if (properties.type == "aircon" && !properties.set_node.empty()) {
            airconBySetNode[properties.set_node].push_back(vertex);
        }
    }
    
    std::vector<std::string> nodeNames;
    std::map<Vertex, size_t> vertexToParameterIndex;
    size_t parameterIndex = 0;
    
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& properties = graph[vertex];
        if (properties.calc_t) {
            nodeNames.push_back(properties.key);
            vertexToParameterIndex[vertex] = parameterIndex++;
        }
    }
    
    if (nodeNames.empty()) {
        writeLog(logFile, "--警告: 温度計算対象のノードがありません");
        return {TemperatureMap{}, HeatRateMap{}, HeatBalanceMap{}};
    }
    
    // 初期温度を設定
    std::vector<double> temperatures(nodeNames.size());
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        auto it = network.getKeyToVertex().find(nodeNames[i]);
        if (it != network.getKeyToVertex().end()) {
            temperatures[i] = graph[it->second].current_t;
        }
    }
    
    // 非線形（熱バランス）チェックのため、線形システムを再構築しつつ反復
    const int maxRebuilds = 1; // 再構築は行わず1回のみ解く
    double maxBalance = std::numeric_limits<double>::infinity();

    for (int rebuild = 1; rebuild <= maxRebuilds; ++rebuild) {
        LinearSystem system;
        buildLinearSystem(
            graph,
            nodeNames,
            vertexToParameterIndex,
            incidentEdges,
            airconBySetNode,
            network.getKeyToVertex(),
            &temperatures,
            system);
        
        // Newton-GS+SORで解く
        // Gauss-Seidelの最大反復回数を緩めて収束しやすくする（デフォルト100 → 300以上）
        int maxIterations = static_cast<int>(std::max(constants.maxInnerIteration * 3.0, 300.0));

        std::vector<double> delta(nodeNames.size(), 0.0);
        bool gsConverged = solveGaussSeidel(
            system,
            delta,
            constants.thermalTolerance,
            maxIterations,
            omega,
            logFile);

        // 解いた増分を温度に反映（ウォームスタート継続）
        for (size_t i = 0; i < nodeNames.size(); ++i) {
            temperatures[i] += delta[i];
        }

        // 温度マップを更新
        TemperatureMap temperatureMap;
        for (size_t i = 0; i < nodeNames.size(); ++i) {
            temperatureMap[nodeNames[i]] = temperatures[i];
        }

        // バランス検証
        HeatBalanceMap heatBalance;
        maxBalance = 0.0;
        double rmseBalance = 0.0;
        for (const std::string& nodeName : nodeNames) {
            auto it = network.getKeyToVertex().find(nodeName);
            if (it == network.getKeyToVertex().end()) continue;
            
            Vertex nodeVertex = it->second;
            double inflow = 0.0, outflow = 0.0;
            
            const auto& nodeEdges = incidentEdges.find(nodeVertex);
            if (nodeEdges != incidentEdges.end()) {
                for (auto edge : nodeEdges->second) {
                    Vertex sv = boost::source(edge, graph);
                    Vertex tv = boost::target(edge, graph);
                    const auto& eprop = graph[edge];
                    
                    double Ts = temperatureMap.count(graph[sv].key) ? temperatureMap[graph[sv].key] : graph[sv].current_t;
                    double Tt = temperatureMap.count(graph[tv].key) ? temperatureMap[graph[tv].key] : graph[tv].current_t;
                    
                    double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);
                    
                    if (sv == nodeVertex) {
                        outflow += Q;
                    } else if (tv == nodeVertex) {
                        inflow += Q;
                    }
                }
            }

            // Aircon肩代わり（線形システムと同じ処理をバランス検証に反映）
            const auto& nodeData = graph[nodeVertex];
            if (nodeData.type == "aircon" && nodeData.on && !nodeData.set_node.empty()) {
                auto itSet = network.getKeyToVertex().find(nodeData.set_node);
                if (itSet != network.getKeyToVertex().end()) {
                    Vertex setV = itSet->second;
                    double setIn = 0.0, setOut = 0.0;
                    const auto& setEdges = incidentEdges.find(setV);
                    if (setEdges != incidentEdges.end()) {
                        for (auto e2 : setEdges->second) {
                            Vertex sv = boost::source(e2, graph);
                            Vertex tv = boost::target(e2, graph);
                            const auto& ep = graph[e2];
                            double Ts = temperatureMap.count(graph[sv].key) ? temperatureMap[graph[sv].key] : graph[sv].current_t;
                            double Tt = temperatureMap.count(graph[tv].key) ? temperatureMap[graph[tv].key] : graph[tv].current_t;
                            double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep);
                            if (sv == setV) {
                                setOut += Q;
                            } else if (tv == setV) {
                                setIn += Q;
                            }
                        }
                    }
                    inflow += -(setIn - setOut); // inflow - outflow に対して肩代わりを加える
                }
            }
            
            double balance = inflow - outflow;
            heatBalance[nodeName] = balance;
            maxBalance = std::max(maxBalance, std::abs(balance));
            rmseBalance += balance * balance;
        }
        rmseBalance = std::sqrt(rmseBalance / static_cast<double>(nodeNames.size()));

        if (gsConverged && rmseBalance <= constants.thermalTolerance) {
            writeLog(logFile, "--------Newton-GS+SORで収束しました (RMSE=" + std::to_string(rmseBalance) +
                               ", maxBalance=" + std::to_string(maxBalance) +
                               ", tol=" + std::to_string(constants.thermalTolerance) + ")");

            // heatRates/heatBalance を計算して返す
            HeatRateMap heatRates;
            for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
                Vertex sv = boost::source(edge, graph);
                Vertex tv = boost::target(edge, graph);
                const auto& eprop = graph[edge];
                
                double Ts = temperatureMap.count(graph[sv].key) ? temperatureMap[graph[sv].key] : graph[sv].current_t;
                double Tt = temperatureMap.count(graph[tv].key) ? temperatureMap[graph[tv].key] : graph[tv].current_t;
                
                double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);
                std::pair<std::string, std::string> edgeKey = std::make_pair(graph[sv].key, graph[tv].key);
                heatRates[edgeKey] = Q;
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            double seconds = duration.count() / 1000.0;
            std::ostringstream oss;
            oss << "--------Newton-GS+SOR 所要時間: " << std::fixed << std::setprecision(3) << seconds << "秒";
            writeLog(logFile, oss.str());

            return {temperatureMap, heatRates, heatBalance};
        }

        // 収束していない場合
        writeLog(logFile, "--------Newton-GS+SORは収束未達または熱バランス超過 (RMSE=" + std::to_string(rmseBalance) +
                           ", maxBalance=" + std::to_string(maxBalance) +
                           ", tol=" + std::to_string(constants.thermalTolerance) + ").");
    }

    // ここまでで収束しなかった場合
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double seconds = duration.count() / 1000.0;
    std::ostringstream oss;
    oss << "--------警告: Newton-GS+SORが収束しませんでした (RMSE不満足, maxBalance=" << maxBalance
        << ", tol=" << constants.thermalTolerance << ", 所要時間=" << std::fixed << std::setprecision(3) << seconds << "秒)";
    writeLog(logFile, oss.str());

    // 最後に計算した温度を返し、以降のフォールバックに任せる
    TemperatureMap temperatureMap;
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        temperatureMap[nodeNames[i]] = temperatures[i];
    }
    HeatRateMap heatRates; // 後段で再計算されるので空のままでOK
    HeatBalanceMap heatBalance; // 後段で再計算されるので空のままでOK
    return {temperatureMap, heatRates, heatBalance};
}

} // namespace ThermalSolverNewtonGS



