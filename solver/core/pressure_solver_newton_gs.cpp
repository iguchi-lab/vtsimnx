#include "core/pressure_solver_newton_gs.h"
#include "core/flow_calculation.h"
#include "utils/utils.h"
#include "../network/ventilation_network.h"
#include "../archenv/include/archenv.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace PressureSolverNewtonGS {

namespace {

double calculateDensity(double temperature) {
    return archenv::STANDARD_ATMOSPHERIC_PRESSURE /
           (archenv::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

double calculateTotalPressure(double pressure, double temperature, double height) {
    double rho = calculateDensity(temperature);
    return pressure - rho * archenv::GRAVITY * height;
}

struct LinearSystem {
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    std::vector<std::vector<size_t>> colIndices;

    void resize(size_t n) {
        A.resize(n);
        b.assign(n, 0.0);
        colIndices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            A[i].clear();
            colIndices[i].clear();
        }
    }

    void addCoefficient(size_t row, size_t col, double value) {
        if (std::abs(value) < 1e-15) return;
        auto& rowA = A[row];
        auto& rowCols = colIndices[row];
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

void buildLinearSystem(
    const Graph& graph,
    const std::vector<std::string>& nodeNames,
    const std::map<Vertex, size_t>& vertexToParameterIndex,
    const std::unordered_map<Vertex, std::vector<Edge>>& incidentEdges,
    const std::vector<double>* pOverride,
    LinearSystem& system) {

    auto getPressure = [&](Vertex v) -> double {
        if (pOverride) {
            auto it = vertexToParameterIndex.find(v);
            if (it != vertexToParameterIndex.end() && it->second < pOverride->size()) {
                return (*pOverride)[it->second];
            }
        }
        return graph[v].current_p;
    };

    auto getTotalPressure = [&](Vertex v, const EdgeProperties& edgeData, bool fromSide) -> double {
        const auto& nd = graph[v];
        double p = getPressure(v);
        double h = fromSide ? edgeData.h_from : edgeData.h_to;
        return calculateTotalPressure(p, nd.current_t, h);
    };

    size_t n = nodeNames.size();
    system.resize(n);

    for (size_t i = 0; i < n; ++i) {
        Vertex nodeVertex{};
        for (const auto& kv : vertexToParameterIndex) {
            if (kv.second == i) {
                nodeVertex = kv.first;
                break;
            }
        }

        const auto& nodeData = graph[nodeVertex];

        // 固定圧力ノードは行をスキップ（calc_p=falseは含まれない想定）
        if (!nodeData.calc_p) continue;

        auto edgesIt = incidentEdges.find(nodeVertex);
        if (edgesIt == incidentEdges.end()) continue;

        double inflow = 0.0;
        double outflow = 0.0;

        for (auto edge : edgesIt->second) {
            Vertex sv = boost::source(edge, graph);
            Vertex tv = boost::target(edge, graph);
            const auto& eprop = graph[edge];

            auto itS = vertexToParameterIndex.find(sv);
            auto itT = vertexToParameterIndex.find(tv);
            bool sIsVar = (itS != vertexToParameterIndex.end());
            bool tIsVar = (itT != vertexToParameterIndex.end());

            double ps_total = getTotalPressure(sv, eprop, true);
            double pt_total = getTotalPressure(tv, eprop, false);
            double dp = ps_total - pt_total;

            double flow = FlowCalculation::calculateUnifiedFlow(dp, eprop);

            // 数値微分で dQ/dp (dp at source, dp at target => +1, -1)
            double eps = std::max(1.0, std::abs(dp)) * 1e-6;
            if (eps < 1e-9) eps = 1e-9;
            double flowPlus = FlowCalculation::calculateUnifiedFlow(dp + eps, eprop);
            double flowMinus = FlowCalculation::calculateUnifiedFlow(dp - eps, eprop);
            double dQddp = (flowPlus - flowMinus) / (2.0 * eps);

            if (sv == nodeVertex) {
                outflow += flow;
                if (sIsVar) system.addCoefficient(i, itS->second, -dQddp);
                if (tIsVar) system.addCoefficient(i, itT->second, dQddp);
            } else if (tv == nodeVertex) {
                inflow += flow;
                if (sIsVar) system.addCoefficient(i, itS->second, dQddp);
                if (tIsVar) system.addCoefficient(i, itT->second, -dQddp);
            }
        }

        double net = inflow - outflow;
        system.b[i] = -net; // J * deltaP = -residual
    }
}

bool solveGaussSeidel(
    const LinearSystem& system,
    std::vector<double>& x,
    double tolerance,
    int maxIterations,
    double omega,
    std::ostream& /*logFile*/) {

    size_t n = x.size();
    std::vector<double> xNew(n);
    if (!(omega > 0.0 && omega < 2.0)) omega = 1.0;

    for (int iter = 0; iter < maxIterations; ++iter) {
        double maxResidual = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double sum = system.b[i];
            const auto& rowA = system.A[i];
            const auto& rowCols = system.colIndices[i];

            for (size_t k = 0; k < rowCols.size(); ++k) {
                size_t j = rowCols[k];
                if (j != i) {
                    sum -= rowA[k] * x[j];
                }
            }

            double diag = system.getCoefficient(i, i);
            if (std::abs(diag) < 1e-15) {
                xNew[i] = x[i];
                continue;
            }

            double gsValue = sum / diag;
            xNew[i] = x[i] + omega * (gsValue - x[i]);
            x[i] = xNew[i];
        }

        for (size_t i = 0; i < n; ++i) {
            double residual = system.b[i];
            const auto& rowA = system.A[i];
            const auto& rowCols = system.colIndices[i];
            for (size_t k = 0; k < rowCols.size(); ++k) {
                size_t j = rowCols[k];
                residual -= rowA[k] * x[j];
            }
            maxResidual = std::max(maxResidual, std::abs(residual));
        }

        if (maxResidual < tolerance) {
            return true;
        }
    }

    return false;
}

} // namespace

std::tuple<PressureMap, FlowRateMap, FlowBalanceMap> solvePressuresNewtonGS(
    VentilationNetwork& network,
    const SimulationConstants& constants,
    double omega,
    std::ostream& logFile) {

    auto startTime = std::chrono::high_resolution_clock::now();
    const auto& graph = network.getGraph();

    std::unordered_map<Vertex, std::vector<Edge>> incidentEdges;
    incidentEdges.reserve(boost::num_vertices(graph));
    for (auto edge : boost::make_iterator_range(boost::edges(graph))) {
        Vertex sv = boost::source(edge, graph);
        Vertex tv = boost::target(edge, graph);
        incidentEdges[sv].push_back(edge);
        incidentEdges[tv].push_back(edge);
    }

    std::vector<std::string> nodeNames;
    std::map<Vertex, size_t> vertexToParameterIndex;
    size_t parameterIndex = 0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const auto& nd = graph[v];
        if (nd.calc_p) {
            nodeNames.push_back(nd.key);
            vertexToParameterIndex[v] = parameterIndex++;
        }
    }

    if (nodeNames.empty()) {
        writeLog(logFile, "--警告: 圧力計算対象のノードがありません");
        return {PressureMap{}, FlowRateMap{}, FlowBalanceMap{}};
    }

    std::vector<double> pressures(nodeNames.size());
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        auto it = network.getKeyToVertex().find(nodeNames[i]);
        if (it != network.getKeyToVertex().end()) {
            pressures[i] = graph[it->second].current_p;
        }
    }

    LinearSystem system;
    buildLinearSystem(
        graph,
        nodeNames,
        vertexToParameterIndex,
        incidentEdges,
        &pressures,
        system);

    int maxIterations = static_cast<int>(std::max(constants.maxInnerIteration * 3.0, 300.0));
    std::vector<double> delta(nodeNames.size(), 0.0);
    bool gsConverged = solveGaussSeidel(
        system,
        delta,
        constants.ventilationTolerance,
        maxIterations,
        omega,
        logFile);

    for (size_t i = 0; i < nodeNames.size(); ++i) {
        pressures[i] += delta[i];
    }

    // 結果組み立て
    PressureMap pressureMap;
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        pressureMap[nodeNames[i]] = pressures[i];
    }
    const auto& g = network.getGraph();
    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        const auto& nd = g[v];
        if (!nd.calc_p) {
            pressureMap[nd.key] = nd.current_p;
        }
    }

    FlowRateMap flowRates;
    for (auto edge : boost::make_iterator_range(boost::edges(g))) {
        Vertex sv = boost::source(edge, g);
        Vertex tv = boost::target(edge, g);
        const auto& eprop = g[edge];
        const auto& sn = g[sv];
        const auto& tn = g[tv];

        double ps = pressureMap.count(sn.key) ? pressureMap[sn.key] : sn.current_p;
        double pt = pressureMap.count(tn.key) ? pressureMap[tn.key] : tn.current_p;
        double ps_total = calculateTotalPressure(ps, sn.current_t, eprop.h_from);
        double pt_total = calculateTotalPressure(pt, tn.current_t, eprop.h_to);
        double dp = ps_total - pt_total;
        double q = FlowCalculation::calculateUnifiedFlow(dp, eprop);
        flowRates[{sn.key, tn.key}] = q;
    }

    FlowBalanceMap balance;
    for (auto v : boost::make_iterator_range(boost::vertices(g))) {
        balance[g[v].key] = 0.0;
    }
    for (const auto& kv : flowRates) {
        const auto& edgeKey = kv.first;
        double q = kv.second;
        balance[edgeKey.first] -= q;
        balance[edgeKey.second] += q;
    }

    double rmse = 0.0;
    double maxBalance = 0.0;
    for (const auto& kv : balance) {
        double b = kv.second;
        maxBalance = std::max(maxBalance, std::abs(b));
        rmse += b * b;
    }
    rmse = std::sqrt(rmse / static_cast<double>(balance.size()));

    auto endTime = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0;

    if (gsConverged && rmse <= constants.ventilationTolerance) {
        std::ostringstream oss;
        oss << "--------Newton-GS+SOR(圧力)で収束しました (RMSE=" << rmse
            << ", maxBalance=" << maxBalance
            << ", tol=" << constants.ventilationTolerance
            << ", time=" << std::fixed << std::setprecision(3) << seconds << "秒)";
        writeLog(logFile, oss.str());
    } else {
        std::ostringstream oss;
        oss << "--------Newton-GS+SOR(圧力)未収束 (RMSE=" << rmse
            << ", maxBalance=" << maxBalance
            << ", tol=" << constants.ventilationTolerance
            << ", time=" << std::fixed << std::setprecision(3) << seconds << "秒)";
        writeLog(logFile, oss.str());
    }

    return {pressureMap, flowRates, balance};
}

} // namespace PressureSolverNewtonGS


