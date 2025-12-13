#pragma once

#include "../vtsim_solver.h"
#include "core/flow_calculation.h"
#include <ceres/ceres.h>
#include <map>
#include <ostream>
#include <vector>

// =============================================================================
// 風量計算のヤコビアン計算ヘルパー関数
// =============================================================================
namespace FlowJacobian {
    // simple_openingのヤコビアン: dQ/dp (解析計算)
    // Q = sign * K * sqrt(abs_dp) の導関数
    // dQ/dp = K * sign * 0.5 / sqrt(abs_dp) * d(abs_dp)/dp
    // d(abs_dp)/dp = sign なので、dQ/dp = K * 0.5 / sqrt(abs_dp) * sign^2 = K * 0.5 / sqrt(abs_dp)
    inline double calcSimpleOpeningJacobian(double dp, const EdgeProperties& edgeData) {
        const double eps = archenv::TOLERANCE_SMALL;
        const double abs_dp = std::abs(dp);
        const double K = edgeData.alpha * edgeData.area * std::sqrt(2.0 / archenv::DENSITY_DRY_AIR);
        
        if (abs_dp >= eps) {
            // sign^2 = 1なので、signを掛ける必要はない
            return 0.5 * K / std::sqrt(abs_dp);
        } else {
            return K * std::sqrt(eps) / eps;
        }
    }
    
    // gapのヤコビアン: dQ/dp (解析計算)
    // Q = sign * a * (abs_dp)^(1/n) の導関数
    // dQ/dp = a * sign * (1/n) * (abs_dp)^(1/n - 1) * d(abs_dp)/dp
    // d(abs_dp)/dp = sign なので、dQ/dp = a * (1/n) * (abs_dp)^(1/n - 1) * sign^2 = a * (1/n) * (abs_dp)^(1/n - 1)
    inline double calcGapJacobian(double dp, const EdgeProperties& edgeData) {
        const double eps = archenv::TOLERANCE_SMALL;
        const double abs_dp = std::abs(dp);
        double n = edgeData.n;
        if (n == 0.0) n = 1.0;
        const double a = edgeData.a;
        
        if (abs_dp >= eps) {
            // sign^2 = 1なので、signを掛ける必要はない
            return a * (1.0 / n) * std::pow(abs_dp, 1.0 / n - 1.0);
        } else {
            return a * std::pow(eps, 1.0 / n - 1.0);
        }
    }
} // namespace FlowJacobian

namespace FanJacobian {
    // fanのヤコビアン: dQ/dp (区分的線形関数として解析計算)
    inline double calcFanJacobian(double dp, const EdgeProperties& edgeData) {
        const double dp_fan = -dp;  // ファンは逆方向の圧力差
        const double p_max = edgeData.p_max;
        const double p1 = edgeData.p1;
        const double q1 = edgeData.q1;
        const double q_max = edgeData.q_max;
        
        // スムージングを使わず、明確な閾値で区分
        if (dp_fan >= p_max) {
            // 領域1: Q = 0
            return 0.0;
        } else if (dp_fan >= p1) {
            // 領域2: Q = q1 * (dp_fan - p_max) / (p1 - p_max)
            // dQ/d(dp_fan) = q1 / (p1 - p_max)
            // dQ/dp = dQ/d(dp_fan) * d(dp_fan)/dp = (q1 / (p1 - p_max)) * (-1)
            if (p1 == p_max) {
                return 0.0;
            }
            return -q1 / (p1 - p_max);
        } else if (dp_fan >= 0.0) {
            // 領域3: Q = q1 + (q_max - q1) * (dp_fan - p1) / (-p1)
            // dQ/d(dp_fan) = (q_max - q1) / (-p1)
            // dQ/dp = (q_max - q1) / (-p1) * (-1) = (q_max - q1) / p1
            if (p1 == 0.0) {
                return 0.0;
            }
            return (q_max - q1) / p1;
        } else {
            // 領域4: Q = q_max
            return 0.0;
        }
    }
} // namespace FanJacobian

// =============================================================================
// 共通ヤコビアン計算ヘルパー関数
// =============================================================================
namespace FlowJacobianCommon {
    // 統一されたヤコビアン計算関数
    // すべての制約クラスで共通使用
    inline double calculateJacobian(double dp, const EdgeProperties& edgeData) {
        if (edgeData.type == "fan") {
            // ファンは解析ヤコビアン
            return FanJacobian::calcFanJacobian(dp, edgeData);
        } else if (edgeData.type == "simple_opening") {
            // simple_openingは解析ヤコビアン
            return FlowJacobian::calcSimpleOpeningJacobian(dp, edgeData);
        } else if (edgeData.type == "gap") {
            // gapは解析ヤコビアン
            return FlowJacobian::calcGapJacobian(dp, edgeData);
        } else if (edgeData.type == "fixed_flow") {
            // fixed_flowは定数なので、ヤコビアンは0
            return 0.0;
        } else {
            // その他は数値微分
            const double eps = 1e-7;
            double q_plus = FlowCalculation::calculateUnifiedFlow(dp + eps, edgeData);
            double q_minus = FlowCalculation::calculateUnifiedFlow(dp - eps, edgeData);
            double dQdp = (q_plus - q_minus) / (2.0 * eps);
            if (!std::isfinite(dQdp)) {
                dQdp = 0.0;
            }
            return dQdp;
        }
    }
} // namespace FlowJacobianCommon

// =============================================================================
// FlowBalanceConstraint - 流量バランス制約（共通ヤコビアン計算関数を使用）
// =============================================================================
class FlowBalanceConstraint : public ceres::CostFunction {
public:
    FlowBalanceConstraint(
        const std::string& nodeName,
        const Graph& graph,
        const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
        const std::map<Vertex, size_t>& vertexToParameterIndex,
        size_t numParameters,
        std::ostream& logFile
    ) : nodeName_(nodeName),
        graph_(graph),
        nodeKeyToVertex_(nodeKeyToVertex),
        vertexToParameterIndex_(vertexToParameterIndex),
        numParameters_(numParameters),
        logFile_(logFile) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters_));
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        auto nodeIt = nodeKeyToVertex_.find(nodeName_);
        if (nodeIt == nodeKeyToVertex_.end()) {
            residuals[0] = 0.0;
            if (jacobians && jacobians[0]) {
                std::fill(jacobians[0], jacobians[0] + static_cast<int>(numParameters_), 0.0);
            }
            return true;
        }

        const double* pressures = parameters[0];
        Vertex nodeVertex = nodeIt->second;
        double inflow = 0.0;
        double outflow = 0.0;
        std::vector<double> grad(numParameters_, 0.0);

        auto edge_range = boost::edges(graph_);
        for (auto edge : boost::make_iterator_range(edge_range)) {
            auto sourceVertex = boost::source(edge, graph_);
            auto targetVertex = boost::target(edge, graph_);
            const auto& edgeData = graph_[edge];

            bool isOutEdge = (sourceVertex == nodeVertex);
            bool isInEdge  = (targetVertex == nodeVertex);
            if (!isOutEdge && !isInEdge) continue;

            auto itS = vertexToParameterIndex_.find(sourceVertex);
            auto itT = vertexToParameterIndex_.find(targetVertex);
            if (itS != vertexToParameterIndex_.end() && itT != vertexToParameterIndex_.end()
                && itS->second == itT->second) {
                continue;
            }

            double sourcePressure = getNodePressure(sourceVertex, pressures);
            double targetPressure = getNodePressure(targetVertex, pressures);

            const auto& sourceNode = graph_[sourceVertex];
            const auto& targetNode = graph_[targetVertex];
            double rho_source = calculateDensity(sourceNode.current_t);
            double rho_target = calculateDensity(targetNode.current_t);

            double source_total_pressure = sourcePressure - rho_source * archenv::GRAVITY * edgeData.h_from;
            double target_total_pressure = targetPressure - rho_target * archenv::GRAVITY * edgeData.h_to;
            double dp = source_total_pressure - target_total_pressure;

            double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);

            // ヤコビアン計算
            if (jacobians && jacobians[0]) {
                double dQdp = FlowJacobianCommon::calculateJacobian(dp, edgeData);
                
                if (isOutEdge) {
                    outflow += flow;
                    if (itS != vertexToParameterIndex_.end()) grad[itS->second] -= dQdp;
                    if (itT != vertexToParameterIndex_.end()) grad[itT->second] += dQdp;
                }
                if (isInEdge) {
                    inflow += flow;
                    if (itS != vertexToParameterIndex_.end()) grad[itS->second] += dQdp;
                    if (itT != vertexToParameterIndex_.end()) grad[itT->second] -= dQdp;
                }
            } else {
                // ヤコビアン不要の場合は残差のみ計算
                if (isOutEdge) {
                    outflow += flow;
                }
                if (isInEdge) {
                    inflow += flow;
                }
            }
        }

        residuals[0] = inflow - outflow;

        if (jacobians && jacobians[0]) {
            std::copy(grad.begin(), grad.end(), jacobians[0]);
        }
        return true;
    }

private:
    std::string nodeName_;
    const Graph& graph_;
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex_;
    const std::map<Vertex, size_t>& vertexToParameterIndex_;
    size_t numParameters_;
    std::ostream& logFile_;

    double getNodePressure(Vertex v, const double* pressures) const {
        auto it = vertexToParameterIndex_.find(v);
        if (it != vertexToParameterIndex_.end()) {
            return pressures[it->second];
        }
        return graph_[v].current_p;
    }
};

// =============================================================================
// GroupFlowBalanceConstraint - グループ流量バランス制約（共通ヤコビアン計算関数を使用）
// =============================================================================
class GroupFlowBalanceConstraint : public ceres::CostFunction {
public:
    GroupFlowBalanceConstraint(
        const std::vector<Vertex>& groupVertices,
        const Graph& graph,
        const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
        const std::map<Vertex, size_t>& vertexToParameterIndex,
        size_t numParameters,
        std::ostream& logFile
    ) : groupVertices_(groupVertices),
        graph_(graph),
        nodeKeyToVertex_(nodeKeyToVertex),
        vertexToParameterIndex_(vertexToParameterIndex),
        numParameters_(numParameters),
        logFile_(logFile) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters_));
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* pressures = parameters[0];
        double inflow = 0.0;
        double outflow = 0.0;
        std::vector<double> grad(numParameters_, 0.0);

        auto edge_range = boost::edges(graph_);
        for (auto edge : boost::make_iterator_range(edge_range)) {
            auto sv = boost::source(edge, graph_);
            auto tv = boost::target(edge, graph_);
            const auto& ep = graph_[edge];

            bool sIn = std::find(groupVertices_.begin(), groupVertices_.end(), sv) != groupVertices_.end();
            bool tIn = std::find(groupVertices_.begin(), groupVertices_.end(), tv) != groupVertices_.end();
            if (sIn == tIn) continue;

            double pS = getNodePressure(sv, pressures);
            double pT = getNodePressure(tv, pressures);

            const auto& sNode = graph_[sv];
            const auto& tNode = graph_[tv];
            double rhoS = calculateDensity(sNode.current_t);
            double rhoT = calculateDensity(tNode.current_t);

            double sTotal = pS - rhoS * archenv::GRAVITY * ep.h_from;
            double tTotal = pT - rhoT * archenv::GRAVITY * ep.h_to;
            double dp = sTotal - tTotal;

            double q = FlowCalculation::calculateUnifiedFlow(dp, ep);

            // ヤコビアン計算
            if (jacobians && jacobians[0]) {
                double dQdp = FlowJacobianCommon::calculateJacobian(dp, ep);
                
                auto itS = vertexToParameterIndex_.find(sv);
                auto itT = vertexToParameterIndex_.find(tv);

                if (sIn) {
                    outflow += q;
                    if (itS != vertexToParameterIndex_.end()) grad[itS->second] -= dQdp;
                    if (itT != vertexToParameterIndex_.end()) grad[itT->second] += dQdp;
                }
                if (tIn) {
                    inflow += q;
                    if (itS != vertexToParameterIndex_.end()) grad[itS->second] += dQdp;
                    if (itT != vertexToParameterIndex_.end()) grad[itT->second] -= dQdp;
                }
            } else {
                // ヤコビアン不要の場合は残差のみ計算
                if (sIn) {
                    outflow += q;
                }
                if (tIn) {
                    inflow += q;
                }
            }
        }

        residuals[0] = inflow - outflow;

        if (jacobians && jacobians[0]) {
            std::copy(grad.begin(), grad.end(), jacobians[0]);
        }
        return true;
    }

private:
    std::vector<Vertex> groupVertices_;
    const Graph& graph_;
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex_;
    const std::map<Vertex, size_t>& vertexToParameterIndex_;
    size_t numParameters_;
    std::ostream& logFile_;

    double getNodePressure(Vertex v, const double* pressures) const {
        auto it = vertexToParameterIndex_.find(v);
        if (it != vertexToParameterIndex_.end()) return pressures[it->second];
        return graph_[v].current_p;
    }
};

// =============================================================================
// SoftAnchorConstraint - アンカー制約（解析ヤコビアン版）
// =============================================================================
class SoftAnchorConstraint : public ceres::CostFunction {
public:
    SoftAnchorConstraint(size_t parameterIndex, double targetPressure, double weight, size_t numParameters)
        : parameterIndex_(parameterIndex), targetPressure_(targetPressure), weight_(weight), numParameters_(numParameters) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters));
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* pressures = parameters[0];
        double p = pressures[parameterIndex_];
        residuals[0] = weight_ * (p - targetPressure_);

        if (jacobians && jacobians[0]) {
            // ヤコビアン: d(residual)/d(p_i)
            // parameterIndex_の位置だけweight_、他は0
            std::fill(jacobians[0], jacobians[0] + static_cast<int>(numParameters_), 0.0);
            jacobians[0][parameterIndex_] = weight_;
        }

        return true;
    }

private:
    size_t parameterIndex_;
    double targetPressure_;
    double weight_;
    size_t numParameters_;
};
