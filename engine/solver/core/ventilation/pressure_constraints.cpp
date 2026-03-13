#include "core/ventilation/pressure_constraints.h"
#include "core/ventilation/flow_calculation.h"
#include "core/ventilation/flow_jacobian.h"

#include <ceres/ceres.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

// 密度計算は pressure_solver.cpp にある free 関数を利用
double calculateDensity(double temperature);

namespace {

// =============================================================================
// FlowBalanceConstraint - 流量バランス制約（共通ヤコビアン計算関数を使用）
// =============================================================================
class FlowBalanceConstraintImpl : public ceres::CostFunction {
public:
    FlowBalanceConstraintImpl(
        const std::string& nodeName,
        const Graph& graph,
        const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
        const std::vector<int>& vertexToParameterIndexVec,
        const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
        size_t numParameters,
        std::ostream& logFile)
        : nodeName_(nodeName),
          graph_(graph),
          nodeKeyToVertex_(nodeKeyToVertex),
          vertexToParameterIndexVec_(vertexToParameterIndexVec),
          incidentEdgesByVertex_(incidentEdgesByVertex),
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

        const auto& inc = incidentEdgesByVertex_[static_cast<size_t>(nodeVertex)];
        for (auto edge : inc) {
            auto sourceVertex = boost::source(edge, graph_);
            auto targetVertex = boost::target(edge, graph_);
            const auto& edgeData = graph_[edge];

            bool isOutEdge = (sourceVertex == nodeVertex);
            bool isInEdge  = (targetVertex == nodeVertex);
            if (!isOutEdge && !isInEdge) continue;

            const int sIdx = vertexToParameterIndexVec_[static_cast<size_t>(sourceVertex)];
            const int tIdx = vertexToParameterIndexVec_[static_cast<size_t>(targetVertex)];
            if (sIdx >= 0 && tIdx >= 0 && sIdx == tIdx) {
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
                    if (sIdx >= 0) grad[static_cast<size_t>(sIdx)] -= dQdp;
                    if (tIdx >= 0) grad[static_cast<size_t>(tIdx)] += dQdp;
                }
                if (isInEdge) {
                    inflow += flow;
                    if (sIdx >= 0) grad[static_cast<size_t>(sIdx)] += dQdp;
                    if (tIdx >= 0) grad[static_cast<size_t>(tIdx)] -= dQdp;
                }
            } else {
                if (isOutEdge) outflow += flow;
                if (isInEdge) inflow += flow;
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
    const std::vector<int>& vertexToParameterIndexVec_;
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex_;
    size_t numParameters_;
    std::ostream& logFile_;

    double getNodePressure(Vertex v, const double* pressures) const {
        const int idx = vertexToParameterIndexVec_[static_cast<size_t>(v)];
        if (idx >= 0) return pressures[idx];
        return graph_[v].current_p;
    }
};

// =============================================================================
// GroupFlowBalanceConstraint - グループ流量バランス制約（共通ヤコビアン計算関数を使用）
// =============================================================================
class GroupFlowBalanceConstraintImpl : public ceres::CostFunction {
public:
    GroupFlowBalanceConstraintImpl(
        const std::vector<Vertex>& groupVertices,
        const Graph& graph,
        const std::vector<int>& vertexToParameterIndexVec,
        const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
        size_t numParameters,
        std::ostream& logFile)
        : groupVertices_(groupVertices),
          graph_(graph),
          vertexToParameterIndexVec_(vertexToParameterIndexVec),
          incidentEdgesByVertex_(incidentEdgesByVertex),
          numParameters_(numParameters),
          logFile_(logFile) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters_));
        groupSet_.reserve(groupVertices_.size() * 2 + 1);
        for (auto v : groupVertices_) groupSet_.insert(v);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* pressures = parameters[0];
        double inflow = 0.0;
        double outflow = 0.0;
        std::vector<double> grad(numParameters_, 0.0);

        // group 内頂点の incident edges だけ走査（cross edge は必ず 1 回だけ現れる）
        for (auto vIn : groupVertices_) {
            const auto& inc = incidentEdgesByVertex_[static_cast<size_t>(vIn)];
            for (auto edge : inc) {
                auto sv = boost::source(edge, graph_);
                auto tv = boost::target(edge, graph_);
                const auto& ep = graph_[edge];

                const bool sIn = (groupSet_.find(sv) != groupSet_.end());
                const bool tIn = (groupSet_.find(tv) != groupSet_.end());
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

                if (jacobians && jacobians[0]) {
                    double dQdp = FlowJacobianCommon::calculateJacobian(dp, ep);

                    const int sIdx = vertexToParameterIndexVec_[static_cast<size_t>(sv)];
                    const int tIdx = vertexToParameterIndexVec_[static_cast<size_t>(tv)];

                    if (sIn) {
                        outflow += q;
                        if (sIdx >= 0) grad[static_cast<size_t>(sIdx)] -= dQdp;
                        if (tIdx >= 0) grad[static_cast<size_t>(tIdx)] += dQdp;
                    }
                    if (tIn) {
                        inflow += q;
                        if (sIdx >= 0) grad[static_cast<size_t>(sIdx)] += dQdp;
                        if (tIdx >= 0) grad[static_cast<size_t>(tIdx)] -= dQdp;
                    }
                } else {
                    if (sIn) outflow += q;
                    if (tIn) inflow += q;
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
    const std::vector<int>& vertexToParameterIndexVec_;
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex_;
    size_t numParameters_;
    std::ostream& logFile_;
    std::unordered_set<Vertex> groupSet_;

    double getNodePressure(Vertex v, const double* pressures) const {
        const int idx = vertexToParameterIndexVec_[static_cast<size_t>(v)];
        if (idx >= 0) return pressures[idx];
        return graph_[v].current_p;
    }
};

// =============================================================================
// SoftAnchorConstraint - アンカー制約（解析ヤコビアン版）
// =============================================================================
class SoftAnchorConstraintImpl : public ceres::CostFunction {
public:
    SoftAnchorConstraintImpl(size_t parameterIndex, double targetPressure, double weight, size_t numParameters)
        : parameterIndex_(parameterIndex),
          targetPressure_(targetPressure),
          weight_(weight),
          numParameters_(numParameters) {
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

} // namespace

namespace PressureConstraints {

ceres::CostFunction* createFlowBalanceConstraint(
    const std::string& nodeName,
    const Graph& graph,
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    size_t numParameters,
    std::ostream& logFile) {
    return new FlowBalanceConstraintImpl(nodeName,
                                        graph,
                                        nodeKeyToVertex,
                                        vertexToParameterIndexVec,
                                        incidentEdgesByVertex,
                                        numParameters,
                                        logFile);
}

ceres::CostFunction* createGroupFlowBalanceConstraint(
    const std::vector<Vertex>& groupVertices,
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    size_t numParameters,
    std::ostream& logFile) {
    return new GroupFlowBalanceConstraintImpl(groupVertices,
                                             graph,
                                             vertexToParameterIndexVec,
                                             incidentEdgesByVertex,
                                             numParameters,
                                             logFile);
}

ceres::CostFunction* createSoftAnchorConstraint(
    size_t parameterIndex,
    double targetPressure,
    double weight,
    size_t numParameters) {
    return new SoftAnchorConstraintImpl(parameterIndex, targetPressure, weight, numParameters);
}

} // namespace PressureConstraints


