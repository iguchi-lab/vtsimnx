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

// The same boundary-flow evaluation serves dense compatibility callers and the
// production scalar blocks. Only incident unknown pressures enter scalar blocks.
class FlowBalanceConstraintImpl : public ceres::CostFunction {
public:
    FlowBalanceConstraintImpl(
        const std::vector<Vertex>& vertices, bool singleNode, const Graph& graph,
        const std::vector<int>& vertexToParameterIndexVec,
        const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
        const std::vector<double>* densityByVertex, size_t numParameters,
        bool scalarBlocks)
        : graph_(graph), densityByVertex_(densityByVertex),
          numParameters_(numParameters), scalarBlocks_(scalarBlocks) {
        const std::unordered_set<Vertex> members(vertices.begin(), vertices.end());
        for (auto v : vertices) {
            for (auto edge : incidentEdgesByVertex[static_cast<size_t>(v)]) {
                const auto s = boost::source(edge, graph);
                const auto t = boost::target(edge, graph);
                const bool out = members.count(s) != 0;
                const bool in = members.count(t) != 0;
                if (!out && !in) continue;
                if (!singleNode && out == in) continue;
                int si = vertexToParameterIndexVec[static_cast<size_t>(s)];
                int ti = vertexToParameterIndexVec[static_cast<size_t>(t)];
                // Preserve the node residual's supernode/self-loop convention.
                if (singleNode && si >= 0 && si == ti) continue;
                const auto& ep = graph[edge];
                const auto law = FlowCalculation::resolveFlowLaw(ep);
                if (law == FlowCalculation::FlowLaw::Disabled ||
                    law == FlowCalculation::FlowLaw::FixedVolume) {
                    si = ti = -1;
                }
                terms_.push_back({edge, s, t, si, ti, out, in, law});
                if (si >= 0) parameterIndices_.push_back(si);
                if (ti >= 0) parameterIndices_.push_back(ti);
            }
        }
        std::sort(parameterIndices_.begin(), parameterIndices_.end());
        parameterIndices_.erase(std::unique(parameterIndices_.begin(), parameterIndices_.end()),
                                parameterIndices_.end());
        set_num_residuals(1);
        if (scalarBlocks_) {
            mutable_parameter_block_sizes()->assign(parameterIndices_.size(), 1);
            auto localIndex = [&](int index) {
                return index < 0 ? -1 : static_cast<int>(std::lower_bound(
                    parameterIndices_.begin(), parameterIndices_.end(), index) - parameterIndices_.begin());
            };
            for (auto& term : terms_) {
                term.sourceIndex = localIndex(term.sourceIndex);
                term.targetIndex = localIndex(term.targetIndex);
            }
        } else {
            mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters_));
        }
    }

    // Problem owns the cost function after this call. Pressures must not resize
    // and the graph's topology / enabled / fixed-flow state must stay unchanged
    // during this solve. A new problem is built for each pressure solve/stage.
    void addToProblem(ceres::Problem& problem, const std::vector<double>& pressures) {
        std::vector<double*> blocks;
        blocks.reserve(parameterIndices_.size());
        for (int index : parameterIndices_) {
            blocks.push_back(const_cast<double*>(&pressures.at(static_cast<size_t>(index))));
        }
        problem.AddResidualBlock(this, nullptr, blocks);
    }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
        if (jacobians != nullptr) {
            if (scalarBlocks_) {
                for (size_t i = 0; i < parameterIndices_.size(); ++i) {
                    if (jacobians[i] != nullptr) jacobians[i][0] = 0.0;
                }
            } else if (jacobians[0] != nullptr) {
                std::fill(jacobians[0], jacobians[0] + numParameters_, 0.0);
            }
        }
        auto pressure = [&](Vertex v, int index) {
            if (index < 0) return graph_[v].current_p;
            return scalarBlocks_ ? parameters[index][0] : parameters[0][index];
        };
        auto addJacobian = [&](int index, double value) {
            if (index < 0 || jacobians == nullptr) return;
            if (scalarBlocks_) {
                if (jacobians[index] != nullptr) jacobians[index][0] += value;
            } else if (jacobians[0] != nullptr) {
                jacobians[0][index] += value;
            }
        };
        double inflow = 0.0;
        double outflow = 0.0;
        for (const auto& term : terms_) {
            const auto& ep = graph_[term.edge];
            const double sTotal = pressure(term.source, term.sourceIndex)
                - densityAt(term.source) * archenv::GRAVITY * ep.h_from;
            const double tTotal = pressure(term.target, term.targetIndex)
                - densityAt(term.target) * archenv::GRAVITY * ep.h_to;
            const double dp = sTotal - tTotal;
            const double q = FlowCalculation::calculateUnifiedFlow(dp, ep, term.law);
            if (term.out) outflow += q;
            if (term.in) inflow += q;
            if (jacobians != nullptr) {
                const double derivative = FlowJacobianCommon::calculateJacobian(dp, ep, term.law);
                if (term.out) {
                    addJacobian(term.sourceIndex, -derivative);
                    addJacobian(term.targetIndex, derivative);
                }
                if (term.in) {
                    addJacobian(term.sourceIndex, derivative);
                    addJacobian(term.targetIndex, -derivative);
                }
            }
        }
        residuals[0] = inflow - outflow;
        return true;
    }

private:
    struct Term {
        Edge edge;
        Vertex source, target;
        int sourceIndex, targetIndex;
        bool out, in;
        FlowCalculation::FlowLaw law;
    };
    const Graph& graph_;
    const std::vector<double>* densityByVertex_;
    size_t numParameters_;
    bool scalarBlocks_;
    std::vector<Term> terms_;
    std::vector<int> parameterIndices_;

    double densityAt(Vertex v) const {
        return densityByVertex_ != nullptr ? (*densityByVertex_)[static_cast<size_t>(v)]
                                          : calculateDensity(graph_[v].current_t);
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

// 存在しないノード名向け（残差・ヤコビアンともゼロ）
class ZeroResidualConstraintImpl : public ceres::CostFunction {
public:
    explicit ZeroResidualConstraintImpl(size_t numParameters)
        : numParameters_(numParameters) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters));
    }

    bool Evaluate(double const* const* /*parameters*/,
                  double* residuals,
                  double** jacobians) const override {
        residuals[0] = 0.0;
        if (jacobians && jacobians[0]) {
            std::fill(jacobians[0], jacobians[0] + static_cast<int>(numParameters_), 0.0);
        }
        return true;
    }

private:
    size_t numParameters_;
};

} // namespace

namespace PressureConstraints {

ceres::CostFunction* createFlowBalanceConstraint(
    Vertex nodeVertex,
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex,
    size_t numParameters,
    std::ostream& logFile) {
    (void)logFile;
    return new FlowBalanceConstraintImpl({nodeVertex}, true, graph,
        vertexToParameterIndexVec, incidentEdgesByVertex, densityByVertex,
        numParameters, false);
}

ceres::CostFunction* createFlowBalanceConstraint(
    const std::string& nodeName,
    const Graph& graph,
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    size_t numParameters,
    std::ostream& logFile) {
    auto it = nodeKeyToVertex.find(nodeName);
    if (it == nodeKeyToVertex.end()) {
        return new ZeroResidualConstraintImpl(numParameters);
    }
    return createFlowBalanceConstraint(it->second,
                                       graph,
                                       vertexToParameterIndexVec,
                                       incidentEdgesByVertex,
                                       /*densityByVertex=*/nullptr,
                                       numParameters,
                                       logFile);
}

ceres::CostFunction* createGroupFlowBalanceConstraint(
    const std::vector<Vertex>& groupVertices,
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex,
    size_t numParameters,
    std::ostream& logFile) {
    (void)logFile;
    return new FlowBalanceConstraintImpl(groupVertices, false, graph,
        vertexToParameterIndexVec, incidentEdgesByVertex, densityByVertex,
        numParameters, false);
}

ceres::CostFunction* createGroupFlowBalanceConstraint(
    const std::vector<Vertex>& groupVertices,
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    size_t numParameters,
    std::ostream& logFile) {
    return createGroupFlowBalanceConstraint(groupVertices,
                                            graph,
                                            vertexToParameterIndexVec,
                                            incidentEdgesByVertex,
                                            /*densityByVertex=*/nullptr,
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

void addFlowBalanceConstraint(
    ceres::Problem& problem, Vertex nodeVertex, const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex, const std::vector<double>& pressures) {
    auto* cost = new FlowBalanceConstraintImpl({nodeVertex}, true, graph,
        vertexToParameterIndexVec, incidentEdgesByVertex, densityByVertex, pressures.size(), true);
    cost->addToProblem(problem, pressures);
}

void addGroupFlowBalanceConstraint(
    ceres::Problem& problem, const std::vector<Vertex>& groupVertices, const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex, const std::vector<double>& pressures) {
    auto* cost = new FlowBalanceConstraintImpl(groupVertices, false, graph,
        vertexToParameterIndexVec, incidentEdgesByVertex, densityByVertex, pressures.size(), true);
    cost->addToProblem(problem, pressures);
}

} // namespace PressureConstraints
