#pragma once

#include "../../vtsim_solver.h"

#include <string>
#include <utility>
#include <vector>

namespace ventilation {

// fallback などで一時的に edge を fixed_flow 化した状態を、例外時も含め必ず復元する。
class EdgeMutationGuard {
public:
    explicit EdgeMutationGuard(Graph& graph) : graph_(graph) {}

    EdgeMutationGuard(const EdgeMutationGuard&) = delete;
    EdgeMutationGuard& operator=(const EdgeMutationGuard&) = delete;

    ~EdgeMutationGuard() { restore(); }

    void convertToFixedFlow(Edge edge, double flow) {
        auto& ep = graph_[edge];
        // 既にこのガードが変更済みなら上書きのみ（元状態は最初のものを保持）
        for (const auto& st : originals_) {
            if (st.edge == edge) {
                ep.current_vol = flow;
                ep.type = "fixed_flow";
                return;
            }
        }
        OriginalState st;
        st.edge = edge;
        st.type = ep.type;
        st.currentVol = ep.current_vol;
        originals_.push_back(std::move(st));
        ep.current_vol = flow;
        ep.type = "fixed_flow";
    }

    void restore() {
        for (auto it = originals_.rbegin(); it != originals_.rend(); ++it) {
            auto& ep = graph_[it->edge];
            ep.type = it->type;
            ep.current_vol = it->currentVol;
        }
        originals_.clear();
    }

    bool empty() const { return originals_.empty(); }
    std::size_t size() const { return originals_.size(); }

private:
    struct OriginalState {
        Edge edge{};
        std::string type;
        double currentVol = 0.0;
    };

    Graph& graph_;
    std::vector<OriginalState> originals_;
};

} // namespace ventilation
