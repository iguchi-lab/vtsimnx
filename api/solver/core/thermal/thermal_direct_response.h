#pragma once

#include "../../vtsim_solver.h"
#include <algorithm>

// DirectT（thermal_solver_linear_direct.cpp）で使う response_conduction の評価ヘルパ。
// 依存を増やさないため header-only で提供する。

namespace thermal_direct_response {

inline double responseArea(const EdgeProperties& ep) {
    return (ep.area > 0.0) ? ep.area : 1.0;
}

inline double evalResponseTempHistoryTermSrc(const EdgeProperties& ep) {
    double s = 0.0;
    for (size_t k = 1; k < ep.resp_a_src.size(); ++k)
        if (k - 1 < ep.hist_t_src.size()) s += ep.resp_a_src[k] * ep.hist_t_src[k - 1];
    for (size_t k = 1; k < ep.resp_b_src.size(); ++k)
        if (k - 1 < ep.hist_t_tgt.size()) s += ep.resp_b_src[k] * ep.hist_t_tgt[k - 1];
    return s;
}

inline double evalResponseTempHistoryTermTgt(const EdgeProperties& ep) {
    double s = 0.0;
    for (size_t k = 1; k < ep.resp_a_tgt.size(); ++k)
        if (k - 1 < ep.hist_t_tgt.size()) s += ep.resp_a_tgt[k] * ep.hist_t_tgt[k - 1];
    for (size_t k = 1; k < ep.resp_b_tgt.size(); ++k)
        if (k - 1 < ep.hist_t_src.size()) s += ep.resp_b_tgt[k] * ep.hist_t_src[k - 1];
    return s;
}

inline double evalResponseHistoryWattSrc(const EdgeProperties& ep) {
    double hW = responseArea(ep) * evalResponseTempHistoryTermSrc(ep);
    for (size_t k = 0; k < ep.resp_c_src.size(); ++k)
        if (k < ep.hist_q_src.size()) hW += ep.resp_c_src[k] * ep.hist_q_src[k];
    return hW;
}

inline double evalResponseHistoryWattTgt(const EdgeProperties& ep) {
    double hW = responseArea(ep) * evalResponseTempHistoryTermTgt(ep);
    for (size_t k = 0; k < ep.resp_c_tgt.size(); ++k)
        if (k < ep.hist_q_tgt.size()) hW += ep.resp_c_tgt[k] * ep.hist_q_tgt[k];
    return hW;
}

inline double evalResponseQSrc(const EdgeProperties& ep, double Ts, double Tt) {
    const double a0 = ep.resp_a_src.empty() ? 0.0 : ep.resp_a_src[0];
    const double b0 = ep.resp_b_src.empty() ? 0.0 : ep.resp_b_src[0];
    double q = responseArea(ep) * (a0 * Ts + b0 * Tt + evalResponseTempHistoryTermSrc(ep));
    for (size_t k = 0; k < ep.resp_c_src.size(); ++k)
        if (k < ep.hist_q_src.size()) q += ep.resp_c_src[k] * ep.hist_q_src[k];
    return q;
}

inline double evalResponseQTgt(const EdgeProperties& ep, double Ts, double Tt) {
    const double a0 = ep.resp_a_tgt.empty() ? 0.0 : ep.resp_a_tgt[0];
    const double b0 = ep.resp_b_tgt.empty() ? 0.0 : ep.resp_b_tgt[0];
    double q = responseArea(ep) * (a0 * Tt + b0 * Ts + evalResponseTempHistoryTermTgt(ep));
    for (size_t k = 0; k < ep.resp_c_tgt.size(); ++k)
        if (k < ep.hist_q_tgt.size()) q += ep.resp_c_tgt[k] * ep.hist_q_tgt[k];
    return q;
}

} // namespace thermal_direct_response


