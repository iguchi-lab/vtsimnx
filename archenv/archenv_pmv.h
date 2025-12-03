#pragma once

#include "archenv.h"

namespace archenv {
namespace internal {
    double calc_R(double f_cl, double t_cl, double t_r);
    double calc_C(double f_cl, double h_c, double t_cl, double t_a);
    double calc_RC(double f_cl, double h_c, double t_cl, double t_a, double t_r);
}

// PMV計算用定数
constexpr double PMV_OMEGA_DEFAULT = 0.5;
constexpr double PMV_TOLERANCE = 1e-6;
constexpr int PMV_MAX_ITERATIONS = 100;
}

