#include "include/archenv.h"
#include <algorithm>
#include <cmath>

namespace archenv {

double calc_fungal_index(double relative_humidity_percent, double t) {
    // 湿度[%] → 0-1に正規化
    double h = std::clamp(relative_humidity_percent / 100.0, 0.0, 1.0);

    // Python版と同じパラメータ
    const double a = -0.3;
    const double b = 0.685;
    const double c1 = 0.95;
    const double c2 = 0.07;
    const double c3 = 25.0;
    const double c4 = 7.2;
    
    // 正規化された変数
    double x = (h - c1) / c2;
    double y = (t - c3) / c4;
    
    // 真菌指数計算
    // FI = 187.25 * exp((((x² - 2*a*x*y + y²)^b) / (2*(a² - 2)))
    double numerator = std::pow(x*x - 2*a*x*y + y*y, b);
    double denominator = 2 * (a*a - 2);
    
    double FI = 187.25 * std::exp(numerator / denominator) - 8.25;
    
    return FI;
}

} // namespace archenv 