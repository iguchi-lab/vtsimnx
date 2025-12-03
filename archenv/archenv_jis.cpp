#include "archenv_jis.h"
#include <sstream>
#include <iomanip>

namespace archenv {
namespace jis {

std::string validate_jis_conditions() {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    oss << "JIS規格条件検証結果:\n";
    oss << "==================\n\n";
    
    // 冷房時条件の検証
    oss << "【冷房時条件】\n";
    
    double x_c_in_calc = calc_x_from_wet_bulb(T_WB_C_IN, T_C_IN);
    double x_c_ex_calc = calc_x_from_wet_bulb(T_WB_C_EX, T_C_EX);
    
    oss << "室内: T=" << T_C_IN << "℃, T_wb=" << T_WB_C_IN << "℃\n";
    oss << "  定数値 X_C_IN = " << X_C_IN << " kg/kg'\n";
    oss << "  計算値        = " << x_c_in_calc << " kg/kg'\n";
    oss << "  差異          = " << std::abs(X_C_IN - x_c_in_calc) << "\n\n";
    
    oss << "外気: T=" << T_C_EX << "℃, T_wb=" << T_WB_C_EX << "℃\n";
    oss << "  定数値 X_C_EX = " << X_C_EX << " kg/kg'\n";
    oss << "  計算値        = " << x_c_ex_calc << " kg/kg'\n";
    oss << "  差異          = " << std::abs(X_C_EX - x_c_ex_calc) << "\n\n";
    
    // 暖房時条件の検証
    oss << "【暖房時条件】\n";
    
    double x_h_in_calc = calc_x_from_wet_bulb(T_WB_H_IN, T_H_IN);
    double x_h_ex_calc = calc_x_from_wet_bulb(T_WB_H_EX, T_H_EX);
    
    oss << "室内: T=" << T_H_IN << "℃, T_wb=" << T_WB_H_IN << "℃\n";
    oss << "  定数値 X_H_IN = " << X_H_IN << " kg/kg'\n";
    oss << "  計算値        = " << x_h_in_calc << " kg/kg'\n";
    oss << "  差異          = " << std::abs(X_H_IN - x_h_in_calc) << "\n\n";
    
    oss << "外気: T=" << T_H_EX << "℃, T_wb=" << T_WB_H_EX << "℃\n";
    oss << "  定数値 X_H_EX = " << X_H_EX << " kg/kg'\n";
    oss << "  計算値        = " << x_h_ex_calc << " kg/kg'\n";
    oss << "  差異          = " << std::abs(X_H_EX - x_h_ex_calc) << "\n\n";
    
    // 検証結果の判定
    const double tolerance = 1e-5;
    bool valid = (std::abs(X_C_IN - x_c_in_calc) < tolerance) &&
                 (std::abs(X_C_EX - x_c_ex_calc) < tolerance) &&
                 (std::abs(X_H_IN - x_h_in_calc) < tolerance) &&
                 (std::abs(X_H_EX - x_h_ex_calc) < tolerance);
    
    oss << "【総合判定】\n";
    oss << "JIS規格条件の整合性: " << (valid ? "OK" : "NG") << "\n";
    oss << "許容誤差: " << tolerance << " kg/kg'\n";
    
    return oss.str();
}

} // namespace jis
} // namespace archenv 