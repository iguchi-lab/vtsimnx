#pragma once

#include <vector>
#include <string>

#include "../archenv/include/archenv.h"

namespace acmodel {

/**
 * @brief エアコン計算で使用する定数を管理するクラス
 */
class Constants {
public:
    // 基本定数
    static constexpr double ERROR_THRESHOLD = 1e-3;  // 温度の許容誤差
    static constexpr double MAX_TEMP = 50.0;          // 最大温度探索範囲
    
    // 物理定数
    // NOTE: 物性値は solver 全体で archenv に統一する（単位系のブレを避ける）
    static constexpr double C_P_AIR = archenv::SPECIFIC_HEAT_AIR;          // 空気の比熱 [J/(kg·K)]
    static constexpr double RHO_AIR = archenv::DENSITY_DRY_AIR;            // 空気の密度 [kg/m³]
    // C_P_W / L_WTR はモデル内部の式（kJ 系）との整合があるため現状維持（単位に注意）
    static constexpr double C_P_W = 1.846;                                 // 水蒸気の定圧比熱 [kJ/(kg·K)] (legacy)
    static constexpr double F = archenv::STANDARD_ATMOSPHERIC_PRESSURE;    // 大気圧 [Pa]
    
    // 水の蒸発潜熱 [kJ/kg]
    static constexpr double L_WTR = 2500.8 - 2.3668 * 27;
    
    // 熱交換器関連
    static constexpr double A_F_HEX = 0.23559;        // 室内機熱交換器の前面面積のうち熱交換に有効な面積 [m²]
    static constexpr double A_E_HEX = 6.396;          // 室内機熱交換器の表面積のうち熱交換に有効な面積 [m²]
    
    // CRIEPIモデル用定数
    static constexpr double BF = 0.2;                 // バイパスファクター
    
    // RACモデル用定数
    static constexpr double C_AF_H = 0.8;
    static constexpr double C_AF_C = 0.85;
    static constexpr double C_HM_C = 1.15;
    static constexpr double SHF_L_MIN_C = 0.4;
    
    // 潜熱評価モデル用定数
    static constexpr double A_F_HEX_SMALL_H = 0.2;    // 定格冷却能力が5.6kW未満の場合
    static constexpr double A_E_HEX_SMALL_H = 6.2;
    static constexpr double A_F_HEX_LARGE_H = 0.3;    // 定格冷却能力が5.6kW以上の場合
    static constexpr double A_E_HEX_LARGE_H = 10.6;
    
    // モードリスト
    static const std::vector<std::string> MODES;
    static const std::vector<std::string> KEYS_CRIEPI;
};

/**
 * @brief RACモデル用の係数テーブル
 */
class RACTables {
public:
    // 表3: 容量可変型でないコンプレッサー用係数
    static const std::vector<std::vector<double>> TABLE_3;
    
    // 表4: 容量可変型コンプレッサー用係数
    static const std::vector<std::vector<double>> TABLE_4_A;
    static const std::vector<std::vector<double>> TABLE_4_B;
    static const std::vector<std::vector<double>> TABLE_4_C;
    
    // 表5: 冷房用係数
    static const std::vector<std::vector<double>> TABLE_5;
    
    // 表6: 冷房用容量可変型係数
    static const std::vector<std::vector<double>> TABLE_6_A;
    static const std::vector<std::vector<double>> TABLE_6_B;
    static const std::vector<std::vector<double>> TABLE_6_C;
};

/**
 * @brief DuctCentralモデル用のテーブル
 */
class DuctCentralTables {
public:
    // 表1: 単位面積当たりの必要暖房能力及び冷房能力 [W/m²]
    static const std::vector<std::vector<double>> TABLE_1;
};

} // namespace acmodel 