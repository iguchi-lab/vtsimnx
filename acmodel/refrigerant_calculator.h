#pragma once

namespace acmodel {

/**
 * @brief 冷媒の熱力学的性質を計算するユーティリティクラス
 */
class RefrigerantCalculator {
public:
    /**
     * @brief 飽和蒸気の温度から圧力を求める関数
     * @param theta 温度 [°C]
     * @return 圧力 [Pa]
     */
    static double getSaturatedGasPressure(double theta);

    /**
     * @brief 圧縮機吸引領域において過熱蒸気の圧力と温度から比エンタルピーを求める関数
     * @param P 圧力 [Pa]
     * @param theta 温度 [°C]
     * @return 比エンタルピー [kJ/kg]
     */
    static double getGasCompressorInletEnthalpy(double P, double theta);

    /**
     * @brief 圧縮機吐出領域において過熱蒸気の圧力と比エントロピーから比エンタルピーを求める関数
     * @param P 圧力 [Pa]
     * @param S 比エントロピー [kJ/kg・K]
     * @return 比エンタルピー [kJ/kg]
     */
    static double getGasCompressorOutletEnthalpy(double P, double S);

    /**
     * @brief 過熱蒸気の圧力と比エンタルピーから比エントロピーを求める関数
     * @param P 圧力 [Pa]
     * @param h 比エンタルピー [kJ/kg]
     * @return 比エントロピー [kJ/kg・K]
     */
    static double getGasEntropy(double P, double h);

    /**
     * @brief 過冷却液の圧力と温度から比エンタルピーを求める関数
     * @param P 圧力 [Pa]
     * @param theta 温度 [°C]
     * @return 比エンタルピー [kJ/kg]
     */
    static double getLiquidEnthalpy(double P, double theta);

    /**
     * @brief ヒートポンプサイクルの理論暖房効率を計算する
     * @param theta_ref_evp 冷媒の蒸発温度 [°C]
     * @param theta_ref_cnd 冷媒の凝縮温度 [°C]
     * @param theta_ref_SC 冷媒の過冷却度 [°C]
     * @param theta_ref_SH 冷媒の過熱度 [°C]
     * @return 理論暖房効率 [-]
     */
    static double calculateTheoreticalHeatingEfficiency(double theta_ref_evp, double theta_ref_cnd, 
                                                        double theta_ref_SC, double theta_ref_SH);


};

} // namespace acmodel 