#include "refrigerant_calculator.h"
#include <cmath>

namespace acmodel {

double RefrigerantCalculator::getSaturatedGasPressure(double theta) {
    return 2.75857926950901e-17 * std::pow(theta, 8) +
           1.49382057911753e-15 * std::pow(theta, 7) +
           6.52001687267015e-14 * std::pow(theta, 6) +
           9.14153034999975e-12 * std::pow(theta, 5) +
           3.18314616500361e-9 * std::pow(theta, 4) +
           1.60703566663019e-6 * std::pow(theta, 3) +
           3.06278984019513e-4 * std::pow(theta, 2) +
           2.54461992992037e-2 * theta +
           7.98086455154775e-1;
}

double RefrigerantCalculator::getGasCompressorInletEnthalpy(double P, double theta) {
    double K = theta + 273.15;
    double K2 = K * K;
    double K3 = K2 * K;
    double P2 = P * P;
    double P3 = P2 * P;

    return -1.00110355e-1 * P3 +
           -1.184450639e1 * P2 +
           -2.052740252e2 * P +
           3.20391e-6 * K3 +
           -2.24685e-3 * K2 +
           1.279436909 * K +
           3.1271238e-2 * P2 * K +
           -1.415359e-3 * P * K2 +
           1.05553912 * P * K +
           1.949505039e2;
}

double RefrigerantCalculator::getGasCompressorOutletEnthalpy(double P, double S) {
    double P2 = P * P;
    double P3 = P2 * P;
    double P4 = P2 * P2;
    double S2 = S * S;
    double S3 = S2 * S;
    double S4 = S2 * S2;

    return -1.869892835947070e-1 * P4 +
           8.223224182177200e-1 * P3 +
           4.124595239531860 * P2 +
           -8.346302788803210e1 * P +
           -1.016388214044490e2 * S4 +
           8.652428629143880e2 * S3 +
           -2.574830800631310e3 * S2 +
           3.462049327009730e3 * S +
           9.209837906396910e-1 * P3 * S +
           -5.163305566700450e-1 * P2 * S2 +
           4.076727767130210 * P * S3 +
           -8.967168786520070 * P2 * S +
           -2.062021416757910e1 * P * S2 +
           9.510257675728610e1 * P * S +
           -1.476914346214130e3;
}

double RefrigerantCalculator::getGasEntropy(double P, double h) {
    double P2 = P * P;
    double P3 = P2 * P;
    double P4 = P2 * P2;
    double h2 = h * h;
    double h3 = h2 * h;
    double h4 = h2 * h2;

    return 5.823109493752840e-2 * P4 +
           -3.309666523931270e-1 * P3 +
           7.700179914440890e-1 * P2 +
           -1.311726004718660 * P +
           1.521486605815750e-9 * h4 +
           -2.703698863404160e-6 * h3 +
           1.793443775071770e-3 * h2 +
           -5.227303746767450e-1 * h +
           1.100368875131490e-4 * P3 * h +
           5.076769807083600e-7 * P2 * h2 +
           1.202580329499520e-8 * P * h3 +
           -7.278049214744230e-4 * P2 * h +
           -1.449198550965620e-5 * P * h2 +
           5.716086851760640e-3 * P * h +
           5.818448621582900e1;
}

double RefrigerantCalculator::getLiquidEnthalpy(double P, double theta) {
    double K = theta + 273.15;
    double K2 = K * K;
    double K3 = K2 * K;
    double P2 = P * P;
    double P3 = P2 * P;

    return 1.7902915e-2 * P3 +
           7.96830322e-1 * P2 +
           5.985874958e1 * P +
           0.0 * K3 +
           9.86677e-4 * K2 +
           9.8051677e-1 * K +
           -3.58645e-3 * P2 * K +
           8.23122e-4 * P * K2 +
           -4.42639115e-1 * P * K +
           -1.415490404e2;
}

double RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
    double theta_ref_evp, double theta_ref_cnd, double theta_ref_SC, double theta_ref_SH) {
    
    double P_ref_evp = getSaturatedGasPressure(theta_ref_evp);              // 蒸発圧力
    double P_ref_cnd = getSaturatedGasPressure(theta_ref_cnd);              // 凝縮圧力
    double theta_ref_cnd_out = theta_ref_cnd - theta_ref_SC;                // 凝縮器出力温度
    double h_ref_cnd_out = getLiquidEnthalpy(P_ref_cnd, theta_ref_cnd_out); // 凝縮器出口比エンタルピー
    double theta_ref_comp_in = theta_ref_evp + theta_ref_SH;                // 圧縮機吸込温度
    double P_ref_comp_in = P_ref_evp;                                       // 圧縮機吸込圧力
    double h_ref_comp_in = getGasCompressorInletEnthalpy(P_ref_comp_in, theta_ref_comp_in); // 圧縮機吸込エンタルピー
    double S_ref_comp_in = getGasEntropy(P_ref_comp_in, h_ref_comp_in);     // 圧縮機吸込比エントロピー
    double S_ref_comp_out = S_ref_comp_in;                                  // 圧縮機吐出比エントロピー
    double P_ref_comp_out = P_ref_cnd;                                      // 圧縮機吐出圧力
    double h_ref_comp_out = getGasCompressorOutletEnthalpy(P_ref_comp_out, S_ref_comp_out); // 圧縮機吐出比エンタルピー
    
    // ヒートポンプサイクルの理論暖房効率
    double e_ref_H_th = (h_ref_comp_out - h_ref_cnd_out) / (h_ref_comp_out - h_ref_comp_in);
    
    return e_ref_H_th;
}

} // namespace acmodel 