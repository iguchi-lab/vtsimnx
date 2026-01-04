#pragma once

#include <vector>

#include "types/common_types.h"

// 1タイムステップ分の結果を格納する構造体
struct TimestepResult {
    // 圧力関連（pressureCalcがtrueの場合のみ有効）
    // 出力はバイナリ(float32)前提のため、出力用配列は float を内部表現として保持する
    std::vector<float> pressure; // ノード圧力（キー順は VentilationNetwork が提供）
    std::vector<float> flowRate; // 個別ブランチの風量データ（キー順は VentilationNetwork が提供）
    FlowBalanceMap flowBalanceMap;

    // 温度関連（temperatureCalcがtrueの場合のみ有効）
    // temperature は 3 系列に分けて出力する
    std::vector<float> temperature;         // main: normal + aircon + unknown
    std::vector<float> temperatureCapacity; // capacity
    std::vector<float> temperatureLayer;    // layer
    // heat_rate はカテゴリ別に出力する
    std::vector<float> heatRateAdvection;
    std::vector<float> heatRateHeatGeneration;
    std::vector<float> heatRateSolarGain;
    std::vector<float> heatRateNocturnalLoss;
    std::vector<float> heatRateConvection;
    std::vector<float> heatRateConduction;
    std::vector<float> heatRateRadiation;
    std::vector<float> heatRateCapacity;
    HeatBalanceMap heatBalanceMap;

    // エアコン関連（temperatureCalcがtrueの場合のみ有効）
    std::vector<float> airconSensibleHeat; // キー順は AirconController が提供
    std::vector<float> airconLatentHeat;   // キー順は AirconController が提供
    std::vector<float> airconPower;        // キー順は AirconController が提供
    std::vector<float> airconCOP;          // キー順は AirconController が提供

    // その他（将来の拡張用）
    AirconDataMap humidityMap;
    AirconDataMap concentrationMap;
};

// シミュレーション結果の履歴を格納する構造体
struct SimulationResults {
    std::vector<TimestepResult> timestepHistory;
};


