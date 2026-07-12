#pragma once

#include <archenv.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

// === 物理定数 ===
namespace PhysicalConstants {
    // archenv から参照（solver側での重複定義を避け、単位系も archenv に揃える）
    inline constexpr double DENSITY_DRY_AIR = archenv::DENSITY_DRY_AIR;                 // [kg/m³]
    inline constexpr double SPECIFIC_HEAT_AIR = archenv::SPECIFIC_HEAT_AIR;             // [J/(kg·K)]
    inline constexpr double LATENT_HEAT_VAPORIZATION = archenv::LATENT_HEAT_VAPORIZATION; // [J/kg] (0℃基準)

    inline constexpr double DEFAULT_VENTILATION_RATE = 28.2; // デフォルト換気量 [m³/min]
}

// === 許容誤差定数 ===
namespace ToleranceConstants {
    inline constexpr double TEMPERATURE_TOLERANCE = 1e-3; // 温度許容誤差 [K]
    inline constexpr double CAPACITY_TOLERANCE = 1e-6;    // 能力許容誤差
    inline constexpr double BOUNDS_TOLERANCE = 1e-6;      // 境界値許容誤差 [K]
    inline constexpr double DEFAULT_HUMIDITY = 0.0;       // デフォルト湿度
    inline constexpr double DEFAULT_LATENT_HEAT = 0.0;    // デフォルト潜熱
}

// 型エイリアス（外部I/F向け: 可読性優先）
using PressureMap    = std::map<std::string, double>;                         // {node} -> [Pa]
using FlowRateMap    = std::map<std::pair<std::string, std::string>, double>; // {source, target} -> [m³/s]
using FlowBalanceMap = std::map<std::string, double>;                         // {node} -> [m³/s] 体積流量収支

using TemperatureMap = std::map<std::string, double>;                         // {node} -> [K]
using HeatRateMap    = std::map<std::pair<std::string, std::string>, double>; // {source, target} -> [W]
using HeatBalanceMap = std::map<std::string, double>;                         // {node} -> [W]

using AirconDataMap = std::map<std::string, double>; // {node} -> [値]

// シミュレーション定数を格納する構造体
struct SimulationConstants {
    std::string startTime;
    std::string endTime;
    int timestep;
    int length;
    double ventilationTolerance; // 圧力ソルバ: calc_p ノード体積流量収支の最大絶対値許容 [m³/s]
    // thermalTolerance は後方互換の代表値（空調温度差 [K]）。
    // 用途別の正本は以下3つ（パーサ未指定時は thermal と同じ値で埋める）。
    double thermalTolerance;
    double airconTemperatureToleranceK = 0.0;                 // 空調ON/OFF判定 [K]
    double thermalBalanceToleranceW = 0.0;                    // 熱収支RMSE合否 [W]
    double thermalLinearResidualRelativeTolerance = 0.0;      // DirectT の max|Ax-b| / max(1,|b|) 相対許容
    double convergenceTolerance;
    // 追加: 圧力-熱の連成反復の停止判定に使う許容誤差（0以下なら convergenceTolerance を使用）
    // 単位: pressure [Pa], temperature [K]
    double couplingPressureTolerance = 0.0;
    double couplingTemperatureTolerance = 0.0;
    // 湿気連成（x/w）の停止判定。0以下なら convergenceTolerance を使用。
    double couplingHumidityTolerance = 0.0;
    // 反復上限（正の整数）。パーサ未指定時は 100。
    // maxCoupling / maxAircon が 0 のときは maxInnerIterations を実効値として使う。
    std::size_t maxInnerIterations = 100;
    std::size_t maxCouplingIterations = 0;
    std::size_t maxAirconControlIterations = 0;
    // 3ネットワーク連成の有効化（既定ON）
    bool moistureCouplingEnabled = true;
    // 内側反復での緩和係数（0<alpha<=1）。1.0=緩和なし
    double humidityRelaxation = 1.0;
    double latentRelaxation = 1.0;
    // 潜熱→熱: 0=Disabled（既定）, 1=FromHumidityChange（実験・非推奨・将来削除予定: Δx全量）,
    // 2=FromPhaseChange（材料 moisture_conductance 相変化のみ）
    int latentCouplingMode = 0;
    // 換気移流・空気蓄熱・空調処理熱を湿り空気エンタルピー収支で解く（既定OFF）
    // from_humidity_change (latentCouplingMode==1) との併用は禁止
    bool moistEnthalpyEnabled = false;
    // 湿気内部ソルバ（直接法）の残差許容誤差
    double humiditySolverTolerance = 1e-9;
    // 潜熱源収束: |ΔQ| ≤ abs + rel * max(|Qold|,|Qnew|)
    double couplingLatentAbsoluteToleranceW = 1.0;
    double couplingLatentRelativeTolerance = 1e-3;
    bool pressureCalc;
    bool temperatureCalc;
    bool humidityCalc = false;
    bool concentrationCalc = false;
    // 追加: ログ・出力制御
    // 0: silent, 1: normal, 2: verbose, 3: debug
    int logVerbosity = 1;
    // フォールバック詳細ログを有効化
    bool logFallbackDetails = true;
    // フォールバック関連の補助出力(csv/txt)を有効化
    bool exportFallbackArtifacts = true;
};

// thermalTolerance のみ設定したテスト/旧コード向けの実効値。
inline double effectiveAirconTemperatureToleranceK(const SimulationConstants& c) noexcept {
    return (c.airconTemperatureToleranceK > 0.0) ? c.airconTemperatureToleranceK : c.thermalTolerance;
}
inline double effectiveThermalBalanceToleranceW(const SimulationConstants& c) noexcept {
    return (c.thermalBalanceToleranceW > 0.0) ? c.thermalBalanceToleranceW : c.thermalTolerance;
}
inline double effectiveThermalLinearResidualRelativeTolerance(const SimulationConstants& c) noexcept {
    return (c.thermalLinearResidualRelativeTolerance > 0.0)
               ? c.thermalLinearResidualRelativeTolerance
               : c.thermalTolerance;
}


