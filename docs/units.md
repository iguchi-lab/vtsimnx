# 単位系

建築シミュレーション向けに、入力フィールドと成果物系列の単位を明示します。
コード正本: `vtsimnx.units`（`FIELD_UNITS` / `SERIES_UNITS`）。
API 入力スキーマ: `engine/app/schemas/config.py` の `json_schema_extra={"unit": ...}`。

## 基本単位

| 物理量 | 表記 | 定数 |
|---|---|---|
| 温度 | `degC` | `TEMPERATURE_C` |
| 圧力 | `Pa` | `PRESSURE_PA` |
| 気積 | `m3` | `VOLUME_M3` |
| 体積流量 | `m3/s` | `VOLUME_FLOW_M3_S` |
| 質量流量 | `kg/s` | `MASS_FLOW_KG_S` |
| 熱流・処理熱量率 | `W` | `HEAT_RATE_W` |
| 熱コンダクタンス | `W/K` | `CONDUCTANCE_W_K` |
| 熱貫流率 | `W/(m2·K)` | `U_VALUE_W_M2K` |
| 面積 | `m2` | `AREA_M2` |
| 長さ | `m` | `LENGTH_M` |
| 絶対湿度（乾き空気基準の湿度比） | `kg/kg'` | `HUMIDITY_RATIO` |
| 発湿 | `kg/s` | `MOISTURE_GEN_KG_S` |
| 時間 | `s` | `TIME_S` |
| 熱容量 | `J/K` | `THERMAL_MASS_J_K` |
| 日射 | `W/m2` | `SOLAR_IRRADIANCE_W_M2` |

補足:

- builder / solver の風量 `vol` の基本単位は **`m3/s`** です（ドキュメント上 `m3/h` で書く場合は換算が必要）。
- 濃度 `c` はモデル依存のため表記は `-`（個/m³、kg/m³、濃度比等）。濃度比を用いる場合も `dust_generation` は「濃度×m³/s」と整合する値にする。ppm と kg/s をそのまま組み合わせない。
- `kg/kg'` の分母は乾き空気質量であり、湿り空気質量や空間体積ではない。
- W はエネルギーの時間率（J/s）である。区間積算値 Wh と区別し、`E[Wh] = sum(P[W] * Δt[s]) / 3600` で換算する。

## 主な入力フィールド

| フィールド | 単位 |
|---|---|
| `t`, `pre_temp` | degC |
| `pre_rh` | %（0より大きく100以下。0.5ではなく50が50%） |
| `p` | Pa |
| `v` | m3 |
| `vol` | m3/s |
| `p_max`, `p1` | Pa |
| `q_max`, `q1` | m3/s |
| `x` | kg/kg' |
| `conductance` | W/K |
| `u_value` | W/(m2·K) |
| `area` | m2 |
| `heat_generation` | W |
| `humidity_generation` | kg/s |
| `thermal_mass` | J/K |
| `solar` | W/m2 |
| `timestep` | s |
| `aircon[].ac_spec.Q.*.*`, `P.*.*`, `P_fan.*.*` | kW |
| `aircon[].ac_spec.V_inner.*.*`, `V_outer.*.*`, `V_vent` | m3/s |

`DUCT_CENTRAL` の風量比は `Q.<mode>.rtd` [kW] を1000倍してWへ換算した値を分母に用いる。
`V_inner.<mode>.dsgn` は風量制御の上限であり、`rtd` や `mid` と役割が異なる。
入力と式の詳細は[全館空調の風量制御](duct_central_airflow_control.md)を参照する。

## 前処理と solver の単位が異なる項目

| 項目 | 前処理・raw 入力 | solver へ渡す値 |
|---|---|---|
| `read_hasp` の外気絶対湿度 | g/kg' | 1000 で除して `x` [kg/kg'] |
| `schedule.latent_moisture` | kg/h | 3600 で除して発湿 [kg/s] |
| raw `moisture_capacity` | 既定 J/(kg/kg') | `2.5e6` J/kg で除して kg/(kg/kg) |
| raw `moisture_capacity_unit="kg/(kg/kg)"` | kg/(kg/kg) | 換算せず、追加材料節点へ展開 |
| `surfaces.solar`, `surfaces.nocturnal` | W/m² | 面積と吸収率等を掛けて発熱 [W] |

`nocturnal_gain_by_angles` の docstring は Wh/m² と記載するが、関数は入力時間間隔を用いた積算・平均化を行わず、直接入力に天空率を掛ける。builder へ接続する場合は損失を正とした W/m² の系列を用いる。積算放射量を使う場合は区間長 [h] で除す。温湿度推算経路の符号は [機能ガイド](archenv_comfort_nocturnal_wind_usage.md) の制約を参照する。

`moisture_capacity` は raw builder と直接 solver 入力で意味が異なる。builder は元の室に容量を残さず、追加節点と `k=C/Δt` の伝達枝を生成する。時間刻み変更時には伝達係数も変わるため、物性を固定した時間離散化誤差の検証と区別する。

## 主な成果物 series

| series | 単位 |
|---|---|
| `vent_pressure` | Pa |
| `vent_flow_rate` | m3/s |
| `thermal_temperature` | degC |
| `thermal_heat_rate_*` | W |
| `humidity_x` | kg/kg' |
| `humidity_flux` | kg/s |
| `aircon_sensible_heat` / `aircon_latent_heat` / `aircon_power` | W |

```python
from vtsimnx.units import unit_for_field, unit_for_series

assert unit_for_field("t") == "degC"
assert unit_for_series("vent_flow_rate") == "m3/s"
```

`get_artifact_file` / `decode_f32_series` で復元した DataFrame には、既知 series について `df.attrs["unit"]` と `df.attrs["series"]` が付きます。
