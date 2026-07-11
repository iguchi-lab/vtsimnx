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
| 熱量（熱流） | `W` | `HEAT_RATE_W` |
| 熱コンダクタンス | `W/K` | `CONDUCTANCE_W_K` |
| 熱貫流率 | `W/(m2·K)` | `U_VALUE_W_M2K` |
| 面積 | `m2` | `AREA_M2` |
| 長さ | `m` | `LENGTH_M` |
| 絶対湿度 | `kg/kg'` | `HUMIDITY_RATIO` |
| 発湿 | `kg/s` | `MOISTURE_GEN_KG_S` |
| 時間 | `s` | `TIME_S` |
| 熱容量 | `J/K` | `THERMAL_MASS_J_K` |
| 日射 | `W/m2` | `SOLAR_IRRADIANCE_W_M2` |

補足:

- builder / solver の風量 `vol` の基本単位は **`m3/s`** です（ドキュメント上 `m3/h` で書く場合は換算が必要）。
- 濃度 `c` はモデル依存のため表記は `-`（無次元または kg/m3 等）。

## 主な入力フィールド

| フィールド | 単位 |
|---|---|
| `t`, `pre_temp` | degC |
| `p` | Pa |
| `v` | m3 |
| `vol` | m3/s |
| `x` | kg/kg' |
| `conductance` | W/K |
| `u_value` | W/(m2·K) |
| `area` | m2 |
| `heat_generation` | W |
| `humidity_generation` | kg/s |
| `thermal_mass` | J/K |
| `solar` | W/m2 |
| `timestep` | s |

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
