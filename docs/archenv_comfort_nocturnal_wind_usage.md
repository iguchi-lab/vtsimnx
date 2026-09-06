# `vtsimnx.archenv`（comfort / nocturnal / ground / wind）使い方ガイド

このドキュメントは、次の関数の実務的な使い方をまとめたものです。

- 風圧: `make_wind`（`wind.py`）
- 夜間放射: `nocturnal_gain_by_angles`（`nocturnal.py`）
- 地盤温度: `ground_temperature_by_depth`（`ground.py`）
- 快適性: `calc_PMV`, `calc_PPD`, `calc_fungal_index`（`comfort.py`）

列名の正本は英語キー（`vtsimnx.archenv.columns`）です。日本語レガシー名への変換は `rename_to_japanese` / `with_japanese_column_aliases` を使えます。

---

## 1. 風圧計算（`make_wind`）

### 関数

- `make_wind(d, s, c_in=0.7, c_out=-0.55, c_horizontal=-0.90, *, air_density_kg_m3=None)`

### 何をするか

風向カテゴリ `d` と風速 `s` から、方位別の風圧（E/S/W/N/H）を計算します。
戻り値は `(中間DataFrame, 風圧dict)` です。

### 入力

- `d`: 風向カテゴリ `Series`
  - `0:無風, 1:NNE, ..., 16:N`
- `s`: 風速 `[m/s]` の `Series`
- `c_in`, `c_out`, `c_horizontal`: 風圧係数
- `air_density_kg_m3`: 空気密度 [kg/m³]。省略時は20℃の近似値

### 出力

- `df`（中間DataFrame）
  - `wind_speed_e` / `wind_speed_s` / `wind_speed_w` / `wind_speed_n`
  - `wind_pressure_e` / `wind_pressure_s` / `wind_pressure_w` / `wind_pressure_n` / `wind_pressure_h`
- `wind_pressure`（`dict[str, Series]`）
  - `"E"`, `"S"`, `"W"`, `"N"`, `"H"` の5方向

### 例

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-01-01 00:00:00", periods=3, freq="1h")
d = pd.Series([4, 8, 12], index=idx)   # E, S, W 相当
s = pd.Series([2.0, 3.5, 1.8], index=idx)

df_wind, p = vt.make_wind(d, s)
print(df_wind[["wind_pressure_e", "wind_pressure_s", "wind_pressure_w", "wind_pressure_n", "wind_pressure_h"]])
print(p["E"].head())
```

### 注意

- `d` と `s` は同じインデックスで使う
- 現行の風下側は `-ρ*c_out*u_component²/2` であり、既定 `c_out=-0.55` でも**正圧**を返す。例えば東風2 m/sでは東面約+1.687 Pa、西面約+1.326 Paとなる。一般の `p=Cp*ρ*u²/2` の負の風下係数をそのまま当てはめた結果とは異なる。実装の符号に関する要検討事項であり、研究計算では外部の風圧係数・圧力基準と照合する。
- `d=0` は名称上「無風」だが、実装は風速を強制的に0にしない。無風データでは `s=0` も指定する。風向カテゴリは太陽方位角の定義と異なる。

---

## 2. 夜間放射（`nocturnal_gain_by_angles`）

### 関数

- `nocturnal_gain_by_angles(tilt_deg, t_out=None, rh_out=None, rn_horizontal=None, return_details=False)`

### 何をするか

任意面（傾斜角指定）の夜間放射量を返します。
水平面夜間放射を

- `t_out`・`rh_out` から推算する
- 直接与える

のどちらにも対応しています。

### 入力パターン（どちらか）

1. `t_out` + `rh_out`
   - 内部で `rn(t,h)` を使って水平面夜間放射を推算
2. `rn_horizontal`
   - 水平面の正味放射損失を正とした値 [W/m²] を直接使用（builderへ渡す場合）

### 幾何の扱い

- `tilt_deg`: `0=水平上向き, 90=鉛直`
- view factor
  `F_sky = (1 + cos(beta)) / 2`
- 面の夜間放射量
  `nocturnal_radiation = rn_horizontal * F_sky`

### 出力

- 既定: `nocturnal_radiation` の `Series`
- `return_details=True`:
  - `nocturnal_radiation_horizontal`（= 入力 `rn_horizontal`）
  - `nocturnal_radiation`

戻り値モード（イメージ）:

```text
return_details=False (既定)
  -> Series: nocturnal_radiation

return_details=True
  -> DataFrame:
       - nocturnal_radiation_horizontal
       - nocturnal_radiation
```

### 温湿度推算経路の制約

現行 `rn(t,h)` は放射収支が**負値**となる場合がある。5℃・相対湿度70%では水平面出力は約 -66.98 となる。builder は `heat_generation=-A*epsilon*nocturnal` とするため、この出力をそのまま渡すと表面加熱になる。推算式の出典・符号の確認が必要であり、以下は関数の呼出例であって、builderへの直接接続例ではない。比較計算には単位・符号が確認できる放射損失系列を用いる。

`nocturnal.py` の docstring には MJ/m²・Wh/m² の表記が残るが、入力の時間刻みによる積算・平均化は行われない。直接入力経路は天空率を掛けるだけなので、builder に渡す W/m² を用い、Wh/m² の積算値は区間長 [h] で除してから渡す。昼夜・雲量・周囲建物の遮蔽を自動判定する機能はない。

### 例（温湿度から推算・符号確認用）

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h")
t = pd.Series(5.0, index=idx)    # 外気温 [degC]
rh = pd.Series(70.0, index=idx)  # 相対湿度 [%]

out = vt.nocturnal_gain_by_angles(
    tilt_deg=90.0,
    t_out=t,
    rh_out=rh,
    return_details=True,
)
print(out.head())
```

### 例（水平面夜間放射を直接入力）

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h")
rn_h = pd.Series(40.0, index=idx)  # 放射損失を正、[W/m²]

out = vt.nocturnal_gain_by_angles(
    tilt_deg=30.0,
    rn_horizontal=rn_h,
    return_details=True,
)
```

---

## 3. ground_temperature（`ground_temperature_by_depth`）

### 関数

- `ground_temperature_by_depth(depth_m, t_out, solar_horizontal=None, nocturnal_horizontal=None, ...)`

### 何をするか

不易層条件（深さ・温度）と気象時系列から、任意深さのground_temperatureを 1 次元熱伝導で推定します。
地盤物性（熱伝導率・体積熱容量）を引数で指定できます。

### 主な入力

- `depth_m`: 取得したい深さ [m]（単一値または配列）
- `t_out`: 外気温 `Series`（必須）
- `solar_horizontal`: 水平面日射量 `Series`（任意）
- `nocturnal_horizontal`: 水平面nocturnal_radiation `Series`（任意）
- `deep_layer_depth_m`: 不易層深さ [m]（既定 `10.0`）
- `deep_layer_temp_c`: 不易層温度 [degC]（既定 `10.0`）
- `thermal_conductivity_w_mk`: 熱伝導率 [W/m/K]
- `volumetric_heat_capacity_j_m3k`: 体積熱容量 [J/m3/K]
- `spinup`: 助走モード（既定 `False`）
- `spinup_cycles`: 助走の繰り返し回数（既定 `5`）

### 境界条件（surface_equivalent_temperature）

`Ts = t_out + a_solar * solar_horizontal - a_noct * nocturnal_horizontal`

- `a_solar` = `solar_to_surface_temp_coeff`
- `a_noct` = `nocturnal_to_surface_temp_coeff`
- 日射・夜間放射を W/m² で入力するとき、係数の次元は m²·K/W である。夜間放射は損失を正にそろえる。
- 地表の `solar_horizontal` は GHI とする。法線面直達DNIと水平面拡散DHIの単純和ではなく、太陽高度を使って水平面へ投影する。

### 出力

- `depth_m` が単一 + `return_details=False`（既定）:
  - `ground_temperature` の `Series`
- `depth_m` が複数:
  - `ground_temperature_0.100m` のような列を持つ `DataFrame`
- `return_details=True`:
  - `surface_equivalent_temperature` を先頭列として追加

助走モード（`spinup=True`）:

- 入力気象（例: 1年）を `spinup_cycles` 回繰り返して内部計算する
- 戻り値は最終周期（1周期分）だけ返す
- 初期条件依存の過渡を減らしたい場合に有効

### 例

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-01-01 00:00:00", periods=24 * 7, freq="1h")
t_out = pd.Series(8.0, index=idx)
solar_h = pd.Series(150.0, index=idx)
rn_h = pd.Series(40.0, index=idx)

tg = vt.ground_temperature_by_depth(
    depth_m=[0.1, 1.0, 3.0],
    t_out=t_out,
    solar_horizontal=solar_h,
    nocturnal_horizontal=rn_h,
    deep_layer_depth_m=10.0,
    deep_layer_temp_c=10.0,
    thermal_conductivity_w_mk=1.5,
    volumetric_heat_capacity_j_m3k=2.2e6,
    solar_to_surface_temp_coeff=0.003,
    nocturnal_to_surface_temp_coeff=0.003,
    spinup=True,
    spinup_cycles=5,
    return_details=True,
)
print(tg.head())
```

### 注意

- `t_out` / `solar_horizontal` / `nocturnal_horizontal` は同じ `DatetimeIndex` を使う
- 時間間隔は等間隔である必要がある
- 係数 `solar_to_surface_temp_coeff`, `nocturnal_to_surface_temp_coeff` はモデル同定で調整する
- 年周期データでは `spinup=True` を使うと深部温度の初期過渡を抑えやすい。回数を増やして周期末の差が十分小さいことを確認する。
- 数値解法は後退Euler法による一次元差分で、下端は固定温度。地下水による移流、凍結・融解、基礎まわりの三次元熱流は含まない。

---

## 4. 快適性（`calc_PMV`, `calc_PPD`）

### 関数

- `calc_PMV(Met=1.0, W=0.0, Clo=1.0, t_a=20, h_a=50, t_r=20, v_a=0.2)`
- `calc_PPD(...)`

### 何をするか

- `calc_PMV`: PMV（温冷感申告の予測平均）を返す
- `calc_PPD`: 同じ温熱条件から内部でPMVを計算し、PPD（不満足者率 [%]）を返す。PMV値一つを渡す関数ではない

### 主な引数

- `Met`: 代謝量 [met]
- `W`: 外部仕事 [W/m2] 相当（通常0）
- `Clo`: 着衣量 [clo]
- `t_a`: 空気温度 [degC]
- `h_a`: 相対湿度 [%]
- `t_r`: 平均放射温度 [degC]
- `v_a`: 風速 [m/s]

### 例

```python
import vtsimnx as vt

pmv = vt.calc_PMV(Met=1.2, Clo=0.7, t_a=26.0, h_a=55.0, t_r=26.0, v_a=0.15)
ppd = vt.calc_PPD(Met=1.2, Clo=0.7, t_a=26.0, h_a=55.0, t_r=26.0, v_a=0.15)
print(pmv, ppd)
```

### 注意

- `h_a` は `%` 前提（`0-100`）
- PPDは `100-95*exp(-0.03353*PMV**4-0.2179*PMV**2)` で、PMV=0でも5%である。
- 衣服表面温度の反復が100回で収束しない場合は警告を出して値を返す。警告を無視して適用しない。
- 現行発汗項は `0.42*((M-W)-58.15)` をそのまま用い、低代謝側で0に制限しない。標準的なFanger実装との一致を無条件に主張できない。規格適合性や個人の快適性を保証するものではない。

---

## 5. カビ指標（`calc_fungal_index`）

### 関数

- `calc_fungal_index(h, t)`

### 何をするか

湿度と温度から Fungal Index（カビ指標）を計算します。

### 入力

- `h`: 相対湿度 **[%]**（C++ と同じ。`0..100`）
  - 後方互換: `0 < h <= 1` は割合とみなし `DeprecationWarning`
- `t`: 温度 [degC]

### 例

```python
import vtsimnx as vt

fi = vt.calc_fungal_index(h=90.0, t=25.0)  # 相対湿度 90%
print(fi)
```

### 注意

- 湿度スケールの正本は **パーセント** です。ただしPythonでは `h=1.0` は互換処理により100%と解釈され、1%とはならない。0より大きく1以下の低湿度を表したい場合に曖昧さがある。
- 指標は経験式の値であり、発生確率や健康リスクそのものではない。材質、濡れ時間、菌種等を直接扱わず、負値も返す。出典・適用条件を確認して相対比較に限定する。

---

## 6. どの関数を使うべきか（目安）

- 風圧境界条件を作る: `make_wind`
- 傾斜面ごとの夜間放射を作る: `nocturnal_gain_by_angles`
- 地盤温度境界条件を作る: `ground_temperature_by_depth`
- 室内快適性の評価: `calc_PMV`, `calc_PPD`
- カビ発生リスクの相対指標: `calc_fungal_index`

