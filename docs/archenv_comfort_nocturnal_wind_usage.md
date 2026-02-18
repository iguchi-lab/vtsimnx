# `vtsimnx.archenv`（comfort / nocturnal / wind）使い方ガイド

このドキュメントは、次の関数の実務的な使い方をまとめたものです。

- 風圧: `make_wind`（`wind.py`）
- 夜間放射: `nocturnal_gain_by_angles`（`nocturnal.py`）
- 快適性: `calc_PMV`, `calc_PPD`, `calc_fungal_index`（`comfort.py`）

---

## 1. 風圧計算（`make_wind`）

### 関数

- `make_wind(d, s, c_in=0.7, c_out=-0.55, c_horizontal=-0.90)`

### 何をするか

風向カテゴリ `d` と風速 `s` から、方位別の風圧（E/S/W/N/H）を計算します。  
戻り値は `(中間DataFrame, 風圧dict)` です。

### 入力

- `d`: 風向カテゴリ `Series`
  - `0:無風, 1:NNE, ..., 16:N`
- `s`: 風速 `[m/s]` の `Series`
- `c_in`, `c_out`, `c_horizontal`: 風圧係数

### 出力

- `df`（中間DataFrame）
  - `風速_E/S/W/N`
  - `風圧_E/S/W/N/H`
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
print(df_wind[["風圧_E", "風圧_S", "風圧_W", "風圧_N", "風圧_H"]])
print(p["E"].head())
```

### 注意

- `d` と `s` は同じインデックスで使う
- 係数の符号を変えると圧力の向きが変わるので、建物モデル側の符号規約と合わせる

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
   - 水平面夜間放射 `[Wh/m2]` を直接使用

### 幾何の扱い

- `tilt_deg`: `0=水平上向き, 90=鉛直`
- view factor  
  `F_sky = (1 + cos(beta)) / 2`
- 面の夜間放射量  
  `夜間放射量（面） = rn_horizontal * F_sky`

### 出力

- 既定: `夜間放射量` の `Series`
- `return_details=True`:
  - `夜間放射量_水平`（= 入力 `rn_horizontal`）
  - `夜間放射量`

戻り値モード（イメージ）:

```text
return_details=False (既定)
  -> Series: 夜間放射量

return_details=True
  -> DataFrame:
       - 夜間放射量_水平
       - 夜間放射量
```

### 例（温湿度から推算）

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
rn_h = pd.Series(40.0, index=idx)  # [Wh/m2]

out = vt.nocturnal_gain_by_angles(
    tilt_deg=30.0,
    rn_horizontal=rn_h,
    return_details=True,
)
```

---

## 3. 快適性（`calc_PMV`, `calc_PPD`）

### 関数

- `calc_PMV(Met=1.0, W=0.0, Clo=1.0, t_a=20, h_a=50, t_r=20, v_a=0.2)`
- `calc_PPD(...)`

### 何をするか

- `calc_PMV`: PMV（温冷感申告の予測平均）を返す
- `calc_PPD`: PMV から PPD（不満足者率 [%]）を返す

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
- 入力が非現実領域だと PMV/PPD も解釈しづらくなるため、シミュレーション出力の妥当範囲確認を推奨

---

## 4. カビ指標（`calc_fungal_index`）

### 関数

- `calc_fungal_index(h, t)`

### 何をするか

湿度と温度から Fungal Index（カビ指標）を計算します。

### 入力

- `h`: 相対湿度（実装上は値をそのまま式に入れる）
- `t`: 温度 [degC]

### 例

```python
import vtsimnx as vt

fi = vt.calc_fungal_index(h=0.9, t=25.0)  # 0-1スケール入力の例
print(fi)
```

### 注意

- `comfort.py` の docstring には「`h` は 0-1 or 0-100」とありますが、式本体では正規化していません。  
  運用時は、プロジェクト内で湿度スケール（0-1 か 0-100）を統一して使うことを推奨します。

---

## 5. どの関数を使うべきか（目安）

- 風圧境界条件を作る: `make_wind`
- 傾斜面ごとの夜間放射を作る: `nocturnal_gain_by_angles`
- 室内快適性の評価: `calc_PMV`, `calc_PPD`
- カビ発生リスクの相対指標: `calc_fungal_index`

