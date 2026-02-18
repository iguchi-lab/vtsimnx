# `vtsimnx.archenv.solar` 使い方ガイド

`solar` 関連モジュールは、主に次の2ステップを扱います。

1. 太陽位置（高度・方位）を求める
2. 日射データ（全天/直達/拡散）から、任意面や窓への日射熱取得量を求める

本ドキュメントでは、公開関数の使い方と、入力の考え方をまとめます。

---

## 1. 座標系と用語（最初にここだけ確認）

- 太陽方位角 `AZs`（本モジュール独自系）
  - `0` = 南
  - `-90` = 東
  - `+90` = 西
  - `±180` = 北
- 面方位角 `azimuth_deg` も同じ座標系
- 面傾斜角 `tilt_deg`
  - `0` = 水平上向き
  - `90` = 鉛直
- 日射の基本量
  - `ghi`（GHI）: 水平面全天日射量
  - `dni`（DNI）: 法線面直達日射量
  - `dhi`（DHI）: 水平面拡散日射量

---

## 2. 公開関数一覧

### `sun_loc(idx, lat=..., lon=..., td=...)`

簡易式（赤緯・均時差・時角）で太陽位置を計算します。

- 用途: 高速に太陽位置を出したいとき
- 入力:
  - `idx`: `DatetimeIndex`
  - `lat`, `lon`: 緯度・経度（deg）
  - `td`: 時刻補正（h）
- 出力（主な列）:
  - `太陽高度 hs`
  - `太陽方位角 AZs`
  - `sin/cos` 系の中間列

### `astro_sun_loc(idx, lat=..., lon=..., td=...)`

`astropy` を使った高精度太陽位置計算です。

- 用途: 精度重視（`astropy` が必要）
- 入力:
  - `lat`, `lon` は float（deg）でも DMS 文字列でも可
- 出力（主な列）:
  - `太陽高度 alt`
  - `太陽方位角 az`（astropy由来）  
    ※ `solar_gain_by_angles` 内では本モジュール系 `AZs` に変換されます

### `sep_direct_diffuse(s_ig, s_hs, min_sun_alt_deg=0.0)`（下位ユーティリティ）

全天日射量（GHI）と太陽高度から Erbs 法で直散分離します。

- 定義場所: `vtsimnx.archenv.solar_separation`
- 通常利用では `solar_gain_by_angles` が内部で呼ぶため、直接呼ばないことも多いです

- 入力:
  - `s_ig`: GHI
  - `s_hs`: 太陽高度（deg）
- 出力:
  - `晴天指数 Kt`
  - `水平面拡散日射量 Id`
  - `法線面直達日射量 Ib`

### `solar_gain_by_angles(...)`

任意の面（方位角・傾斜角）の日射熱取得量を計算する基本APIです。

代表的な引数（主要部）:

```python
solar_gain_by_angles(
    *,
    azimuth_deg, tilt_deg, lat_deg=..., lon_deg=...,
    ghi=None, dni=None, dhi=None,
    glass=False, return_details=False,
    use_astro=False, time_alignment="timestamp", timestamp_ref="start",
    min_sun_alt_deg=0.0, solar_mode="all",
)
```

- 主な入力:
  - 幾何: `azimuth_deg`, `tilt_deg`, `lat_deg`, `lon_deg`
  - 日射（次のいずれか）
    1. `ghi` のみ（内部で Erbs 分離）
    2. `ghi + dni`（`Id` を復元）
    3. `dni + dhi`（そのまま使用）
  - そのほか: `glass`, `return_details`, `solar_mode`, `use_astro`, `time_alignment`, `timestamp_ref`
- 出力:
  - 既定: `日射熱取得量` の `Series` を返す
  - `return_details=True` のとき `DataFrame` を返し、`入射角cos`、直達/拡散/反射、`Ib/Id/hs/AZs` を含む
  - `glass=False` は壁面、`glass=True` はガラスを対象に内訳を作る

### `solar_gain_by_angles_with_shade(...)`

`solar_gain_by_angles` に窓シェードの幾何遮蔽（直達のみ低減）を加えた拡張APIです。

代表的な引数（主要部）:

```python
solar_gain_by_angles_with_shade(
    *,
    azimuth_deg, tilt_deg, window_width, window_height, shade_coords,
    shade_origin_mode="window_center",
    lat_deg=..., lon_deg=...,
    ghi=None, dni=None, dhi=None,
    glass=False, return_details=False,
    use_astro=False, time_alignment="timestamp", timestamp_ref="start",
    min_sun_alt_deg=0.0, solar_mode="all",
)
```

- 追加入力:
  - `window_width`, `window_height`
  - `shade_coords`
    - 単一ポリゴン: `[(x, y, z), ...]`
    - 複数ポリゴン: `[[(x, y, z), ...], ...]`
  - `shade_origin_mode`
    - `"window_center"`: 窓中心基準
    - `"window_top_center"`: 窓上端中央基準
- シェード座標の意味（窓ローカル座標）
  - `x`: 右正
  - `y`: 上正
  - `z`: 外向き法線方向正（窓面は `z=0`）
- 挙動:
  - 拡散・反射は**変更しない**
  - 直達のみ `日向率(1-η)` を掛ける（`η`: 被影率）
  - 複数ポリゴンは重なりを二重計上しない（和集合面積）
- 追加出力列:
  - 既定: `日射熱取得量` の `Series`
  - `return_details=True` のとき `被影率η`, `日向率(1-η)` を含む詳細 `DataFrame`

---

## 3. よく使う例

最初に用語だけ整理:

- `ghi` (Global Horizontal Irradiance): 水平面全天日射量
- `dni` (Direct Normal Irradiance): 法線面直達日射量
- `dhi` (Diffuse Horizontal Irradiance): 水平面拡散日射量

HASP 等の気象データを使う場合は、典型的に  
`ghi = df["水平面全天日射量"]`、`dni = df["直達日射量"]`、`dhi = df["水平面拡散日射量"]`  
のように対応付けて渡します（データに `ghi` がない場合は `dni + dhi` で使う運用でも可）。

### 3-1. 基本（DNI + DHI を直接与える）

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-06-21 12:00:00", periods=24, freq="1h")
s_dni = pd.Series(800.0, index=idx)  # dni: 法線面直達日射量
s_dhi = pd.Series(100.0, index=idx)  # dhi: 水平面拡散日射量

out = vt.solar_gain_by_angles(
    azimuth_deg=0.0,   # 南面
    tilt_deg=90.0,     # 鉛直
    lat_deg=35.0,
    lon_deg=139.0,
    dni=s_dni,
    dhi=s_dhi,
    glass=False,  # 壁面
)

print(out.head())
```

### 3-2. `ghi` のみ（内部で直散分離）

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-06-21 06:00:00", periods=12, freq="1h")
s_ghi = pd.Series(300.0, index=idx)  # ghi: 水平面全天日射量

out = vt.solar_gain_by_angles(
    azimuth_deg=-90.0,  # 東面
    tilt_deg=90.0,
    lat_deg=35.0,
    lon_deg=139.0,
    ghi=s_ghi,
)
```

### 3-3. シェードあり（複数ポリゴン）

```python
import pandas as pd
import vtsimnx as vt

idx = pd.date_range("2026-06-21 09:00:00", periods=8, freq="1h")
s_dni = pd.Series(700.0, index=idx)  # dni
s_dhi = pd.Series(120.0, index=idx)  # dhi

shade_polys = [
    # 庇（上側）
    [(-1.0, 1.0, 0.4), (1.0, 1.0, 0.4), (1.0, 0.7, 0.4), (-1.0, 0.7, 0.4)],
    # 左袖壁（例）
    [(-1.0, 1.0, 0.3), (-0.8, 1.0, 0.3), (-0.8, -1.0, 0.3), (-1.0, -1.0, 0.3)],
]

out = vt.solar_gain_by_angles_with_shade(
    azimuth_deg=0.0,
    tilt_deg=90.0,
    window_width=2.0,
    window_height=2.0,
    shade_coords=shade_polys,
    shade_origin_mode="window_center",
    lat_deg=35.0,
    lon_deg=139.0,
    dni=s_dni,
    dhi=s_dhi,
    glass=True,  # ガラス面を対象にする
    return_details=True,
)

print(out[["被影率η", "日向率(1-η)"]].head())
```

---

## 4. 実務上の注意

- `DatetimeIndex` は日射データと同じインデックスを使う
- `time_alignment="center"` を使う場合は `timestamp_ref`（`"start"`/`"end"`）を意識する
- `solar_mode="diffuse_only"` は、直達を0扱いにしたいケース（周辺遮蔽を別途見たい場合など）に便利
- `solar_gain_by_angles_with_shade` のシェード幾何は窓ローカル座標で統一する
- 影計算は幾何学ベースのため、詳細形状が複雑な場合はポリゴン分割して与えると扱いやすい
- 日射データの引数は `ghi` / `dni` / `dhi` を使う（旧日本語名は廃止）

