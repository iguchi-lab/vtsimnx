## `vtsimnx.schedule` 使い方ガイド

`vtsimnx.schedule` は、**1年8760時間分のスケジュール**をまとめて扱うためのユーティリティ群です。

- 休日フラグ (`holiday`)
- 暖冷房期間 (`period_1`..`period_8`)
- 換気量 (`vol`)
- 顕熱発熱 (`sensible_heat`)
- 潜熱由来の水分発生量 (`latent_moisture`)
- 空調モード／設定温度／設定湿度 (`ac_mode`, `pre_tmp`, `pre_rh`)

を一括生成してくれます。

---

### 1. どこにあるか

- パッケージ: `vtsimnx.schedule`（`__init__.py` から再エクスポート）
- モジュール構成:
  - `schedule/common.py` — 休日フラグ、暖冷房期間、8760生成関数
  - `schedule/aircon.py` — 空調モード (`ac_mode`)、設定温度 (`pre_tmp`)、設定湿度 (`pre_rh`)
  - `schedule/vol.py` — 換気量プロファイルと `vol`
  - `schedule/sensible_heat.py` — 顕熱プロファイルと `sensible_heat`
  - `schedule/latent_moisture.py` — 潜熱由来の水分発生プロファイルと `latent_moisture`

トップレベルでは次のように使えます:

```python
import vtsimnx as vt

ac_mode = vt.schedule.ac_mode
pre_tmp = vt.schedule.pre_tmp
pre_rh  = vt.schedule.pre_rh
vol     = vt.schedule.vol
sensible_heat = vt.schedule.sensible_heat
latent_moisture = vt.schedule.latent_moisture
```

---

### 2. 共通の考え方（365日×24時間）

#### 2.1 `holiday` と `period_x`

`schedule/common.py` では、1年365日分の配列が定義されています。

- `holiday`: 長さ 365 の配列、`1`=休日, `0`=平日
- `period_1`..`period_8`: 地域別の暖冷房期間
  - 値は `1`=暖房期, `0`=非空調期, `-1`=冷房期

これらをもとに、1日の 24 要素プロファイル（平日/休日別）を 8760 に展開する関数が **`make_8760_data`** と **`make_8760_by_holiday`** です。

```python
from vtsimnx.schedule import make_8760_data, make_8760_by_holiday, period_1, holiday

# 例: 暖房/冷房の出力コードをもつ 24h プロファイル（平日/休日）
w_h = [...]  # 暖房期・平日の24要素
h_h = [...]  # 暖房期・休日の24要素
w_c = [...]  # 冷房期・平日の24要素
h_c = [...]  # 冷房期・休日の24要素

ac_mode_8760 = make_8760_data(period_1, holiday, w_h, h_h, w_c, h_c, default=0)
```

#### 2.2 24h プロファイル → 8760 の基本パターン

- **日内（24h）**: 「平日用」「休日用」の 24 要素配列を用意する。
- **年間（365日）**: `holiday` と `period_x` を見ながら、日ごとにどの 24 要素を使うかを決める。
- **時間（8760h）**: 365日分の 24h を順に連結したものが最終的なスケジュール。

---

### 3. 空調スケジュール（`ac_mode`, `pre_tmp`, `pre_rh`）

`schedule/aircon.py` では、**地域×部屋**ごとの空調関連スケジュールを組み立てています。

#### 3.1 空調モードコード

- `AC_MODE_STOP = 0`
- `AC_MODE_HEATING = 1`
- `AC_MODE_COOLING = 2`
- `AC_MODE_AUTO = 3`

`ac_mode_profiles` は「部屋 × (暖房/冷房) × (平日/休日)」で 24 要素ずつ持ちます。  
これを `make_8760_data(period_x, holiday, ...)` で展開して 8760 要素の `ac_mode` にします。

#### 3.2 設定温度・設定湿度

同様に `pre_tmp_profiles` / `rh_profiles` をもとに、

- `pre_tmp`: 地域×部屋の設定温度 [℃]
- `pre_rh`: 地域×部屋の設定相対湿度 [%]

の 8760 シリーズを構成しています。

生成関数:

- `build_ac_mode(*, holiday_days=holiday)` → `ac_mode`
- `build_pre_tmp(*, holiday_days=holiday)` → `pre_tmp`
- `build_pre_rh(*, holiday_days=holiday)` → `pre_rh`

これらはモジュール import 時に一括生成され、`vt.schedule.ac_mode` などとして直接参照できます。

---

### 4. 換気スケジュール（`vol`）

`schedule/vol.py` では、換気量プロファイルを定義し、8760 の `vol` を構成します。

#### 4.1 `vent_profiles`（1日24要素 × 平日/休日）

```python
from vtsimnx import schedule as sch

sch.vent_profiles["LD"]["平日"]   # LD の平日 24h 換気プロファイル
sch.vent_profiles["LD"]["休日"]   # LD の休日 24h プロファイル
```

値は「割合×定格風量」を [m³/s] 単位に換算したもの（`* 300 / 3600` のような形）になっています。

#### 4.2 `build_vol_schedule` と `vol`

```python
from vtsimnx import schedule as sch

vol_8760 = sch.build_vol_schedule()  # room → 8760 Series
# 互換性のため、モジュール import 時点で sch.vol も生成済み
LD_vol = sch.vol["LD"]
```

`build_vol_schedule` は `vent_profiles` と `holiday` をもとに `make_8760_by_holiday` を呼び出し、各部屋の 8760h 換気スケジュールを返します。

---

### 5. 顕熱スケジュール（`sensible_heat`）

`schedule/sensible_heat.py` は、照明・機器・人体などの **顕熱発熱** の 24h プロファイルを定義し、8760 に展開します。

#### 5.1 `sensible_heat_profiles`

構造は次のようになっています。

```python
from vtsimnx import schedule as sch

sch.sensible_heat_profiles["LD"]["人体"]["平日"]   # LD・人体・平日 24h [W]
sch.sensible_heat_profiles["LD"]["照明"]["休日"]   # LD・照明・休日 24h [W]
```

#### 5.2 `build_sensible_heat_schedule` と `sensible_heat`

```python
from vtsimnx import schedule as sch

sh_8760 = sch.build_sensible_heat_schedule()
LD_heat_profiles = sh_8760["LD"]    # {用途名: 8760 Series} の dict

# 互換性のため、モジュール import 時点で sch.sensible_heat も生成済み
LD_heat_legacy = sch.sensible_heat["LD"]
```

`build_sensible_heat_schedule` は、`holiday` を見ながら 24h プロファイルを 8760h に展開し、  
部屋ごとの「用途別発熱（人体 / 照明 / 機器など）」の 8760 シリーズを返します。

---

### 6. 潜熱（水分発生）スケジュール（`latent_moisture`）

`schedule/latent_moisture.py` は、潜熱に対応する **水分発生量 [kg/h]** の 24h プロファイルを定義し、8760 に展開します。

- 現状の対象:
  - 人体: 顕熱スケジュールと同じ「在室プロファイル」を用い、**56 W/人** を水分発生量に換算（蒸発潜熱で割り戻し）。
  - 台所機器: 最大 50 g/h (=0.05 kg/h) を上限として、指定の時間帯プロファイル [%] を乗算。

#### 6.1 `latent_moisture_profiles`

構造は顕熱スケジュールとほぼ同じですが、値の単位が **kg/h** になっています。

```python
from vtsimnx import schedule as sch

sch.latent_moisture_profiles["LD"]["人体"]["平日"]   # LD・人体・平日 24h [kg/h]
sch.latent_moisture_profiles["台所"]["機器"]["休日"] # 台所・機器・休日 24h [kg/h]
```

- 人体プロファイルは、`sensible_heat_profiles["部屋"]["人体"][...]` を 63 W/人 で割って人数に戻し、
  さらに `56 W/人` を蒸発潜熱で割り戻して **kg/h/人** に換算したものを掛けています。
- 台所機器プロファイルは、与えられた [%] を `0.05 kg/h` に掛けたものです。

#### 6.2 `build_latent_moisture_schedule` と `latent_moisture`

```python
from vtsimnx import schedule as sch

lm_8760 = sch.build_latent_moisture_schedule()
LD_lm_profiles = lm_8760["LD"]        # {"人体": 8760 Series(kg/h)} の dict
KITCHEN_lm = lm_8760["台所"]["機器"]  # 台所機器からの 8760h 水分発生量 [kg/h]

# 互換性のため、モジュール import 時点で sch.latent_moisture も生成済み
LD_lm_legacy = sch.latent_moisture["LD"]
```

`build_latent_moisture_schedule` は、`holiday` を見ながら 24h プロファイルを 8760h に展開し、  
部屋ごとの「用途別水分発生（人体 / 機器など）」の 8760 シリーズを返します。

---

### 7. カスタマイズの入り口

- **休日パターンを変えたい**: `holiday` を自前で作って `build_*` 関数に渡す。
- **期間区分を変えたい**: `period_x` 相当の 365 要素配列を自分で用意し、`make_8760_data` の第1引数に渡す。
- **プロファイルを変えたい**: `ac_mode_profiles` / `pre_tmp_profiles` / `vent_profiles` / `sensible_heat_profiles` を上書きしてから `build_*` を呼び直す。

```python
import vtsimnx as vt

# 例: LD の暖房設定温度プロファイルを 22℃ に変更して再生成
vt.schedule.pre_tmp_profiles["LD"]["暖房"]["平日"] = [22.0] * 24
vt.schedule.pre_tmp_profiles["LD"]["暖房"]["休日"] = [22.0] * 24
custom_pre_tmp = vt.schedule.build_pre_tmp()
```

---

### 8. このドキュメントの対象範囲

`vtsimnx` 側では、

- スケジュールの**定義**（休日/季節区分/24h プロファイル）
- それを 8760 に展開する**仕組み**（`make_8760_data` / `make_8760_by_holiday`）

を中心に説明しています。

実際に `vt.run_calc` の `input_data` にどう埋め込むかは、次を起点に確認してください。

- 利用者向け最短導線: `builder_input_quickstart.md`
- ノード/ブランチの早見表: `node_branch_schema.md`
- 空調制御の詳細仕様（必要時のみ）: `engine/docs/aircon_control_overview.md`

