## SimHeat 比較用入力例（vs_simheat_r14 の要約）

このドキュメントは、`vtsimnx_test/vs_simheat_r14.ipynb` で行っている **SimHeat との比較ケース**のうち、
「VTSimNX に渡している入力 JSON（`input_data`）」の組み立て方を、高レベルでまとめたものです。

ノートブック本体は GitHub 上の次を参照してください。

- `iguchi-lab/vtsimnx_test` リポジトリ: `vs_simheat_r14.ipynb`

ここでは、「どの情報をどう組み合わせて builder 入力を作っているか」に絞って説明します。

最小構成から入りたい場合は `builder_input_quickstart.md` を先に参照してください。

---

### 1. 気象と SimHeat 出力の読み込み

1. `vt.read_hasp("<weather>.has")` で気象データを読み込む。  
   - 列: 外気温・地下温度・水平面日射・夜間放射など。
2. 独自関数 `read_simheat_csv(...)` で SimHeat の CSV 出力（室温、熱流、換気、透過日射、冷暖房負荷など）を読み込む。
   - `Output室温湿度.csv`
   - `Output熱流(和室).csv` など
   - `Output窓透過日射(放射成分).csv`
   - `Output時刻別冷暖房負荷.csv`

これらは **VTSimNX 側の入力には使わず**、主に結果比較のために保持しています。

---

### 2. 日射・夜間放射の計算（solar/nocturnal）

1. `df_i` から直達/拡散日射量 (`dni`/`dhi`) と夜間放射量 (`rn_horizontal`) を取り出す。
2. `vt.solar_gain_by_angles(...)` を複数回呼び出し、次のような列をもつ `solar` DataFrame を作る。
   - `日射熱取得量（南面）`, `日射熱取得量（東面）`, `日射熱取得量（西面）`, `日射熱取得量（北面）`
   - 各方位のガラス用 `日射熱取得量（南面ガラス）` など
   - 屋根面用 `日射熱取得量（南面屋根）` など
3. 必要に応じて遮蔽ポリゴン `shade_polys` を定義し、遮蔽込みの日射取得を計算する（`solar_usage.md` 参照）。

→ これらの列は後で `surface[...]` の `solar` に渡されます。

---

### 3. 部屋の気積と材料・層（`room_volume`, `layers`, `surface`）

1. 各室の気積 [m³] を `room_volume` dict にまとめる。
2. `vt.materials` を使って **層構成 `layers[...]`** を定義する（詳しくは `surface_usage.md` 参照）。
3. それらを元に、部位ごとの「表面カタログ」 `surface[...]` を構成する。
   - 例: 外壁一般部、熱橋部、外皮床、天井、屋根、間仕切り、建具、各窓（方位別）など。

```python
import vtsimnx as vt
materials = vt.materials

layers = {
    "外壁_一般部": [
        {"key": "木片セメント板", **materials["木片セメント板"], "t": 0.015},
        # ...
    ],
    # ...
}

surface = {
    "E_外壁_一般部": {"part": "wall", "layers": layers["外壁_一般部"], "solar": solar["日射熱取得量（東面）"]},
    "S_窓":          {"part": "glass", "u_value": 4.65, "eta": 0.90, "solar": solar["日射熱取得量（南面ガラス）"]},
    # ...
}
```

---

### 4. builder 入力 `input_data` の構成

ノートブック内では、次のような dict を組み立てて `vt.run_calc(base_url, input_data, ...)` に渡しています。

```python
input_data = {
    "builder": {...},
    "simulation": {...},
    "nodes": [...],
    "ventilation_branches": [...],
    "thermal_branches": [...],
    "surfaces": [...],
    "aircon": [...],
    "heat_source": [...],
}
```

#### 4.1 `builder`

```python
input_data["builder"] = {
    # "add_surface_solar": False,
    # "add_surface_nocturnal": False,
    # "add_surface_radiation": False,
    # "add_surface_radiation_exclude_glass": True,
}
```

- 特殊な検証のときだけコメントアウトを外し、日射/夜間放射/長波放射の扱いを切り替えています。
- 通常利用では builder 側の既定値に任せています（詳細は API 側 `builder_json.md` を参照）。

#### 4.2 `simulation`

```python
input_data["simulation"] = {
    "index": {
        "start": "2026-01-01 01:00:00",
        "end":   "2026-01-01 10:00:00",
        "timestep": 3600,
        "length":   10,
    },
    "tolerance": {
        "ventilation": 1e-6,
        "thermal": 1e-6,
        "convergence": 1e-6,
    },
}
```

#### 4.3 `nodes`

```python
input_data["nodes"] = [
    {"key": "外部",      "t": df_i["外気温"]},
    {"key": "地下1m",    "t": df_i["地下1m温度"]},
    {"key": "床下",      "calc_t": True, "thermal_mass": room_volume["床下"] * 12.6 * 1000},
    {"key": "和室",      "calc_t": True, "thermal_mass": room_volume["和室"] * 12.6 * 1000},
    # ... 各室・小屋裏・階間など
]
```

- 室ノードには `thermal_mass` で空気＋内装の熱容量を与えています。
- 温度を未知数とするノードは `calc_t: True`。

#### 4.4 `ventilation_branches`

- 24時間換気: 外部→室→ホール→外部、などの鎖を `vol` [m³/s] で指定。
- 室間換気: LD↔台所、ホール↔2階ホール など。
- 局所換気: `vol: vt.schedule.vol["LD"]` のように **スケジュール換気量**を直接指定。
- 床下・小屋裏換気: `vol: room_volume[...] * 5.0 / 3600` のように換気回数から換算。

#### 4.5 `surfaces`

- `surface_usage.md` で説明したパターンを、SimHeat モデルの実寸・開口寸法に合わせて適用しています。
- 例: 和室の南外壁・窓・床・天井・間仕切りなどをすべて `surfaces` に列挙。

#### 4.6 `aircon` / `heat_source`

- エアコン: `vt.schedule.ac_mode` / `pre_tmp` などのスケジュールを用いて、個別空調を再現。
- 発熱: `vt.schedule.sensible_heat` を元に、室ごとの内部発熱プロファイルを `heat_source` に設定。

（具体的な JSON 形は API 側 `builder_json.md` と `schedule_usage.md` を参照してください。）

---

### 5. 実行と結果取得

最後に、環境変数 `VTSIMNX_API_URL` から API ベースURLを取得し、次のように呼び出します。

```python
from google.colab import userdata

base_url = userdata.get("VTSIMNX_API_URL")
result = vt.run_calc(base_url, input_data, request_output_path="result_vs_simheat.json")
print(result.log)
```

`result` からは `get_series_df("thermal_temperature")` などで各種系列を取り出し、SimHeat 側の出力と比較しています。

---

### 6. このドキュメントの対象範囲

このドキュメントはあくまで「**大規模ケースの入力をどう組み立てているか**」を俯瞰するためのものです。

- 日射 / 夜間放射の詳細: `solar_usage.md`, `archenv_comfort_nocturnal_wind_usage.md`
- スケジュール（空調・換気・発熱）: `schedule_usage.md`
- surface / materials / layers: `surface_usage.md`
- 利用者向けの入力組み立て導線: `builder_input_quickstart.md`, `node_branch_schema.md`
- builder・API 側の厳密 JSON 仕様: `engine/docs/builder_json.md`

