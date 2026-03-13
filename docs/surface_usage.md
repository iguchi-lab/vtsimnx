## `surfaces` 入力の実装ガイド

このドキュメントでは、`vtsimnx` で使う **`surfaces` 入力** と、サンプル入力構築における表面定義の考え方をまとめます。

- 物理背景（外皮伝熱・窓の日射熱取得）は `building_environment_engineering_basics.md` を参照してください。
- API 側の RC/応答係数法・builder の展開ロジックは `engine/docs/thermal_rc.md` / `engine/docs/thermal_response_factor.md` / `engine/docs/builder_json.md` を参照してください。

---

### 1. `surfaces` の役割

`surfaces` は「2つのノードの間にある壁・床・窓などの**面要素**」を表します。

- 入力 JSON では `surfaces: []` として与えます。
- builder/API 側で RC/CTF などのネットワークに展開され、最終的には `thermal_branches` + 追加ノードに変換されます。

入力構築時の最終イメージ:

- `nodes`: 室や床下・屋根などの**空気ノード**
- `surfaces`: `"<室A>-><室B>||<面ID>"` という `key` で、**どの室とどの室/外部を結ぶ面か**を指定

---

### 2. 表面カタログと `surfaces` の関係

一般的には、まず「表面カタログ」として `layers` / `surface` を定義し、その後で `surfaces` に面積と接続先を展開します。

#### 2.1 材料テーブルと層定義

```python
import vtsimnx as vt
materials = vt.materials  # 材料物性テーブル（熱伝導率 λ, 比熱 c, 密度 ρ など）

layers = {
    "外壁_一般部": [
        {"key": "木片セメント板",                   **materials["木片セメント板"],                  "t": 0.015},
        {"key": "合板",                             **materials["合板"],                            "t": 0.012},
        {"key": "住宅用グラスウール断熱材16K相当",  **materials["住宅用グラスウール断熱材16K相当"], "t": 0.076},
        {"key": "中空層",                           **materials["中空層(1cm以上)"],                 "t": 1.000},
        {"key": "せっこうボード",                   **materials["せっこうボード"],                  "t": 0.0095},
    ],
    # ... 他の部位（基礎外壁・外皮床・天井・間仕切壁など）
}
```

ここでは:

- `t`: 各層の厚さ [m]
- `materials[...]`: 物性（λ, ρ, c 等）が dict で展開される

`vt.materials` の 1 要素は、概ね次のような内容です（例: 合板）。

```python
import vtsimnx as vt

name = "合板"
print(name, vt.materials[name])
# 出力イメージ:
# 合板 {'lambda': 0.13, 'rho': 550.0, 'cp': 1210.0, ...}
```

典型的なキー:

- `lambda` または `k`: 熱伝導率 [W/(m·K)]
- `rho`: 密度 [kg/m³]
- `cp`: 比熱 [J/(kg·K)]
- 必要に応じて放射率・比透過率など（モデルが参照する分だけ）

**カスタム材料を作りたい場合**:

```python
import copy
import vtsimnx as vt

materials = copy.deepcopy(vt.materials)
materials["自作断熱材"] = {
    "lambda": 0.030,   # W/mK
    "rho": 30.0,       # kg/m3
    "cp": 1400.0,      # J/kgK
}

layers["外壁_自作断熱"] = [
    {"key": "自作断熱材", **materials["自作断熱材"], "t": 0.100},
    {"key": "せっこうボード", **materials["せっこうボード"], "t": 0.0125},
]
```

- ライブラリ同梱の `vt.materials` を直接書き換えるより、上記のように **ローカルコピー (`materials`) を作って上書きする** と安全です。
- 厚さ `t` の単位は常に **メートル [m]** で揃えてください（cm ではない点に注意）。 例: 12mm → `0.012`。

#### 2.2 表面カタログ `surface`（部位ごとのテンプレート）

```python
surface = {
    "E_外壁_一般部": {"part": "wall", "layers": layers["外壁_一般部"], "solar": solar["日射熱取得量（東面）"]},
    "N_外壁_一般部": {"part": "wall", "layers": layers["外壁_一般部"], "solar": solar["日射熱取得量（北面）"]},
    # ...
    "外皮床_一般部": {"part": "floor", "layers": layers["外皮床_一般部"], "alpha-o": 4.4},
    "間仕切壁":      {"part": "wall",  "layers": layers["間仕切壁"],     "alpha-o": 4.4},
    "室内建具":      {"part": "wall",  "u_value": 2.33, "alpha-o": 4.4},
    "S_窓":          {
        "part": "glass",
        "u_value": 4.65,
        "eta": 0.90,
        "solar": solar["日射熱取得量（南面ガラス）"],
        "noctural": 10.0,
    },
}
```

主なキー:

- `part`: `"wall"`, `"floor"`, `"ceiling"`, `"glass"` など部位種別
- `layers`: 壁・床など多層構造を持つ部位の層リスト
- `u_value`: 単層扱い（窓・建具など）の熱貫流率 [W/m²K]
- `eta`: 窓の日射熱取得率
- `alpha-o`: 外気側総合熱伝達率
- `solar`: 日射熱取得量の時系列（`solar_usage.md` で計算）
- `noctural`: 夜間放射に対応する係数（必要に応じて使用）

---

### 3. `surfaces` 配列への展開（部屋間・外部との接続）

最終的な入力では、上記カタログを使って `surfaces` 配列を作ります。

```python
input_data = {
    "simulation": {...},
    "nodes": [...],
    "ventilation_branches": [...],
    "surfaces": [],
}

input_data["surfaces"] = [
    # 室と外部の間の外壁
    {"key": "和室->外部||N_外壁_一般部", **surface["N_外壁_一般部"], "area": 2.18 * 0.83},
    {"key": "和室->外部||N_外壁_熱橋部", **surface["N_外壁_熱橋部"], "area": 2.18 * 0.17},
    # 室と床下の間の床
    {"key": "和室->床下||床_一般部",     **surface["外皮床_一般部"], "area": 16.56 * 0.83},
    {"key": "和室->床下||床_熱橋部",     **surface["外皮床_熱橋部"], "area": 16.56 * 0.17},
    # 室間の間仕切り
    {"key": "和室->ホール||間仕切壁",    **surface["間仕切壁"],      "area": 3.28 - 1.422},
    # 窓
    {"key": "和室->外部||S_窓",          **surface["S_窓"],          "area": 4.59},
    # ...
]
```

`key` のルール:

- 形式: `"<室A>-><室B>||<面ID>"`（`||` より右は自由な表面ID）
- 左側の `<室A>`, `<室B>` は `nodes[].key` と一致させる
- 面ID は `surface` カタログのキーと対応づけておくと管理しやすい

`area` は面積 [m²] です。熱橋部などを**比率で分割**しているのは、外皮の一部だけが柱・梁など高熱貫流部であることを表現するためです。

---

### 4. どこまで core/docs で書くか

`vtsimnx` 側（core）は主に「**サンプルとして surface をどう組み立てるか**」を示す位置付けです。

- 材料物性テーブル `vt.materials`
- 日射取得 `vt.solar_gain_by_angles`（`docs/solar_usage.md`）
- 地盤・夜間放射の係数 `solar_to_surface_temp_coeff` / `nocturnal_to_surface_temp_coeff`（`docs/archenv_comfort_nocturnal_wind_usage.md`）

などと組み合わせて、`surfaces` を構成します。

**実運用での surface 入力仕様（builder → solver への展開）や RC/応答係数法の詳細** は、`engine/docs/` 配下に集約しています。

- surface 入力と RC 法: `engine/docs/thermal_rc.md`
- 応答係数法（CTF）: `engine/docs/thermal_response_factor.md`
- builder の `surfaces` 展開ルール: `engine/docs/builder_json.md`

core/docs では「**表面をどう分解し、どのようなパラメータを与えるか**」の例を中心に押さえてください。

