# `surfaces` 入力と表面熱収支

`surfaces` は、二つの節点間にある壁・床・天井・窓を記述する builder 入力である。builder が表面・層境界の温度節点、伝導・対流・放射・容量・発熱の枝へ展開し、solver が熱収支を解く。入力全体は [クイックスタート](builder_input_quickstart.md)、単位は [units.md](units.md) を参照する。

## 1. 接続方向と層の順序

`key="室A->室B||面ID"` の `室A`、`室B` は既存の `nodes[].key` と一致させる。`||` 以降は識別用コメントである。層は **室A側から室B側へ** 並べる。外壁を `室->外部` と書く場合、内装材が先、外装材が最後になる。

`part` は室A側の部位を表す。`floor` の反対側は `ceiling`、`ceiling` の反対側は `floor` として扱う。室間の同じ壁を両方向に二重定義しない。柱・断熱部を面積比で分ける方法は並列一次元伝熱の近似であり、二次元熱橋を直接解くものではない。

## 2. 材料・中空層を定義する

```python
import vtsimnx as vt

materials = vt.materials
layers = {
    "外壁_一般部": [  # 室内 → 外部
        {"key": "せっこうボード", **materials["せっこうボード"], "t": 0.0095},
        {"key": "中空層", "air_layer": True,
         "thermal_resistance": 0.09, "t": 0.020},
        {"key": "断熱材", **materials["住宅用グラスウール断熱材16K相当"], "t": 0.076},
        {"key": "合板", **materials["合板"], "t": 0.012},
        {"key": "木片セメント板", **materials["木片セメント板"], "t": 0.015},
    ]
}
```

上例の中空層厚さ・熱抵抗は説明用の仮定値であり、実際の構法に合わせて指定する。

| キー | 意味 | 単位 |
|---|---|---|
| `lambda` | 熱伝導率 | W/(m·K) |
| `v_capa` | 体積熱容量 | J/(m³·K) |
| `t` | 実際の層厚さ | m |
| `air_layer` | 中空層として展開する指定 | bool |
| `thermal_resistance` | 中空層の面積当たり熱抵抗 | m²·K/W |

通常層では `G=A*lambda/t`、`C=A*v_capa*t` とし、RC 展開では容量を層両端へ半分ずつ配分する。中空層は `G=A/thermal_resistance` とし、容量には実厚さと空気の体積熱容量（既定1298 J/(m³·K)）を用いる。熱抵抗を表すために仮想厚さ1 mを入れると容量の解釈を誤るため、`air_layer` による明示指定を用いる。

`vt.materials` は読み取り専用である。独自物性は可変コピーへ追加する。

```python
from vtsimnx.materials import copy_materials, get_material

materials_custom = copy_materials()
materials_custom["自作断熱材"] = {"lambda": 0.030, "v_capa": 42000.0}
gypsum = get_material("せっこうボード")
```

材料テーブルの名称だけで、温湿度依存性や任意製品の性能が保証されるわけではない。研究計算では採用値と出典を記録する。

## 3. 表面熱伝達率と `u_value`

| キー | 現行実装での意味 | 既定値 |
|---|---|---|
| `alpha_i` | 室A空気と室A側表面を結ぶ係数 [W/(m²·K)] | 4.4 |
| `alpha_o` | 室B側表面と室B空気を結ぶ係数 [W/(m²·K)] | 20.3 |
| `u_value` | 層構成省略時の**表面間**コンダクタンス/面積 [W/(m²·K)] | 必要時に指定 |
| `area` | 表面積 [m²] | 必須 |

正しいキーは `alpha_i` / `alpha_o` である。ハイフン表記 `alpha-o` は対応していない。室間壁の室B側にも室内条件に応じた `alpha_o` を指定する。

**現行の `surfaces.u_value` は、表面抵抗を含む部位全体の熱貫流率とは異なる。** `_process_surface_u_value` は `A*alpha_i`、`A*u_value`、`A*alpha_o` の3枝を直列に生成する。放射等の並列経路を除いた定常等価値は次式である。

$$
U_{\mathrm{eq}}=\left(\frac1{\alpha_i}+\frac1{u_{\mathrm{value}}}+\frac1{\alpha_o}\right)^{-1}
$$

例えば `u_value=0.5`、`alpha_i=4.4`、`alpha_o=20.3` なら `U_eq≈0.4393` W/(m²·K) となる。カタログU値をそのまま指定せず、そのU値に含まれる表面抵抗とモデルの放射経路を確認する。表面温度を不要とし、部位全体の `U*A` のみを表したい場合は、室間の `thermal_branches.conductance` として明示する方法がある。

室内長波放射は既定で有効であり、同じ室に接する表面間に `G_ij=4.7*A_i*A_j/sum(A)` を加える。これは一定係数と面積配分による近似で、形状から求めた厳密な形態係数・T⁴の放射交換ではない。4.7には既定の放射率の効果が含まれる。室内側に総合熱伝達率を指定して放射経路も加える場合は、二重計上を確認する。

## 4. 日射・夜間放射

| キー | 用途・符号 |
|---|---|
| `solar` | 面の日射照度系列 [W/m²] |
| `eta` | 不透明面の短波吸収率（既定0.8）。ガラスでは未指定 `SCR` の代替にも使われる |
| `SCR`, `SCC` | ガラス日射の表面配分係数・室空気配分係数 |
| `nocturnal` | 外表面からの放射損失を正とした系列 [W/m²] |
| `night_radiation` | `nocturnal` の互換キー |
| `epsilon` | 夜間放射に掛ける長波放射率（既定0.9） |

`noctural` は誤記で、夜間放射枝を生成しない。`nocturnal` は係数ではなく放射照度である。外壁の発熱は `A*eta*solar`、夜間放射は `-A*epsilon*nocturnal` となり、いずれも室B側表面へ加える。入力が既に吸収率・放射率を含む場合は、再度掛けない設定にする。

`solar_gain_by_angles(glass=False)` は壁への入射日射、`glass=True` はガラスの角度・透過補正後の値である。後者を `solar` に渡す場合、builder はさらに次の配分を行う。

- `A*solar*SCR` の50%を床、50%を壁・天井へ、各群の面積比で配分する。
- 受熱面ごとの `eta`（既定0.8）をさらに掛ける。
- `A*solar*SCC` は室空気へ加える。

この実装は短波の多重反射を追跡しない。床や壁・天井が存在しない群の配分は再配分されず、受熱面の未吸収分も追跡されない。窓の日射熱取得率一つだけで全処理を表すと考えず、各係数の基準と、生成された枝の合計熱流を確認する。

## 5. `surfaces` への展開例

以下は定数日射を用いた入力断片である。室・外部の節点と解析期間は [クイックスタート](builder_input_quickstart.md) に従って別途定義する。

```python
surface = {
    "外壁": {
        "part": "wall", "layers": layers["外壁_一般部"],
        "alpha_i": 4.4, "alpha_o": 20.3,
        "solar": 200.0, "eta": 0.8,
        "nocturnal": 40.0, "epsilon": 0.9,
    },
    "窓": {
        "part": "glass", "u_value": 5.0,  # 表面間の係数。説明用の仮定値
        "alpha_i": 4.4, "alpha_o": 20.3,
    },
}

surfaces = [
    {"key": "室1->外部||外壁", **surface["外壁"], "area": 10.0},
    {"key": "室1->外部||窓", **surface["窓"], "area": 2.0},
]
```

地盤温度関数の `solar_to_surface_temp_coeff` / `nocturnal_to_surface_temp_coeff` は、地表の相当外気温を作るための係数であり、`surfaces` の入力キーではない。同一の放射作用を相当外気温と表面発熱の両方に入れない。

## 6. 実装・理論への参照

- [表面層の実装](../engine/app/builder/surface_layers.py)
- [日射・夜間放射の実装](../engine/app/builder/surface_solar.py)
- [室内放射の実装](../engine/app/builder/surface_radiation.py)
- [RC法](../engine/docs/thermal_rc.md)、[応答係数法](../engine/docs/thermal_response_factor.md)
- [builder入力仕様](../engine/docs/builder_json.md)

## 応答係数法を使う場合

`layer_method="response"` の式、係数生成、RC法との違いと履歴初期化は
[応答係数法ガイド](response_factor_method.md)を参照してください。
