# 入力 JSON の組み立て

対象版: **v1.8.1**（`GET /version` または `pyproject.toml`）。人間も AI も、入力を書くときはこのページから始めてください。

次に読むもの: [クイックスタート](builder_input_quickstart.md)（実行コード）、[早見表](node_branch_schema.md)（キー一覧）、[単位](units.md)。項目の定義が必要なときだけ [builder 入力正本](../engine/docs/builder_json.md) に進みます。

## 1. 何をモデルにするか

計算対象は回路網です。建物そのものをそのまま書くのではなく、状態を持つ点と、その間の流れを書きます。

| 書くもの | 意味 | 例 |
|---|---|---|
| ノード `nodes` | 温度・圧力・湿度などを持つ点 | 室、外部、後から生成される壁層や空調機 |
| 換気枝 `ventilation_branches` | 空気の移動 | 給気、排気、開口、ファン |
| 熱枝 `thermal_branches` | 熱の移動や発生 | 貫流、発熱 |
| 面 `surfaces` | 壁・床・窓 | builder が層ノードと熱枝に展開する |
| 空調 `aircon` | 機器と制御対象 | builder が空調ノードと送風枝を足す |
| 発熱・発湿 | 室内の発生量 | builder が枝に変換する |

自分が書く JSON を **raw 入力**、builder が展開したあとを **計算用 JSON** と呼びます。C++ ソルバが読むのは後者です。壁や空調を手でノードに展開する必要はありません。

```text
raw 入力（この文書の対象）
  → builder（展開・検証）
  → 計算用 JSON
  → ソルバ
```

`calc_t` / `calc_p` / `calc_x` / `calc_c` は「その量を未知数として解く」印です。外部気温のように既知の境界には付けません。室の温度を解くなら `calc_t: true` です。

## 2. 単位（間違えると静かに壊れます）

| 量 | 単位 | よくある誤り |
|---|---|---|
| 風量 `vol` | m³/s | m³/h のまま書く（30 m³/h = 0.008333 m³/s） |
| 発熱 `generation_rate` | W | kW で書く |
| 発湿 `generation_rate` | kg/s | g/h のまま書く |
| 空調能力 `ac_spec.Q` | kW | W で書く |
| 温度 | °C | |
| 熱容量 `thermal_mass` | J/K | |

一覧の正本は [units.md](units.md) です。OpenAPI の各フィールドにも `unit` が付きます。

## 3. 標準の最小モデル

一室と外部、給気と排気、熱の逃げ、空気の熱容量がある形が出発点です。実ファイル: [`examples/ai_input/minimal_room.json`](../examples/ai_input/minimal_room.json)。

- **給気と排気の両方**を書きます。給気だけだと計算は通ることがありますが、室の空気収支としては不自然です。
- 室には `v`（気積 m³）と `thermal_mass`（J/K、空気分を含む）を付けます。`v` だけでは家具等の容量にはなりません。
- 外部は既知温度、室の `t` は初期値です。
- 時系列を使うなら配列長を `simulation.index.length` に合わせます。`end = start + (length-1)*timestep` です。
- `simulation.coupling` と `simulation.iteration` は v1.8.1 以降、検証したうえで計算に渡ります。

組み立てる順は、時刻 → ノード名 → 換気と熱の接続 → 壁 → 空調・発熱・発湿、です。存在しないノード名を参照するとそこで止まります。

## 4. 用途別の雛形

[`examples/ai_input/`](../examples/ai_input/) に、上の最小形へ足していく例があります。

| ファイル | 足すもの |
|---|---|
| [`minimal_room.json`](../examples/ai_input/minimal_room.json) | 一室・給排気・熱コンダクタンス |
| [`heat_source.json`](../examples/ai_input/heat_source.json) | 発熱 [W] |
| [`humidity_source.json`](../examples/ai_input/humidity_source.json) | 発湿 [kg/s] と `calc_x` |
| [`surfaces_wall.json`](../examples/ai_input/surfaces_wall.json) | 壁1面 |
| [`aircon_rac.json`](../examples/ai_input/aircon_rac.json) | RAC 空調 |

Python から動かす最小コードは [クイックスタート](builder_input_quickstart.md) と [`examples/run_calc_minimal.py`](../examples/run_calc_minimal.py) です。中身は同じ一室モデルです。

## 5. 本計算の前に検査する

ソルバを回さずに直せます。要求ボディは `POST /run` と同じです。認証も `/runs` と同じく `X-API-Key`（未設定のローカルでは不要）です。

```text
OpenAPI と雛形を読む
  → JSON を書く
  → POST /validate（unknown_keys=error）
  → path を直す
  → POST /build（展開後の件数を見る）
  → POST /runs（本計算）
```

機械可読な型・必須項目は `GET /openapi.json` の `RawSimConfig`、`AcSpecModel`、`HeatSourceModel`、`HumiditySourceModel` です。文書と違う場合は OpenAPI と実装を優先します。

`POST /validate` の応答:

```json
{
  "ok": true,
  "issues": [
    {
      "code": "fixed_flow_imbalance",
      "message": "...",
      "path": ["nodes", 0],
      "severity": "warning",
      "hint": "..."
    }
  ]
}
```

- `path` は常に配列です。例: `["heat_source", 0, "generation_rate"]`。
- HTTP 200 でも `ok=false` なら error があります。リクエスト自体が型不正なら 422 です。
- error は型・必須・未知キー・参照切れです。warning は「計算はできるが、意図とずれている可能性がある」印です。

| code | 意味 |
|---|---|
| `unknown_field` | 未定義キー。`unknown_keys=error` のとき error |
| `fixed_flow_imbalance` | 固定流量の給排気が室で釣り合っていない |
| `missing_room_capacity` | `calc_t` なのに `v` も `thermal_mass` もない |
| `humidity_source_without_calc_x` | 発湿先に `calc_x` がない |
| `isolated_node` | どの枝・壁・空調からもつながっていない |
| `aircon_missing_field` | 空調の `set` / `outside` / `pre_temp` / `mode` が欠けている |
| `no_calc_flag` | 展開後に解く量が一つもない |

`POST /build` は展開後 JSON と `stats`（ノード数・枝数）を返します。`ok=false` のとき `config` は null です。

未知キーを残したまま本計算したくない場合は `"unknown_keys": "error"` を付けます。`ac_spec`・`heat_source`・`humidity_source` の内側も対象です。仮定した値は JSON の未知キーに書かず、生成ログなど JSON の外に残してください。

## 6. まだ自動では足りないこと

「東京の2階建住宅、全館空調」のような建物条件から、寸法・隣接・8760時間の時系列まで一気に展開する共通処理はありません。不足は仮定し、仮定は記録してください。MCP は連携先が決まってから足します。

## 7. 設備を足すとき

| 対象 | ガイド | 正本 |
|---|---|---|
| 壁・窓 | [surface_usage.md](surface_usage.md) | builder_json § surfaces |
| 空調 | [aircon_model_and_control.md](aircon_model_and_control.md) | [aircon_spec_reference.md](../engine/docs/aircon_spec_reference.md) |
| 全館空調 | [duct_central_airflow_control.md](duct_central_airflow_control.md) | 同上 |
| 熱交換換気 | [heat_recovery_vent.md](heat_recovery_vent.md) | builder_json § HRV |
| HTTP 全体 | — | [api_reference.md](../engine/docs/api_reference.md) |
