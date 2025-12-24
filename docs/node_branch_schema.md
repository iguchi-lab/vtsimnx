# Nodes / Branches 入力仕様メモ（ユーザー向け）

このドキュメントは、`config`（入力JSON）内の **nodes / ventilation_branches / thermal_branches** のキー仕様を、ユーザーが迷わない形でまとめたものです。

## 共通ルール

- **`key` は識別子**: 同一セクション内で原則ユニークにしてください。重複する場合は末尾にナンバリング（例: `A->B(2)`）などで区別します。
- **時系列の指定**: 多くの値は「定数」または「時系列（配列）」で指定できます。
  - **定数**: `number`（例: `t: 20.0`）
  - **時系列**: `number[]`（例: `t: [20.0, 20.1, ...]`）
  - 配列長は通常 **`simulation.index.length`（または同等のシミュレーション長）**に合わせます。
- **単位は明示**: 下の表の単位に合わせて入力してください。

## Nodes（ノード）

ノードは「室」「外部」「容量」「表面」「空調」など、状態量（圧力/温度/湿度/濃度）を持つ点です。

### Nodes: キー一覧

| key | 意味 | 型 | 単位/備考 |
|---|---|---:|---|
| `key` | ノード名 | string | 必須 |
| `type` | ノードタイプ | string | 例: `normal`, `layer`, `capacity`, `aircon` |
| `subtype` | サブタイプ | string | 例: `"surface"`, `"internal"` |
| `ref_node` | 参照ノード | string | 参照先の `nodes[].key` |
| `comment` | コメント | string | 任意 |
| `calc_p` | 圧力を未知数として解く | bool | 任意 |
| `calc_t` | 温度を未知数として解く | bool | 任意 |
| `calc_x` | 絶対湿度を未知数として解く | bool | 任意 |
| `calc_c` | 濃度を未知数として解く | bool | 任意 |
| `p` | 圧力 | number \| number[] | Pa |
| `t` | 温度 | number \| number[] | ℃ |
| `x` | 絶対湿度 | number \| number[] | kg/kg' |
| `c` | 濃度 | number \| number[] | - |
| `pre_temp` | エアコン設定温度 | number \| number[] | ℃ |
| `v` | 気積 | number | m3 |
| `beta` | 沈着係数 | number | 1/s |

### Nodes: 例

```json
{
  "nodes": [
    { "key": "外部", "t": [5.0, 5.1, 5.2] },
    { "key": "室1", "calc_t": true, "v": 30.0 }
  ]
}
```

## Ventilation branches（換気ブランチ）

換気ブランチは、空気の流れ（風量）や圧力差に関する接続です。

### Ventilation branches: キー一覧

| key | 意味 | 型 | 単位/備考 |
|---|---|---:|---|
| `key` | ブランチ名 | string | 必須（重複時はナンバリング推奨） |
| `source` | ソースノード | string | `nodes[].key` |
| `target` | ターゲットノード | string | `nodes[].key` |
| `type` | ブランチタイプ | string | 例: `simple_opening`, `gap`, `fan`, `fixed_flow` |
| `subtype` | サブタイプ | string | 任意（空文字など） |
| `h_from` | 出発点高さ | number | m（仕様に合わせて運用） |
| `h_to` | 到達点高さ | number | m（仕様に合わせて運用） |
| `enable` | 有効フラグ | bool \| bool[] | 任意 |
| `comment` | コメント | string | 任意 |
| `alpha` | 有効開口率 | number | - |
| `area` | 面積 | number | m2 |
| `a` | 開口率 | number | - |
| `n` | 隙間係数 | number | - |
| `p_max` | 最大静圧 | number | Pa |
| `q_max` | 最大風量 | number | m3/h |
| `p1` | 点の静圧 | number | Pa |
| `q1` | 点の風量 | number | m3/h |
| `vol` | 風量 | number \| number[] | m3/h |
| `eta` | 除塵効率 | number | - |
| `humidity_generation` | 発湿源 | number \| number[] | g/s（※元メモ表記: `humitidy_generation` は誤字の可能性） |
| `dust_generation` | 発塵源 | number \| number[] | g/s |

### Ventilation branches: 例

```json
{
  "ventilation_branches": [
    { "key": "外部->室1", "source": "外部", "target": "室1", "type": "fixed_flow", "vol": 30.0 }
  ]
}
```

## Thermal branches（熱ブランチ）

熱ブランチは、熱の伝達（コンダクタンス）や発熱などの接続です。

### Thermal branches: キー一覧

| key | 意味 | 型 | 単位/備考 |
|---|---|---:|---|
| `key` | ブランチ名 | string | 必須（重複時はナンバリング推奨） |
| `source` | ソースノード | string | `nodes[].key` |
| `target` | ターゲットノード | string | `nodes[].key` |
| `type` | ブランチタイプ | string | 例: `conductance`, `heat_generation` |
| `subtype` | サブタイプ | string | 例: `convection`, `conduction`, `radiation`, `solar_gain` |
| `enable` | 有効フラグ | bool \| bool[] | 任意 |
| `comment` | コメント | string | 任意 |
| `conductance` | コンダクタンス | number | W/K |
| `u_value` | U値 | number | W/(m2・K) |
| `area` | 面積 | number | m2 |
| `heat_generation` | 発熱源 | number \| number[] | W |

### Thermal branches: 例

```json
{
  "thermal_branches": [
    { "key": "外部->室1", "source": "外部", "target": "室1", "type": "conductance", "conductance": 50.0 }
  ]
}
```

## よくあるミス

- **`source` / `target` に存在しないノード名**を入れる
- **配列長がシミュレーション長と合っていない**（例: 8760 なのに 24 要素）
- **単位混在**（`vol` を m3/s で入れてしまう等）


