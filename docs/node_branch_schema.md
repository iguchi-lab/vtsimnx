# Nodes / Branches 入力仕様メモ（ユーザー向け）

> このページは `vt.run_calc(base_url, input_data)` で使う `input_data` のうち、`nodes` / `ventilation_branches` / `thermal_branches` を素早く確認するための早見表です。

このドキュメントは、入力JSON（`input_data`）内の **nodes / ventilation_branches / thermal_branches** のキー仕様を、ユーザーが迷わない形でまとめたものです。

## 使い方（最初にここだけ）

まずは次の最小形で `vt.run_calc` を通し、必要な項目を段階的に追加してください。

```json
{
  "builder": {},
  "simulation": {
    "index": {
      "start": "2026-01-01 01:00:00",
      "end": "2026-01-02 00:00:00",
      "timestep": 3600,
      "length": 24
    }
  },
  "nodes": [
    { "key": "外部", "t": 5.0 },
    { "key": "室1", "calc_t": true, "t": 20.0, "v": 30.0, "thermal_mass": 36216.0 }
  ],
  "ventilation_branches": [
    { "key": "外部->室1", "source": "外部", "target": "室1", "type": "fixed_flow", "vol": 0.008333 },
    { "key": "室1->外部", "source": "室1", "target": "外部", "type": "fixed_flow", "vol": 0.008333 }
  ],
  "thermal_branches": [
    { "key": "外部->室1", "source": "外部", "target": "室1", "type": "conductance", "conductance": 50.0 }
  ]
}
```

`vol` の単位は **m3/s** です（上例は約 30 m3/h）。単位正本: [`units.md`](units.md)。
`surfaces` / `aircon` / `heat_source` は、この最小形が動いてから追加するのが安全です。

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
| `calc_x` | 絶対湿度を未知数として解く | bool | 空調ノードは吹出湿度の固定境界としてfalse |
| `calc_c` | 濃度を未知数として解く | bool | 任意 |
| `p` | 圧力 | number \| number[] | Pa |
| `t` | 温度 | number \| number[] | ℃ |
| `x` | 絶対湿度 | number \| number[] | kg/kg' |
| `c` | 濃度 | number \| number[] | - |
| `pre_temp` | エアコン設定温度 | number \| number[] | ℃ |
| `pre_rh` | 空調の理想除湿目標 | number \| number[] | %、0より大きく100以下。下記参照 |
| `v` | 気積 | number | m3 |
| `beta` | 沈着・一次減衰率 | number \| number[] | 1/s |
| `thermal_mass` | builder が展開する総熱容量 | number | J/K、空気分を含めて指定 |
| `moisture_capacity` | builder が展開する付加湿気容量 | number | 既定は J/(kg/kg')。下記参照 |
| `moisture_capacity_unit` | 湿気容量の入力単位 | string | `"J/(kg/kg')"` または `"kg/(kg/kg)"` |

`pre_rh` は `aircon[].pre_rh` でも指定できる。吸込温度から目標絶対湿度を求め、
冷房時の吹出湿度へ適用する。室の相対湿度を直接固定する指定ではない。
境界条件・能力判定との関係は[空調湿度ガイド](aircon_humidity_control.md)を参照する。

### Nodes: 例

```json
{
  "nodes": [
    { "key": "外部", "t": [5.0, 5.1, 5.2] },
    { "key": "室1", "calc_t": true, "t": 20.0, "v": 30.0, "thermal_mass": 36216.0 }
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
| `type` | ブランチタイプ | string | 例: `simple_opening`, `gap`, `fan`, `fixed_flow`, `pressure_loss` |
| `subtype` | サブタイプ | string | 任意（空文字など） |
| `h_from` | 出発点高さ | number | m（各ノードの圧力基準面からの高さ。基準面を統一する） |
| `h_to` | 到達点高さ | number | m（各ノードの圧力基準面からの高さ。基準面を統一する） |
| `enable` | 有効フラグ | bool \| bool[] | 任意 |
| `comment` | コメント | string | 任意 |
| `alpha` | 流量係数 | number | - |
| `area` | 面積 | number | m2 |
| `a` | 隙間流量係数 | number | m3/(s·Pa^(1/n)) |
| `n` | 隙間の指数（式では逆数を使用） | number | - |
| `p_max` | 最大静圧 | number | Pa |
| `q_max` | 最大風量 | number | **m3/s**（単位正本: [`units.md`](units.md)） |
| `p1` | 点の静圧 | number | Pa |
| `q1` | 点の風量 | number | **m3/s** |
| `vol` | 風量 | number \| number[] | **m3/s**（builder も換算しない。例: 30 m3/h → `30/3600`） |
| `k_total` | 圧損係数（合成） | number | `pressure_loss` 用 |
| `friction_factor` | 摩擦係数 λ | number | `pressure_loss` 用 |
| `length` | 要素長 | number | m, `pressure_loss` 用 |
| `diameter` | 水力直径 | number | m, `pressure_loss` 用 |
| `zeta_total` | 局所損失係数合計 | number | `pressure_loss` 用（任意） |
| `eta` | 除塵効率 | number | - |
| `humidity_generation` | 発湿源 | number \| number[] | kg/s（正本: [`units.md`](units.md)） |
| `dust_generation` | 発塵源 | number \| number[] | モデル依存（濃度単位に合わせる） |

### Ventilation branches: 例

```json
{
  "ventilation_branches": [
    { "key": "外部->室1", "source": "外部", "target": "室1", "type": "fixed_flow", "vol": 0.008333 }
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
| `u_value` | 面積当たり熱コンダクタンス | number | W/(m2・K)、`conductance = u_value * area` |
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

## 物理的な意味と符号

- 換気流量は `source → target` を正とする。負値では上流・下流が入れ替わる。`gap` は `Q = sign(Δp) * a * abs(Δp)**(1/n)` であり、`a` は開口率ではない。
- 圧力計算は `calc_p=true` の節点で**体積流量**収支を解く。温度依存密度を用いた厳密な可変密度質量保存とは区別する。固定流量だけの室でも、利用者が給排気の収支をそろえる。
- `v` だけでは熱容量 branch は生成されない。上例の `thermal_mass=36216` J/K は `1.2 * 1006 * 30` の空気熱容量である。家具等を含める場合は総量を指定する。
- builder の `moisture_capacity` は材料側の追加節点へ展開される。既定単位の値は `2.5e6` J/kg で除して内部容量へ変換する。solver_config の同名フィールドは内部単位であり、raw 入力と混同しない（[単位系](units.md)）。
- 伝導出力は `G*(T_source-T_target)`。移流出力は `ρ*cp*Q*(T_source-T_target)`（湿り空気モードではエンタルピー差）で、下流節点への温度差に基づく寄与である。伝導と同じ反対称の節点流入・流出として単純に加算しない。
- `heat_generation` と `humidity_generation` は target への生成率。濃度 `c` は発生率と整合する濃度基準を選ぶ。

## よくあるミス

- **`source` / `target` に存在しないノード名**を入れる
- **配列長がシミュレーション長と合っていない**（例: 8760 なのに 24 要素）
- **単位誤り**（`vol` / `q_max` / `q1` を m3/h のまま入れる等。正本は [`units.md`](units.md)）


