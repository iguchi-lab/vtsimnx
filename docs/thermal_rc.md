### 壁モデル（RC法）入力仕様（VTSimNX）

このドキュメントは、熱計算で **RCネットワーク（層分割）**を使うための入力 JSON の書き方と、builder が生成するノード/ブランチ、単位の約束、精度上の注意点をまとめたものです。

---

### 1. RC法とは（何がネットワークに入るか）

RC法では壁を層で表現し、builder は壁の内部状態を **「層ノード（熱容量）」**として熱回路網に明示的に追加します。

- ノード: `type="layer"` の層ノード（表面/内部を含む）
- ブランチ:
  - 室内側の **対流**（`subtype="convection"`）
  - 壁体の **伝導**（`subtype="conduction"`、層数に応じて複数）
  - 室外側の **対流**（`subtype="convection"`）

---

### 2. surface での指定（builderに任せる）

RCはデフォルトです（`layer_method` を省略すると `"rc"` 扱い）。

#### 2.1 surface個別に layers を与える（推奨）

```json
{
  "surfaces": [
    {
      "key": "室->外部",
      "part": "wall",
      "area": 10.0,
      "layers": [
        {"lambda": 0.8, "t": 0.12, "v_capa": 900000.0},
        {"lambda": 0.04, "t": 0.08, "v_capa": 30000.0}
      ]
    }
  ]
}
```

`layers` を与えると、builder は以下を生成します（例: 2層の場合）:

- 層ノード:
  - 室内側表面ノード（`subtype="surface"`）
  - 内部ノード（`subtype="internal"`）
  - 室外側表面ノード（`subtype="surface"`）
- ブランチ列（直列）:
  - `室 -> 室内側表面`（対流）
  - `室内側表面 -> 内部`（伝導）
  - `内部 -> 室外側表面`（伝導）
  - `室外側表面 -> 外部`（対流）

#### 2.2 layers を使わず u_value で簡略化

`layers` が無い場合、壁体の伝熱は `u_value`（面積あたり）と `area` から **1本の伝導**として表現されます（内部熱容量は持ちません）。

```json
{
  "surfaces": [
    {
      "key": "室->外部",
      "part": "wall",
      "area": 10.0,
      "u_value": 0.5
    }
  ]
}
```

この場合、builder は概ね以下の等価を作ります:

- `conductance = area * u_value`（\[W/K]）
- 熱容量は `a_capacity` を指定しない限り 0（`a_capacity` を与えると簡易的に容量を付与）

#### 2.3 特殊層（中空層・通気層）

RC法では、`layers` 内の各層にフラグを付けて特殊な熱回路を作れます。

- 中空層:
  - `air_layer: true`
  - `thermal_resistance`（互換: `r_value`, `r`）を与えると
    `conductance = area / thermal_resistance` で伝導ブランチを生成
- 通気層:
  - `ventilated_air_layer: true`
  - その層を「両端ノード + 中心ノード」の3ノードとして扱い、
    - 両端-中心: `alpha_c1`, `alpha_c2` の対流ブランチ
    - 両端同士: `alpha_r` の放射ブランチ
    - デフォルト値:
      - `alpha_c1=4.4 [W/m2/K]`
      - `alpha_c2=4.4 [W/m2/K]`
      - `alpha_r=4.7 [W/m2/K]`
  - 中心ノードの熱容量: `area * t * air_v_capa`（省略時 `air_v_capa=1200 [J/m3/K]`）

---

### 3. 単位の約束

RC法では、builder が **面積を含めた値**を network に入れます（solver内部の熱収支が \[W] なので整合します）。

- `conductance`（ブランチ）: \([W/K]\)
  - 例: 対流は `area * alpha`（alphaは \([W/m^2/K]\)）
  - 例: 伝導は `area * lambda / t`（lambdaは \([W/m/K]\), tは \([m]\)）
- `thermal_mass`（層ノード）: \([J/K]\)
  - `area * v_capa * thickness`（v_capaは \([J/m^3/K]\)）

---

### 4. 精度・チューニング上の注意（よくある落とし穴）

- **層分割が粗いと数値拡散が大きい**:
  - RCは離散化モデルなので、分割が粗いほど短周期の変動が過度に減衰しやすいです。
- **timestep 依存**:
  - timestep が大きいと（特に壁の時定数より大きいと）動きが鈍く見えることがあります。
- **対流熱伝達率（alpha）**:
  - `alpha_i`, `alpha_o` の設定は応答に強く効きます。未指定時は builder 既定値が使われます。

---

### 5. 応答係数法（CTF）との関係

CTF（`response_conduction`）の入力仕様は以下を参照してください:

- `docs/thermal_response_factor.md`


