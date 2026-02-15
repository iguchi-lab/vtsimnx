### builder入力JSON（raw_config）の作り方（VTSimNX）

このドキュメントは `app/builder` が受け取る **生JSON（raw_config）** の書き方を、実装に沿って整理したものです。

builder は raw_config を **正規化/展開**して、C++ solver が読める形式（`simulation/nodes/ventilation_branches/thermal_branches`）へ変換します。

シミュレーション全体の処理順（概略）は `docs/simulation_overview.md` を参照してください。

---

### 1. 全体像（raw → solver用）

処理の流れ（概念）:

- `parse_all(raw)`
  - `simulation` / `nodes` / `ventilation_branches` / `thermal_branches` / `surfaces` / `aircon` を読み取る
  - `nodes` には自動で `{"key":"void"}` が先頭に追加される
  - `key` のコメント記法（`||`）やチェーン展開（`A->B->C`）を処理
- `process_surfaces(...)`（任意）
  - `surfaces` から層ノード＋熱ブランチ（日射/放射含む）を生成して `nodes/thermal_branches` に追加
- `process_aircons(...)`（任意）
  - `aircon` から aircon ノード＋換気ブランチを生成して `nodes/ventilation_branches` に追加
- `process_capacities(...)`（任意）
  - `thermal_mass` を持つノードから capacity ノード/ブランチを生成し、元ノードの `thermal_mass` を削除
- `validate_dict_with_warnings(...)`
  - **未知フィールドの削除**、**type推定**、**必須フィールドチェック**、**重複keyのリネーム**等を行う

---

### 2. トップレベル構造（raw_config）

builder が期待するトップレベルは概ね以下です:

- `builder`（任意）: builderオプション
- `simulation`（必須）: 時間刻み等
- `nodes`（必須）: 室/外気などのノード定義
- `ventilation_branches`（必須だが空OK）: 換気ブランチ定義
- `thermal_branches`（必須だが空OK）: 熱ブランチ定義
- `surfaces`（任意）: 壁/床/天井/ガラス等（builderが展開）
- `aircon`（任意）: 空調設定（builderが展開）
- `heat_source`（任意）: 発熱（人体/機器等の顕熱）設定（builderが `thermal_branches` の `heat_generation` に変換）
- `humidity_source`（任意）: 発湿源（加湿器/人体等の水蒸気発生）（builderが `ventilation_branches` の `humidity_generation` に変換）

最小例（表面・空調なし）:

```json
{
  "simulation": {
    "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T01:00:00", "timestep": 3600, "length": 2},
    "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6}
  },
  "nodes": [
    {"key": "室", "type": "normal", "t": 20.0},
    {"key": "外部", "type": "normal", "t": 0.0}
  ],
  "ventilation_branches": [],
  "thermal_branches": []
}
```

---

### 3. `builder` オプション

#### 3.1 `surface_layer_method`（全surface一括の壁モデル指定）

```json
{
  "builder": {
    "surface_layer_method": "rc"
  }
}
```

- `"rc"`: RC法（デフォルト）
- `"response"`: 応答係数法（CTF, `response_conduction`）

互換としてトップレベルに `"surface_layer_method"` を置くことも許可されています。

#### 3.1.1 builder の処理ON/OFF（raw_configだけで再現したい場合）

`builder` では、builder の自動展開処理を **raw_config内でON/OFF**できます。
（API からフラグを渡す場合は、API側の指定が優先されます。）

```json
{
  "builder": {
    "add_surface": true,
    "add_surface_solar": true,
    "add_surface_nocturnal": true,
    "add_surface_radiation": true,
    "add_aircon": true,
    "add_capacity": true
  }
}
```

各フラグの意味:

- `add_surface`: `surfaces` の展開を行うか
- `add_surface_solar`: `surfaces` の日射処理を行うか（`add_surface=true` のときに有効）
- `add_surface_nocturnal`: `surfaces` の夜間放射（長波放射）処理を行うか（`add_surface=true` のときに有効）
- `add_surface_radiation`: `surfaces` の室内放射処理を行うか（`add_surface=true` のときに有効）
- `add_aircon`: `aircon` の展開を行うか
- `add_capacity`: `thermal_mass` から熱容量ノード/ブランチを生成するか

#### 3.2 応答係数の自動生成方法（CTF）

`layer_method="response"` で `surface.response` を省略した場合、builder は `layers` から CTF係数を自動生成します。

```json
{
  "builder": {
    "response_method": "arx_rc",
    "response_terms": 6
  }
}
```

- `response_method`:
  - `"arx_rc"`（既定）: RC連鎖（状態数＝層数）をそのままARX化（次数＝層数）
  - `"modal_expsum"`: 離散系をモード分解して **指数項（λ^k）の和**で近似し、`response_terms` 個に縮約してARX化
- `response_terms`:
  - `"modal_expsum"` のときに効く「項数（次数）」です（未指定なら層数相当）
  - `resp_a_* / resp_b_*` の長さは `response_terms+1`
  - `resp_c_*` の長さは `response_terms`

互換としてトップレベルに `"response_method"` / `"response_terms"` を置くことも許可されています。

---

### 4. `key` の特殊記法（重要）

builder は `key` 文字列にいくつかの記法を持ちます。

#### 4.1 インラインコメント（`||`）

`"A->B||説明"` のように書くと、`key` は `"A->B"` として解釈し、`comment` フィールドへ `"説明"` を格納します。

#### 4.2 ノードの複合キー（`&&`）

`nodes` の `key` は `"A&&B"` のように書くと **複数ノードに展開**されます（同一設定を複数ノードに適用したいとき用）。

#### 4.3 ブランチのチェーン展開（`A->B->C`）

`ventilation_branches` と `thermal_branches` の `key` は `"A->B->C"` のように書くと **連続する2本**に分解されます。

- `"A->B->C"` → `"A->B"` と `"B->C"`

#### 4.4 `void` ノードと省略

- `void` ノードは builder が自動追加します（`nodes` に自分で書かない）
- ブランチの端点は空文字を許し、空文字は `void` として扱われます
  - 例: `"->外部"` は `"void->外部"`

---

### 5. 時系列（スカラー or 配列）

builder/validation/solver はフィールドによって **スカラー**または **配列（時系列）**を受けます。

- builder は `numpy.ndarray` / `pandas.Series` を受けた場合、listへ正規化します
- solver側で「配列が短い」場合は、実装により最後の値が使われます（枝/項目による）

実務的には:

- まずスカラーで作る
- 必要な項目だけ配列にする（巨大JSON回避）

---

### 6. `simulation`

主に以下を使います:

- `simulation.index.start/end`: ISO形式推奨
- `simulation.index.timestep`: 秒
- `simulation.index.length`: ステップ数
- `simulation.tolerance.*`: 収束閾値

`calc_flag` は builder が自動設定します（入力で省略してOK）。

補足（濃度/湿度）:

- **`simulation.calc_flag.c`**: 濃度計算（c）の全体ON/OFF
- **`nodes[].calc_c`**: ノード単位の更新ON/OFF（`true` のノードだけが `concentration_c` 出力対象にもなる）
- 同様に、湿度は `simulation.calc_flag.x` と `nodes[].calc_x` の組み合わせで制御します

---

### 7. `nodes`

代表的なフィールド:

- `key`（必須）: ノード名（`void`は禁止）
- `t` / `p` / `x` / `c`（任意）: 初期値または時系列
- `calc_t` / `calc_p` / `calc_x` / `calc_c`（任意）: 計算対象フラグ
- `type`（solver形式では必須）: `"normal"`, `"capacity"`, `"aircon"` など
  - builderの raw_config では省略されることがありますが、solver側の parser では `nodes[].type` を必須として扱います（builder/validation が `normal` を補完する前提）。

濃度計算で使う追加フィールド:

- `beta`（任意）: **沈着係数 \([1/s]\)**（スカラー or 配列）
  - `calc_flag.c=true` かつ `calc_c=true` のノードで、濃度 \(c\) の一次減衰（沈着）として使用します。
  - `beta=0` の場合は沈着なし（流入・流出・生成のみ）

#### 7.1 熱容量の付与（`thermal_mass`）

ノードに `thermal_mass`（\[J/K]）を持たせると、builder が以下を自動生成します:

- 容量ノード: `"{key}_c"`（`type="capacity"`, `calc_t=false`, `ref_node=元ノード`）
- 容量ブランチ: `"{key}_c->{key}"`（`subtype="capacity"`, `conductance=thermal_mass/timestep`）

このとき元ノードの `thermal_mass` は削除されます。

---

### 8. `ventilation_branches`

solver側の parser では **以下が必須**です:

- `key`
- `type`
- `source`
- `target`

builder の raw_config では `key` 記法（`A->B` / `->外部` 等）から `source/target` が補完されることがありますが、
**solver形式の JSON を直接書く場合は必ず明示してください**。

代表:

- `fixed_flow`: `vol`（流量）を指定
- `simple_opening`: `alpha`, `area`
- `gap`: `a`, `n`
- `fan`: `p_max`, `q_max`, `p1`, `q1`

湿度/濃度計算で使う追加フィールド（任意）:

- `humidity_generation`（任意）: **発湿量 \([kg/s]\)**（スカラー or 配列）
  - ブランチの `target` ノードへ水蒸気生成項として加算されます（空気移動が無くても生成項だけ入れたい場合、`void->{room}` の `fixed_flow` に `vol=0` として付与する運用が可能です）
- `dust_generation`（任意）: **発塵量 \([count/s]\)**（スカラー or 配列）
  - ブランチの `target` ノードへ濃度の生成項として加算されます（同様に `vol=0` の `void->{room}` ブランチに付与可能）
- `eta`（任意）: **除去効率 \([-]\)**（スカラー or 配列）
  - 濃度計算で、流入側の寄与に \((1-\eta)\) として適用されます（フィルタ等）
  - 注意: 現状の実装では **濃度（c）のみに適用**されます（湿度xには適用しません）

---

### 9. `thermal_branches`

solver側の parser では **以下が必須**です:

- `key`
- `type`
- `source`
- `target`

builder の raw_config では `key` 記法（`A->B` / `->外部` 等）から `source/target` が補完されることがありますが、
**solver形式の JSON を直接書く場合は必ず明示してください**。

代表:

- `conductance`
  - `conductance`（\[W/K]）を直接指定
  - 互換: `u_value`（\[W/m2/K]）と `area`（\[m2]）があれば `conductance=u_value*area` に正規化
- `heat_generation`
  - `heat_generation`（\[W]、スカラーor配列）
- `response_conduction`（CTF）
  - `area` が必須（係数が per m² のため）
  - 係数の書式は `docs/thermal_response_factor.md` を参照

---

### 9.1 `heat_source`（発熱: 人体/機器など）

`heat_source` は **入力の発熱量（顕熱）**を、solver が理解できる `thermal_branches` の `heat_generation` に変換するための補助入力です。

- **対流成分**: `void->{room}` の `heat_generation`（\[W]）として追加
- **放射成分**: その room に属する `surfaces` がある場合は `void->{surface_node}` へ **面積按分**して追加  
  （`surfaces` が無い場合は `void->{room}` へまとめて追加）

例:

```json
{
  "heat_source": [
    {
      "key": "LD_人体",
      "room": "LD",
      "generation_rate": [100.0, 200.0, 150.0],
      "convection": 0.50,
      "radiation": 0.50
    }
  ]
}
```

フィールド:

- `key`（任意）: ログ/コメント用途（重複しても可）
- `room`（必須）: 対象室ノード（`nodes[].key`）
- `generation_rate`（必須）: 発熱量（\[W]、スカラー or 配列）
- `convection` / `radiation`（任意）: 比率（0..1）
  - 両方未指定: convection=1, radiation=0
  - 片方のみ指定: 残りは 1-指定値

---

### 10. `surfaces`（builder展開）

`surfaces` を書くと、builder が層ノード/熱ブランチ/日射/放射などを生成して `nodes/thermal_branches` に追加します。

代表フィールド:

- `key`（必須）: `"室->外部"` のような2端子
- `part`（必須）: `"wall" | "floor" | "ceiling" | "glass"`
- `area`（必須）
- `layers`（任意）: `lambda`（熱伝導率）, `t`（厚さ）, `v_capa`（体積熱容量）
- `u_value`（任意）: `layers` が無い場合の簡略伝熱
- `alpha_i/alpha_o`（任意）: 内外表面の対流熱伝達率
- `layer_method`（任意）: `"rc"` or `"response"`（未指定なら builder の `surface_layer_method` が入る）

RC/CTFの詳細は以下:

- RC法: `docs/thermal_rc.md`
- 応答係数法: `docs/thermal_response_factor.md`

---

### 11. `aircon`（builder展開）

`aircon` を書くと、builder が aircon ノードと換気ブランチ（2本）を生成して `nodes/ventilation_branches` に追加します。

代表フィールド（概略）:

- `key`（必須）: airconノード名として使われる
- `set`（必須）: 制御対象ノード
- `outside`（必須）: 外気ノード
- `pre_temp`（必須）: 目標温度（スカラー/配列）
- `model`（必須）, `mode`（必須）
- `vol`（任意）: 風量（未指定時は既定値）
- `in/out`（任意）: 吸込/吹出ノード（省略時は `set`）

---

### 11.1 `humidity_source`（発湿源: 加湿器/人体など）

`humidity_source` は raw_config の補助入力で、builder が **`ventilation_branches[].humidity_generation`** に変換します。
（solver 形式の JSON を直接書く場合は `ventilation_branches` に `humidity_generation` を直接書いてOKです。）

例:

```json
{
  "humidity_source": [
    {"key": "加湿器", "target_node": "室A", "generation_rate": [0.0, 0.001, 0.0]}
  ]
}
```

フィールド:

- `key`（任意）: ログ/コメント用途
- `target_node`（必須）: 対象室ノード（`nodes[].key`）
- `generation_rate`（必須）: 発湿量 \([kg/s]\)（スカラー or 配列）

注意:

- `calc_flag.x` が `true` のときに意味があります（湿度計算OFFなら出力にも反映されません）

---

### 13. solver 出力（artifact: schema.json + *.f32.bin）

solver は `artifact_dir` に **schema（キー配列）** と **時系列バイナリ（float32 little-endian）**を出力します。
湿度/濃度の系列は以下です:

- `humidity_x`: 絶対湿度 \(x\)（`calc_x=true` のノード、キー順は昇順）
- `concentration_c`: 濃度 \(c\)（`calc_c=true` のノード、キー順は昇順）


### 12. validationの挙動（知っておくと便利）

- **未知フィールドは警告して削除**（タイプミス検知のため）
- ブランチ `key` が重複した場合:
  - solver側はエラーにします
  - builder validation は `(...01)` のようにリネームして回避します（`source/target`は保持）

#### 12.1 `enable` の既定値（ブランチの有効/無効）

`ventilation_branches[].enable` / `thermal_branches[].enable` は **省略すると既定で有効（true）**です。

- boolean: `true/false`
- array<boolean>: 時系列（短い場合は末尾を使用）


