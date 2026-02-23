### シミュレーション全体の概略（VTSimNX）

このドキュメントは、VTSimNX の「入力→builder→solver→出力」の流れと、1タイムステップ内で **何をどの順序で計算しているか**の概要をまとめたものです。

---

### 1. 入力と builder の役割

#### 1.1 入力JSONの2層

- **raw_config（builder入力）**: `app/builder` が受け取る生JSON
  - `surfaces` / `aircon` / `heat_source` / `humidity_source` など「書きやすい入力」を含む
- **solver_config（solver入力）**: C++ solver が読む正規化済みJSON
  - 主要キーは `simulation` / `nodes` / `ventilation_branches` / `thermal_branches`

builder は raw_config を正規化・展開して solver_config を作ります。
（詳細は `docs/builder_json.md`）  
建築環境工学の背景（換気回路網/熱回路網/日射・放射/湿気・濃度）は `docs/theory_basics.md` を参照してください。

---

### 2. solver の計算フラグ（calc_flag）

`simulation.calc_flag` により、計算する物理量をON/OFFします。

- `p`: 圧力（換気回路網）
- `t`: 温度（熱回路網）
- `x`: 絶対湿度（湿気移流＋発湿）
- `c`: 濃度（粒子等、移流＋発塵＋沈着）

注: builder は入力内容に応じて `calc_flag` を自動設定します（未指定でもOK）。

各計算対象の理論的な意味は `docs/theory_basics.md` に整理しています。

---

### 3. 1タイムステップの処理順（概略）

1ステップの大枠は次の通りです（ログや収束判定のため一部ループ構造になっています）。

背景となる連成の考え方（換気と熱、湿気と濃度の役割分担）は `docs/theory_basics.md` の全体像を参照してください。

#### 3.1 反復の外枠

- **圧力＋熱（連成）を反復して収束**
- **（収束後）湿度 x を 1 ステップ更新**
- **エアコン制御（必要なら再計算）**
- **（エアコン制御完了後）濃度 c を 1 ステップ更新**
- **結果出力（artifact）**

補足（連成範囲の考え方）:

- 現状は、湿度 \(x\)・濃度 \(c\) が **圧力/熱の収束判定には入っていません**（まずは「圧力＋熱」を収束させ、その確定流量で x を更新し、最後に c を更新する方針）。
- 湿度はエアコン入力（`X_in/X_ex`）に関わるため **エアコン制御の前**に更新し、濃度はエアコン制御に影響しない想定のため **エアコン制御の後**に更新します。

#### 3.2 詳細（タイムステップ内）

1. **ネットワーク更新（時変プロパティ）**
   - ノード/ブランチの時系列（`t`, `x`, `c`, `beta`, `vol`, `humidity_generation`, `dust_generation` など）を当該ステップ値へ更新
   - スケジュール配列の `index=0` は **開始時刻 (1/1 0:00)〜1:00 の状態**として扱い、`timestep=0` の計算に使う
2. **圧力計算（換気回路網, `p`）**
   - `pressureCalc=true` のとき、圧力を解き、各換気ブランチの流量（体積流量）を更新
3. **熱計算（熱回路網, `t`）**
   - `temperatureCalc=true` のとき、換気流量（移流）も取り込んで温度を解く
4. **圧力-熱 連成の収束判定**
   - `p` と `t` が両方有効なとき、圧力変化量・温度変化量が許容誤差以下になるまで反復
5. **湿度計算（`x`）**
   - 圧力＋熱が落ち着いた後に、換気流量と `humidity_generation` を使って絶対湿度を 1 ステップ更新
6. **エアコン制御**
   - 設定温度/モードに基づきON/OFF等を更新し、必要なら熱計算側を再計算
   - 湿度計算が有効な場合、エアコンモデルへの入力湿度（`X_in/X_ex`）はノードの `current_x` を優先し、取得できない場合は警告してJIS条件へフォールバック
7. **濃度計算（`c`）**
   - エアコン制御が完了した後に、換気流量・`dust_generation`・沈着率 `beta`・除去効率 `eta` を用いて 1 ステップ更新
8. **出力**
   - `schema.json`（キー配列）と、各系列の `*.f32.bin`（float32 little-endian, timestep-major）を出力

---

### 4. 出力（artifact）の概要

出力は `artifact_dir` にまとまります。

- `schema.json`: 出力系列（series）と各系列の `keys`（並び順）
- `*.f32.bin`: 各系列の時系列バイナリ（float32）

湿度/濃度の系列:

- `humidity_x`: `calc_x=true` のノードの `x`
- `concentration_c`: `calc_c=true` のノードの `c`

（系列名・ファイル名は `output.json` の `result_files` でも参照できます）

---

### 5. エアコンモデル（acmodel）の位置づけ

エアコンの COP/電力推定は `acmodel` が担当します。

- 入力: ノード温度・外気温度・（必要なら）絶対湿度・要求能力・風量
- 出力: COP と消費電力（およびログ）

詳細は `docs/acmodel_overview.md` を参照してください。


