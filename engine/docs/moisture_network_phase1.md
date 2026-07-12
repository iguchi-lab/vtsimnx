# 湿気回路網（Phase1: 線形RC）

Phase1 では、既存の移流ベース湿度計算に加えて、線形RC型の湿気回路網を導入します。

> 方針: 現段階の完成範囲は「Phase1（線形RC）」までとし、非線形HAMは将来課題として扱います。

## 目的

- 壁体/材料側の湿気容量を持つノードを追加できるようにする
- ノード間の湿気伝達を `moisture_conductance` で表現する
- 既存入力との後方互換を維持する（新フィールド未使用時は既存挙動）

## 入力フィールド（追加）

### ノード (`nodes[]`)

- `moisture_capacity` (number, optional)
  - ノードの湿気容量
  - `>0` のとき、湿度更新で容量項として使われる
  - 既定単位: `[J/(kg/kg')]`
- `moisture_capacity_unit` (string, optional)
  - `moisture_capacity` の入力単位を指定
  - 対応:
    - `"J/(kg/kg')"`（既定、builder で内部単位へ変換）
    - `"kg/(kg/kg)"`（内部単位としてそのまま使用）
  - `"J/(kg/kg')"` 指定時は `moisture_capacity / Lv` で換算（`Lv=2.5e6 [J/kg]`）
- `w` (number または array, optional)
  - 材料側含湿状態の時系列入力（内部状態 `current_w`）
  - 湿度更新は `x` を主状態として計算し、`current_w` は同値で追従

### 熱ブランチ (`thermal_branches[]`)

- `moisture_conductance` (number, optional)
  - source/target 間の湿気伝達コンダクタンス `[kg/s]`
  - 湿度方程式で双方向結合として扱う
  - 既存の温度計算への影響を避けるため、Phase1 では `conductance` と独立に扱う

#### 記述規約（混乱防止）

- 入力JSON上の表記は、熱ブランチと同様に `source -> target` で統一してください。
- ただし `moisture_conductance` は実装上、方程式組み立て時に双方向リンクとして扱われます。
- つまり「表記方向」は主に可読性・命名規約のためで、物理モデルとしては双方向伝達です。

## builder 拡張（任意）

- `nodes[].moisture_capacity` を持つノードに対して、以下を自動生成:
  - 容量ノード: `<key>_mx`
  - 湿気伝達枝: `<key>_mx-><key>` (`moisture_conductance = moisture_capacity / timestep`)
- builder オプション:
  - `builder.add_moisture_capacity`（既定: true）

## 互換性

- 既存 JSON（新フィールドなし）: 従来の湿度計算結果を維持
- 新フィールドあり: 移流 + 生成項 + 湿気回路網項を同時に陰的更新

## スコープ（完成範囲と将来課題）

### 現在の完成範囲（Phase1）

- 線形RCとしての湿気計算
  - 換気移流（flow_rate）
  - 発湿（humidity_generation）
  - 湿気伝達（moisture_conductance）
  - 湿気容量（moisture_capacity）
- 圧力・熱との連成ループへの統合
- 既存入力との後方互換（新フィールド未使用時）

### 将来課題（Phase2以降）

- 非線形HAM
  - 吸着等温線（非線形・ヒステリシス）
  - 温湿度依存の材料物性
  - 液水移動、結露・再蒸発の詳細扱い
- 多層壁の高忠実度湿気移動モデル

## Phase1.5: 水分収支診断と材料相変化潜熱

連成ループ構造はそのままに、湿気項の意味別整理と潜熱連成を追加します。

### 水分収支内訳（診断）

求解後にノードごとの `[kg/s]` 内訳を評価します（ソルバ数値には影響しません）。

- `ventilationTransport`: 換気による正味水蒸気流入
- `vaporGeneration`: `humidity_generation`
- `materialPhaseChange`: `moisture_conductance` による正味水蒸気流入
- `airconCondensation`: 現状 0（将来の空調除湿）
- `storage`: \(C(x^{n+1}-x^n)/\Delta t\)
- `residual`: `storage - (vent+gen+material+aircon)`（方程式適合の検算）

符号規約（相変化）:

- 正の \(\dot m_\mathrm{phase}\): 材料 → 空気（蒸発）
- 負: 空気 → 材料（凝縮）
- 材料ノードでは \(\dot m_\mathrm{phase} = -\texttt{materialPhaseChange}\)

換気枝の任意フィールド `humidity_source_type`（既定 `vapor_injection`）:

- `vapor_injection` / `room_evaporation` / `external_source`
- Phase1.5 では診断分類のみ。`room_evaporation` の室内潜熱連成は将来課題

### 潜熱連成モード

`simulation.coupling.latent_coupling_mode`:

| 値 | 意味 |
|----|------|
| `disabled` (0, 既定) | 潜熱フィードバックなし |
| `from_humidity_change` (1) | **非推奨・将来削除予定**。\(Q=-\rho V L \Delta x/\Delta t\)（換気を相変化と誤認し得る）。パーサは WARN を出す |
| `from_phase_change` (2) | 推奨。材料ノードのみ \(Q=-L_v \dot m_\mathrm{phase}\) |

`from_phase_change` では空気ノード・換気・発湿・空調は熱源に載せません。熱の載せ先は `moisture_capacity > 0` の材料ノードです。

有効条件: `calc_flag.x` かつ `calc_flag.t` かつ `moisture_enabled=true`。

### 換気エンタルピー（湿り空気エネルギー収支）

`simulation.coupling.moist_enthalpy_enabled`（既定 `false`）:

- ON 時、換気移流と空気ノード capacity 蓄積を \(h(T,x)=(c_{pa}+x c_{pv})T+x L_v\) の収支で解く（未知数は温度 \(T\)、連成中の \(x\) は既知）
- 空調処理熱（能力・COP・DUCT 風量連動の全熱）も \(\dot m |h_\mathrm{in}-h_\mathrm{out}|\) に統一（顕熱/潜熱は acmodel 互換のため分解）
- 吹出絶対湿度 `supplyX` を空調ノード `current_x` に反映し、除湿量 \(\dot m(x_\mathrm{in}-x_\mathrm{supply})\) を `airconCondensation` 診断へ記録（湿気移流境界と能力計算を一致）
- 必須: `calc_flag.x` かつ `calc_flag.t` かつ `moisture_enabled=true`（非連成では当該ステップの更新後 \(x\) を熱へ戻せない）
- `from_humidity_change` との併用は禁止（二重計上）
- `from_phase_change` との併用も当面禁止（材料側のみ \(Q=-L\dot m\) だと空気側の対向項がなくエネルギーが片側欠損する）

### 将来（Phase1.5 以降）

- 相変化潜熱の材料↔空気 対向項（同一基準エンタルピー）
- `room_evaporation` 発湿の室内潜熱
- `from_humidity_change` の削除
- 空気 capacity の \(\rho V/\Delta t\) を体積・dt から直接計算（付加熱容量との分離）
- `moisture_transfer_type` による相変化枝の明示
## 圧力・熱・湿気の連成

Phase1 実装では、1タイムステップの内側反復で次のように連成します。

- `air (pressure) -> thermal (temperature) -> moisture (humidity x)`
- 収束判定には **圧力 + 温度 + 湿気 (x)** を同時に用いる
- 潜熱（除湿に伴う熱のやり取り）は **熱ネットワークの heat_source には現在フィードバックしていません**（仕様B）
- 反復は有効状態量が2つ以上ある場合に有効化（例: `p+t`, `t+x`, `p+x`）

### 既定ON / 切替

- 既定: `simulation.coupling.moisture_enabled = true`
- 連成OFF（従来互換に近い挙動）:
  - `simulation.coupling.moisture_enabled = false`
  - この場合、湿気更新は外側ループで1回のみ実行

### 調整パラメータ（任意）

- `simulation.tolerance.coupling_humidity`:
  - 湿気収束許容誤差（未指定時は `simulation.tolerance.convergence`）
- `simulation.coupling.humidity_relaxation`:
  - 湿気反復の緩和係数 `(0,1]`（既定 `1.0`）
- `simulation.coupling.latent_relaxation`:
  - 潜熱フィードバック緩和係数 `(0,1]`（既定 `0.5`）
- `simulation.coupling.humidity_solver_tolerance`:
  - 湿気内部ソルバ（直接法）の相対残差許容誤差（既定 `1e-9`）
  - 判定は `||Ax-b|| / ||b||`（`||b||=0` の場合は `||Ax-b||`）

### 設定互換（移行中）

- `simulation.coupling.humidity_solver_max_iter` は後方互換のため受理しますが、直接法では使用しません（ログに WARN を出力）。

## 実装レイヤ（C-3 進行中）

- 正規実装は `core/humidity` に配置
  - `core/humidity/humidity_solver.*`
  - `core/humidity/humidity_coupling.*`
- `transport` 層の湿気ソルバ入口は廃止し、呼び出しは `core/humidity/humidity_solver.*` に統一
- `core/humidity/humidity_solver` は湿気反復の収束情報（反復回数・残差）を返し、連成ログ診断に利用

