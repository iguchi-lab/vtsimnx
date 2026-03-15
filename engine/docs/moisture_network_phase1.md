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

