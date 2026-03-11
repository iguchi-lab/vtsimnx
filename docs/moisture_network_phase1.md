# 湿気回路網（Phase1: 線形RC）

Phase1 では、既存の移流ベース湿度計算に加えて、線形RC型の湿気回路網を導入します。

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

## builder 拡張（任意）

- `nodes[].moisture_capacity` を持つノードに対して、以下を自動生成:
  - 容量ノード: `<key>_mx`
  - 湿気伝達枝: `<key>_mx-><key>` (`moisture_conductance = moisture_capacity / timestep`)
- builder オプション:
  - `builder.add_moisture_capacity`（既定: true）

## 互換性

- 既存 JSON（新フィールドなし）: 従来の湿度計算結果を維持
- 新フィールドあり: 移流 + 生成項 + 湿気回路網項を同時に陰的更新

