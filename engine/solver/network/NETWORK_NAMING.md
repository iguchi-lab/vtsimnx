# Network Layer Naming Guide

このドキュメントは、`solver/network` 配下の4ネットワーク（換気・熱・湿気・濃度）で
命名規則と責務を揃えるための最小ルールを定義する。

## 1. 基本方針

- ノード状態は共通（shared node state）として扱う。
- ネットワーク間の内部実装を直接参照しない。
- ネットワーク間の受け渡しは明示的な I/F（`sync*`, `buildTerms`, `FlowRateMap` 等）で行う。
- 計算順序の制御は `simulation_runner` 側に集約する。

## 2. Public API の命名規約

各ネットワークの公開メソッドは、可能な限り次のライフサイクルで揃える。

1. 構築: `buildFromData(...)`
2. 時変更新: `updatePropertiesForTimestep(...)`
3. 計算: `solveXxx(...)`
4. 出力: `get*Keys()` / `collect*Values()`
5. キャッシュ無効化: `invalidateCaches()`

補足:

- solve 後の結果反映が必要な場合は `applySolveResults(...)` を使う。
- 将来共通化する場合は薄い統一ラッパ（例: `solveStep`）を追加し、ドメイン別 solve を内部委譲にする。

## 3. Cross-Network API の命名

他ネットワークから値を取り込む処理は `sync*From*` に統一する。

- 良い例:
  - `syncFlowRatesFromVentilationNetwork(...)`
  - `syncTemperaturesFromThermalNetwork(...)`（必要時）

ネットワーク内部の状態更新（外部入力を伴わないもの）は `update*` を使う。

- 例:
  - `updateFlowRatesInGraph(...)`
  - `updateNodePressures(...)`

## 4. Terms Builder の統一

湿気・濃度のように連立項を組み立てるネットワークは次を標準とする。

- 型名: `<Domain>NetworkTerms`
- 組み立て関数: `buildTerms(ConstNodeStateView nodeState, const VentilationNetwork&, <Domain>NetworkTerms&)`
- 係数・更新対象のフィールド名:
  - `genByVertex`
  - `outSum`
  - `inflow`
  - `updateVertices`

`ensureNodeIndex(...)` は private ヘルパーとして許可する。

## 5. Output API の統一

出力 API は以下の2層で整理する。

- 汎用層: `getOutputKeys(...)` / `collectOutputValues(...)`
- 専用層: `getTemperatureKeys*`, `collectHeatRateValues*` など

ルール:

- 新規ネットワークはまず汎用層を提供する。
- 専用層は性能上または可読性上の明確な理由がある場合に追加する。
- `*_outputs.cpp` の関数順は以下に統一する。
  1. `get*Keys`
  2. `collect*Values`
  3. `invalidateCaches`

## 6. Class Header の並び順

`*.h` の public セクションは次の順序を推奨する。

1. Node/Graph access（`getGraph`, `getKeyToVertex`, `nodeStateView`）
2. Build / Update（`buildFromData`, `updatePropertiesForTimestep`, `sync*`）
3. Solve / Calculate
4. Output APIs
5. Diagnostics / cache controls

private では、`ensure*` 系を先に置き、キャッシュ変数を後ろにまとめる。

## 7. 既存コードへの適用方針

- 一度に全面リネームしない（差分肥大を避ける）。
- 機能変更のついでに周辺のみ段階的に揃える。
- 互換性に影響する公開シグネチャ変更は `simulation_runner` とテストを同時更新する。

