# 今後の課題メモ（性能最適化と Colab 二層運用）

最終更新: 2026-04-19

このメモは、直近の検証と議論内容を「次に何をやるか」に落とすための作業メモ。

## 1. 直近の実測サマリ（熱ソルバ）

- 対象ケースでは、熱ソルバの行列次元は `n=929`（`DirectT cache stats` で確認）。
- 主要な時間消費は依然として `solveFull`（フル LU）側。
- `rhs-cached` と `post=cached` の導入で、1回あたり平均時間は改善済み。
  - ただし直近は改善幅が小さく、追加の大幅改善には `solveFull` 削減が必要。

## 2. 現在のボトルネック認識

- `solveFull` が多い理由は `coeffSigMismatch` が継続して発生しているため。
- 内訳カウンタでは以下が大きい:
  - `coeffSigAirconOnChanged`
  - `coeffSigSetNodeChanged`
  - `coeffSigFlowChanged`
- つまり、エアコン ON/OFF と flow 変化で係数行列 `A` が頻繁に変化し、LU 再因子化が必要になる。
- `cholFactorize=0` で、Cholesky 系はこのケースでは使えていない。

## 3. すでに実施した改善

- `rhs` 同一時の解再利用（`rhs-cached`）。
- `rhs-cached` 時の後処理再利用（`post=cached`）。
- 一時バッファ再利用（`Eigen::VectorXd`、温度バッファ）。
- `coeffSig` 変化要因の分解ログ追加（flow/aircon/set-node）。
- `DirectT cache stats` に `n` を追加。
- artifact 命名を時刻先頭へ変更。
  - 新形式: `artifacts.<epoch_ms>.output.<run_id>`
  - 目的: ファイル名ソートで時系列管理しやすくする。

## 4. 残課題（性能）

### 優先度 High

- `solveFull` を減らすための制御安定化:
  - エアコン ON/OFF のヒステリシス導入。
  - 最小 ON/OFF 継続時間の導入。
  - 制御更新の間引き（毎反復更新を緩和）。

### 優先度 Mid

- flow 変化に対するデッドバンド導入（近似、要 A/B 検証）。
- 圧力側更新頻度の見直し（`A` 更新回数削減狙い）。

### 優先度 Low

- LU バックエンド比較（KLU/UMFPACK/PARDISO 等）。
- 反復法（SOR など）の検証は可能だが、現状は `solveFull` 削減が先。

## 5. Colab 二層運用（方針）

### 方針

- remote（現行 API）を本命として維持。
- light（Colab 同梱）を追加して用途分離。
- 失敗時は remote へフォールバック可能にする。

### light 版の現実的スコープ

- `pressureCalc=true` でも Ceres 非依存の軽量解法を用意。
- 未収束時は明示フラグを返すか、remote へ自動フォールバック。

## 6. Colab 移植時の懸念点

- Ceres 周辺依存（SuiteSparse、glog、gflags 等）の導入難度。
- セッション再作成時のビルド再現性と時間。
- VM 個体差による実行時間のばらつき。
- artifact I/O 負荷。

## 7. 運用上の整理

- 入力条件としての ON/OFF は変えられないため、実装側でチャタリング耐性を上げる。
- 2回目反復は軽量化済みで、そこを削る効果は限定的。
- 大きな短縮には `solveFull` 発生回数そのものを下げる必要がある。

## 8. 次アクション（提案）

1. エアコン制御にヒステリシス + 最小 ON/OFF 時間を導入。
2. `solveFull` / `coeffSig*Changed` の減少を同一ケースで再測定。
3. 近似導入時は結果差分（温度/湿度/flow）を検証して許容範囲を定義。

