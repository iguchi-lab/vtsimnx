# 第五章 計算用 JSON 及び計算初期化

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲

本章は C++ ソルバが読む構造を対象とする。ビルダーが生成する部分集合と、C++ が追加で受理する設定を区別する。API の raw JSON に追加設定を書くだけでは、この構造がそのまま渡されるとは限らない。

## 2. 主要項目

| パス | 内容・単位 | 初期化上の役割 |
|---|---|---|
| simulation.index | start、end、timestep [s]、length | 計算期間 |
| simulation.tolerance | ventilation、thermal、convergence、用途別値 | 停止及び物理収支の判定 |
| simulation.calc_flag | p、t、x、c | 物理分野の実行選択 |
| nodes | 展開済みノード | 固定状態・未知状態・機器状態 |
| ventilation_branches | 展開済み換気枝 | 接続と流量特性 |
| thermal_branches | 展開済み熱枝、湿気枝 | 伝達・容量・履歴・発生 |
| aircon、heat_recovery_vent | ビルダーが返す機器情報 | 実際の回路網接続は生成されたノード・枝で表す |

計算用ノードの ref_node は容量等の参照先、in_node・set_node・outside_node は機器の参照先とする。raw の thermal_mass、surfaces を C++ が同じ方法で自動展開すると仮定しない。

## 3. C++ の計算条件

start・end は文字列、timestep・length は数値として読み込む。tolerance の ventilation・thermal・convergence は数値が必要であり、欠落又は型不正は runtime_error とする。

thermal は、aircon_temperature [K]、thermal_balance [W]、thermal_linear_residual [相対値] の初期値へコピーする。個別キーがある場合は個別値で上書きする。同じ数値から始まっても、三者の次元は異なる。

## 4. ビルダー外の追加設定

| 計算用 JSON の項目 | C++ 側の取扱い | raw ビルダー |
|---|---|---|
| simulation.iteration.max_inner | 正の整数、既定100 | 引継ぎなし |
| simulation.iteration.max_coupling | 正の整数、未指定は max_inner | 引継ぎなし |
| simulation.iteration.max_aircon_control | 正の整数、未指定は max_inner | 引継ぎなし |
| simulation.max_inner_iteration | iteration 自体がない場合の旧形式 | 引継ぎなし |
| simulation.max_coupling_iteration | 読込み順で新形式の値より後に上書き | 引継ぎなし |
| simulation.max_aircon_control_iteration | 同上 | 引継ぎなし |
| simulation.coupling | 湿気・潜熱・エンタルピー等の連成設定 | 引継ぎなし |
| simulation.log | ログの設定 | raw 値の引継ぎなし。サービスで別設定 |

C++ の機能を直接調査する場合は、ビルド済み JSON にこれらを加えたことを記録する。通常の run_calc がビルダーを迂回するという意味ではない。

## 5. 未知量と境界

calc_p、calc_t、calc_x、calc_c を物理量ごとに解釈する。同一ノードで、温度は未知、湿度は固定という指定ができる。空調ノードの湿度は制御処理が設定するため未知湿度に含めない。

熱容量ノードは ref_node を用いた旧状態を熱枝へ与える。湿気容量ノードは正の moisture_capacity を持つ未知湿度として方程式へ入る。response_conduction は温度・熱流の履歴を保持し、各反復を時間の進行として扱わない。

## 6. 出力・例外・確認

初期化の結果はノード状態、接続グラフ、計算定数であり、まだ結果系列ではない。欠落ノード・型・参照不正の拒否条件はパーサごとに異なる。raw 検証だけで計算成立を保証しない。特異な圧力・熱・湿気系は求解段階でも失敗し得る。

根拠：[simulation パーサ](../../engine/solver/parser/sim_constants_parser.cpp)、[ノードパーサ](../../engine/solver/parser/nodes_parser.cpp)、[枝パーサ](../../engine/solver/parser/branches_parser.cpp)、[builder の型](../../engine/app/builder/config_types.py)。
SPEC-05-001：用途別許容値。SPEC-05-002：反復上限の優先順。読解確認、C++ の起動確認は今回未実施。
2026-09-17：入力経路の差と初期化を初稿化。
