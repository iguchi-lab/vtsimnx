# 付録 — 定数・入力変換・実装差異

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## A. 共通記号

第一章の記号表及び各章の式直後の定義による。入力パラメータ名と数式記号は一対一とは限らない。特に熱流Q、風量q、機器仕様Qを区別する。

## B. 確認した定数

| 用途 | 値 | 実装 |
|---|---|---|
| 乾き空気密度 | 1.2 kg/m³ | engine/archenv/include/archenv.h |
| 乾き空気定圧比熱 | 1006 J/(kg K) | 同上 |
| 水蒸気比熱 | 1846 J/(kg K) | 同上 |
| 蒸発潜熱定数 | 2500800 J/kg | 同上 |
| 重力加速度 | 9.80665 m/s² | 同上 |
| 開口・隙間等の微小差圧 | 1e-10 Pa | TOLERANCE_SMALL |
| ファン平滑化パラメータ | 1e-6 | TOLERANCE_MEDIUM |
| 微小流量 | 1e-10 m³/s | FLOW_RATE_MIN |
| 湿気容量の単位換算用潜熱 | 2500000 J/kg | builder/moisture_capacity.py |
| HRV回収潜熱用 | 2501000 J/kg | solver/hrv/hrv_controller.cpp |

潜熱の数値は用途ごとに異なる。仕様書上で一つの値に統合しない。温度依存の vapor_latent_heat を呼ぶ処理も別に存在する。

## C. 主要な変換対応

| raw | ビルダー出力 | ソルバでの役割 |
|---|---|---|
| nodes[].thermal_mass、v | key_air・key_c、capacity枝 | 旧温度と熱容量の離散化 |
| nodes[].moisture_capacity | key_mx、moisture_conductance枝 | 材料湿度の未知数と線形結合 |
| surfaces | 層・表面ノード、伝導・放射・日射枝等 | 熱回路網 |
| aircon | 機器ノード、吸込・吹出換気枝 | 設定温度・能力・湿度境界 |
| heat_recovery_vent | 給気境界・排気ジャンクション・換気枝 | 温湿度回収 |
| heat_source | heat_generation枝 | 熱源 |
| humidity_source | humidity_generation枝、発湿先calc_x | 水蒸気源 |
| simulation.calc_flag | 展開後のcalc_*から再生成 | 分野別実行選択 |
| simulation.coupling、iteration | 現行ビルダーは保持しない | C++直入力とは経路が異なる |

ノード key の && は複合ノードを展開、枝 key の A->B->C は A->B と B->C を展開、|| 以降はコメントに分離する。枝の空端点は void に置換する。根拠：[utils.py](../../engine/app/builder/utils.py)。

## D. 照合例

[実行用スクリプト](examples/verify_examples.py)、[raw入力](examples/raw_example.json)、[ビルダー出力](examples/solver_example.json)、[確認記録](examples/verification_results.json)。

## E. 実装に合わせた記述と例外

| 項目 | 既存説明からの修正・補足 | 新本文 |
|---|---|---|
| 湿気解法 | Gauss–Seidelの説明をSparseLU直接法へ | 第九章 |
| raw枝配列 | 直接ビルダーは未指定を空として受理 | 第三章 |
| 追加simulation設定 | coupling・iterationはC++対応があってもrawビルダーで保持しない | 第三～五章 |
| 熱交換換気 | 排気0時は給気流量を使用する分岐を追加 | 第十一章 |
| 電力推定失敗 | 例外を機器別に捕捉しログに記録、系列は0のまま | 第十二章 |
| 運転点風量 | 連成側のV_outer=25.5/60、V_vent=0を明記 | 第十二章 |
| 濃度単位 | メタ情報のkg/sと個数基準等の計算の差を記録 | 第一・十章 |
| 潜熱定数 | 2.5e6、2.5008e6、2.501e6を用途別に記載 | 本付録 |
| 日時精度 | asi8を常にnsと解釈。us精度の1時間間隔はtimestep=4になる | 第二章 |
| 時刻 | タイムゾーン付き入力はUTC相当のnaive時刻へ | 第二章 |

これらは本仕様書を実装に合わせた記録である。既存ソースを仕様書に合わせて変更していない。

## F. 版及び残作業

2026-09-17 構成案0.1：目次と執筆要領。\
2026-09-17 本文初稿0.2：14章、付録、変換例と確認手順。実装を基準にする方針へ更新。

全定数・全フィールド・全エラーの機械可読一覧及び機種別係数表は、本文初稿では未完了である。残作業は第十四章と整備計画で管理する。
