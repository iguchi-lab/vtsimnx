# 第十三章 計算実行及び出力仕様

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 実行経路

Python run_calc の既定は POST /runs による非同期投入とポーリング。互換経路は POST /run による同期実行。HTTP 要求の config は raw JSON であり、サービスはビルダーを呼出し、その結果をソルバに渡す。

| 操作 | HTTP 経路 |
|---|---|
| 投入 | POST /runs |
| 状態取得 | GET /runs/{run_id} |
| 結果取得 | GET /runs/{run_id}/result |
| キャンセル | DELETE /runs/{run_id} |
| 同期 | POST /run |
| メタ情報 | GET /artifacts/{artifact_dir}/manifest |
| ファイル一覧 | GET /artifacts/{artifact_dir}/files |
| ファイル実体 | GET /artifacts/{artifact_dir}/download/{key} |

要求受付、計算成功、成果物取得成功を区別する。実行要求の返却だけで物理計算の完了を意味しない。API の詳細な状態遷移・認証・保持期間は既存契約を参照し、本文への全項目展開は追加整備対象とする。

## 2. サービスの処理

入力を prepare_raw_config で整え、ビルダーで展開・検証する。ビルド後にログ冗長度を設定し、ソルバを呼ぶ。builder ログを成果物へ添付し、manifest と計測時間を付ける。HTTP の事前警告とビルダー警告は集約する。

## 3. 数値系列の形式

schema の dtype は f32le、layout は timestep-major、length は非負整数 T。series の keys の個数を N とする。数値ファイルの期待サイズは式 (13-1) による。

$$ B=4TN\quad{\rm bytes}. \tag{13-1} $$

順序は時刻を外側、キーを内側とする。値(n,j)は先頭から4(nN+j) bytes の位置にある。列を独自に再ソートせず schema の keys に従う。

C++ 内部の double と保存する float32 の精度は同一ではない。保存前の収束許容値を、そのまま float32 の比較許容差にしない。

## 4. Python による復元

decode_f32_series は little-endian float32 として読み、要素数 T×N を検証して(T,N)に整形する。dtype、layout、length 不正、バイト数・要素数不一致は ArtifactDecodeError とする。

N=0 の場合は0 bytesから(T,0)の DataFrame を返す。T=0と同義ではない。時刻は start、timestep、length から復元し、end はこの復元計算に使わない。length 不一致又は必須情報不正では時刻付与を行わない経路がある。timestep=0なら同一時刻ラベルを繰り返す。

run_calc は output.index を優先し、復元可能でなければ送信 config の index を参照する。単位は df.attrs["unit"]、系列名は df.attrs["series"] に付与する。

## 5. 主要出力

| 系列 | 単位・意味 |
|---|---|
| vent_pressure | Pa |
| vent_flow_rate | m³/s |
| thermal_temperature | °C |
| thermal_heat_rate_* | W、種類別熱流 |
| humidity_x、humidity_flux | kg/kg(DA)、kg/s |
| concentration_c、concentration_flux | メタ情報と物質量の対応は第十章参照 |
| aircon_sensible_heat、aircon_latent_heat | W |
| aircon_power、aircon_cop | W、無次元 |
| hrv_sensible_heat、hrv_latent_heat | W |

系列が存在しない、空系列である、取得に失敗した、値が0である、の四者を区別する。電力モデルの失敗が数値0に残る場合はログも確認する。

## 6. 再現条件・検証

raw JSON、生成された計算用 JSON、クライアント・engine の版と commit、Δt、展開オプション、schema、警告、solverログを対応付ける。モデルの既定値が変わると同じ raw JSON でも結果が変わり得る。

照合例：T=2、N=2の値 [1,2,3,4] は [[1,2],[3,4]]、16 bytes。T=2、N=0は0 bytes、2行0列。

根拠：[サービス](../../engine/app/services/simulation.py)、[出力](../../engine/solver/output/artifact_io.cpp)、[復元](../../vtsimnx/artifacts/_decode.py)、[API契約](../../engine/docs/api_reference.md)。
SPEC-13-001：サイズ・配列順。SPEC-13-002：空系列。SPEC-13-003：時刻復元。
検証スクリプトは実際の Python decoder を呼出す。HTTP サーバーとの通信は今回未実施。
2026-09-17：実行・保存・復元を初稿化。
