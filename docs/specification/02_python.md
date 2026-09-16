# 第二章 Python モジュール

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲及び構成

Python モジュールは入力データの作成、環境条件の補助計算、実行要求、結果復元を行う。C++ の物理ソルバをクライアント内で実行するものではない。

| モジュール | 主な処理 | 入出力の受渡し |
|---|---|---|
| archenv | 湿り空気、太陽位置、日射・日影、風、夜間放射、地盤温度、快適性 | 数値・配列・DataFrame → 境界条件又は評価値 |
| schedule | 空調、風量、顕熱、発湿等のスケジュール | 生活条件等 → 時系列 |
| materials | 材料テーブル | 材料情報 → 物性値 |
| utils | JSON 化、ファイル入出力等 | Python オブジェクト ↔ 保存用データ |
| run_calc | HTTP 実行及び結果コンテナ | raw 入力 → CalcRunResult 又は dict |
| artifacts | manifest・schema・数値系列取得と復元 | bytes → DataFrame |
| units | フィールド・系列の単位表 | 名前 → 単位文字列 |

公開面の安定性は [api_stability.py](../../vtsimnx/api_stability.py) と [public_api.md](../public_api.md)による。各補助関数の詳細係数までの転記は本初稿の残作業として管理する。

## 2. run_calc の入力と戻り値

| 引数 | 型・既定値 | 処理 |
|---|---|---|
| base_url | str、必須 | engine の基準 URL |
| config_json | dict / str / Path、必須 | dict 又は JSON・JSON.gz ファイル |
| as_result | None（解決後 True） | True は CalcRunResult、False はレスポンス dict |
| with_dataframes | None | as_result の旧別名。矛盾する同時指定は ValueError |
| raise_on_error | False | 結果コンテナの系列・ログ取得失敗を例外にするか |
| compress_request | True | 要求の圧縮 |
| timeout | 600.0 s | 通信・非同期完了待ちの上限 |
| use_legacy_run | False | False：/runs、True：/run |
| poll_interval | 1.0 s | 非同期実行の照会間隔 |
| request_output_path | None | 指定時、送信前の正規化済み config を保存 |
| output_path | None | 指定時、API レスポンスを保存 |

入力辞書を deepcopy し、時刻正規化、JSON 互換化の順に処理する。送信 payload は {"config": 正規化済み辞書} とする。request_output_path に保存するのは外側の config 包装を含まない辞書である。

timeout は物理計算の Δt 又は反復上限を変更しない。結果の dataframes プロパティは取得済みの系列だけを返す。未取得系列は get_series_df 等で取得する。

## 3. JSON 互換化

| Python の値 | 変換先 |
|---|---|
| None、bool、int、str、有限 float | その値 |
| float の NaN・±Inf | None（JSON の null） |
| datetime、date | ISO 文字列 |
| Path | パス文字列 |
| Series、Index、ndarray、tuple | 配列（要素を再帰変換） |
| DataFrame | 列名 → 値配列の辞書（行 index を含めない） |
| numpy scalar | Python scalar にして再帰変換 |
| dict | キーを str にし、値を再帰変換 |
| 対応しない型 | TypeError |

null への変換は、そのフィールドで欠損値が許可されることを保証しない。後段の検証条件を適用する。NaT の型別挙動は datetime 判定との順序も関係するため、全ての日時欠損が必ず null になるとは規定しない。

## 4. 時刻正規化

simulation.index が dict の場合はこの関数では変更しない。DatetimeIndex 又は datetime 配列は start、end、timestep、length の辞書にする。間隔が一定でない場合は ValueError とする。1点の場合は timestep=0 とするため、正の時間刻みが必要な計算には明示した辞書を用いる。

タイムゾーン付きの時刻は tz_convert(None) により UTC 相当の naive 時刻へ変換する。JST の時刻文字列をそのまま保持する処理ではない。実装は DatetimeIndex.asi8 の差を常に 1,000,000,000 で除し、丸めて整数にする。asi8 がナノ秒精度の場合は秒間隔となるが、精度を明示的に ns へ変換する処理はない。マイクロ秒精度の1時間間隔では timestep=4 となることを実行確認した。秒間隔を保持する入力は ns 精度にそろえるか、timestep を明記した辞書を用いる。これは現行実装の制約であり、実装を修正した想定の仕様にはしない。

## 5. 補助計算と入力の接続

湿度比関数の戻り値を nodes[].x に、風圧を固定境界の p に、面別の日射・夜間放射を surfaces の対応項目に、発熱・発湿の系列を対応する発生源に渡す。API の出力単位を変換してから渡し、変数名だけで単位を判断しない。スケジュールの 8760 要素と simulation.index.length は一致させる。うるう年等で長さを変える場合は各関数の対応を別途確認する。

## 6. 出力、例外及び照合例

CalcRunResult は系列を遅延取得する。raise_on_error=False では取得失敗を errors に記録し None を返す経路がある。計算送信自体の例外が全て None に変わるわけではない。

照合例：Series([1,2]) → [1,2]、DataFrame({"a":[1,2]}) → {"a":[1,2]}、風量 360 m³/h → 入力 vol=0.1 m³/s。

根拠：[run_calc.py](../../vtsimnx/run_calc/run_calc.py)、[jsonable.py](../../vtsimnx/utils/jsonable.py)、[_index.py](../../vtsimnx/run_calc/_index.py)。
SPEC-02-001：入力のコピーと正規化順。SPEC-02-002：時刻変換。検証記録は第十四章。
2026-09-17：クライアント契約とモジュールの役割を初稿化。
