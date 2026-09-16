# 第三章 raw JSON の入力仕様

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲

本章は dict として build_config 系関数へ渡す入力を規定する。HTTP 経由では先に Pydantic による検証を受けるため、直接呼出しと受入範囲が同一ではない。parse_all 単体の文字列・PathLike 対応を build_config 全体の対応とみなさない。build_config は parse_all より前に辞書としてオプションを参照する。

## 2. トップレベル

| 項目 | 型 | 現行ビルダーでの取扱い |
|---|---|---|
| simulation | object | index と tolerance を読み込む |
| nodes | array of object | 必須。各要素に key が必要 |
| ventilation_branches | array | 未指定／None は空。旧資料の「必須」と異なる |
| thermal_branches | array | 未指定／None は空 |
| surfaces | array | 未指定は面の展開なし |
| aircon | array | 未指定は空調機の展開なし |
| heat_recovery_vent | 設備入力 | 専用パーサで読み込む |
| heat_source / heat_sources | 発熱源入力 | 前者を優先して読む |
| humidity_source / humidity_sources | 発湿源入力 | 前者を優先して読む |
| builder | object | 展開オプション。第四章参照 |

nodes にユーザーが key="void" を指定した場合はエラーとする。void はビルダーが先頭に追加する。

## 3. 計算条件

simulation.index は start、end（日時文字列）、timestep（s）、length（要素数）を持つ。直接パーサの初期値は空文字・0であり、使用可能な計算条件を補完する既定値ではない。

tolerance.ventilation、thermal、convergence のパーサ既定値は各 1e-6 とする。用途別の aircon_temperature、thermal_balance、thermal_linear_residual、coupling_pressure、coupling_temperature、coupling_humidity は保持する。

**実装上の受渡し制限：** _parse_simulation が raw からコピーするのは index と tolerance だけである。raw の calc_flag はノードから再計算される。coupling、iteration、max_*、log 等を raw に追加しても、このビルダーでは引き継がれない。HTTP サービスはビルド後にログ冗長度を別途設定する。C++ が読む追加設定は第五章に分ける。

## 4. ノード

| 項目 | 意味・単位 | 正規化 |
|---|---|---|
| key | 識別名 | コメント・複合キーを展開 |
| calc_p / calc_t / calc_x / calc_c | 未知数の選択 | 展開後に simulation.calc_flag を導出 |
| p | 圧力 Pa | 換気枝に接続する場合、未指定は0。非接続では除去 |
| t | 温度 °C | 熱又は換気枝に接続する場合、未指定は20。非接続では除去 |
| x | 絶対湿度 kg/kg(DA) | スカラー・時系列 |
| c | 濃度 | スカラー・時系列。単位は第十章 |
| v | 体積 m³ | 湿気又は濃度有効時、未指定は0 |
| beta | 沈着率 1/s | 濃度有効時、未指定は0 |
| thermal_mass | 熱容量 J/K | 容量枝へ展開して元項目を除去 |
| moisture_capacity | 湿気容量 | 単位を変換し材料ノードへ展開 |
| moisture_capacity_unit | 湿気容量の入力単位 | 既定 J/(kg/kg')、内部 kg/(kg/kg) |
| w | 材料状態 | 時系列。湿気求解後は current_x と同値に追従 |

type の未指定値は normal。空調機以外の pre_temp、pre_rh はノード検証で除去される。

## 5. ブランチ

表 3-1　換気枝の型と必要項目

| type | 必要項目 | 単位 |
|---|---|---|
| simple_opening | alpha、area | —、m² |
| gap | a、n | m³/s/Pa^(1/n)、— |
| fan | p_max、q_max、p1、q1 | Pa、m³/s、Pa、m³/s |
| fixed_flow | vol | m³/s |
| pressure_loss | area と圧損を規定する値 | m²、k_total 又は摩擦・形状条件 |

eta は流入側の濃度除去効率、humidity_generation は kg/s、dust_generation は選択した物質量/s とする。発生量は枝の target に加える。流向を反転しても発生先は target のままである。

表 3-2　熱枝の型と必要項目

| type | 必要項目 | 単位 |
|---|---|---|
| conductance | conductance | W/K |
| heat_generation | heat_generation | W |
| response_conduction | area、resp_a_src、resp_b_src、resp_a_tgt、resp_b_tgt | m²、W/(m² K) |
| 共通の湿気結合 | moisture_conductance、任意の moisture_transfer_type | kg/s、分類 |

enable は有効・無効の条件である。CTF の resp_c_* は履歴熱流への無次元係数。熱・湿気コンダクタンスを同じものとして扱わない。

## 6. 高位入力及び時系列

surfaces は層・面・日射・放射からノードと熱枝を作る。aircon は吸込 in、吹出 out、制御対象 set、外気 outside 等を機器ノードへ変換する。heat_recovery_vent は OA・SA・RA・EA の接続を生成する。細目は第四・八・十一章に記す。

対象時系列は長さ N=simulation.index.length とする。長さ1は N 個へ展開し、それ以外の不一致はエラーとする。全数値項目を無条件に配列化するのではなく、フィールドごとの正規化を適用する。

空調 pre_temp の欠損は OFF 時刻なら20°Cに置換し、OFF以外はエラー。pre_rh の有限値は (0,100] % とし、欠損時刻は理想除湿を使わず従来方式を用いる。

## 7. 例外・検証・実装対応

未知キーは API の prepare_raw_config とビルダーの検証の段階で扱いが異なる。直接ビルダーの unknown_keys="error" は検証に到達した項目が対象であり、パース時に採用されなかった項目を全て検出する仕組みではない。

根拠：[parsers.py](../../engine/app/builder/parsers.py)、[validate.py](../../engine/app/builder/validate.py)、[API schema](../../engine/app/schemas/config.py)。
SPEC-03-001：枝配列省略。SPEC-03-002：simulation の保持範囲。SPEC-03-003：時系列長。
全材料・全設備パラメータの個別範囲一覧は追加整備対象。実行済みの範囲は第十四章に記す。
2026-09-17：実装の受入・除去条件に合わせて初稿化。
