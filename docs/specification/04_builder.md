# 第四章 ビルダーによる計算用 JSON の生成

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲・入力・出力

build_config は検証後の dict、build_config_with_warnings は (dict, 警告文字列配列)、build_config_with_warning_details は (dict, 警告文字列配列, 構造化警告配列) を返す。入力を deepcopy してから変換する。output_path=None は JSON ファイルを保存しないが、ログ出力まで無効にする指定ではない。

## 2. オプション

優先順は、関数引数の非 None 値、raw.builder の値、raw トップレベルの同名値、既定値の順とする。JSON の bool 設定には bool 型の値を用いる。

| オプション | 既定 |
|---|---|
| add_surface、add_aircon、add_heat_recovery_vent | True |
| add_capacity、add_moisture_capacity | True |
| add_surface_solar、add_surface_nocturnal、add_surface_radiation | True |
| add_surface_radiation_exclude_glass | False |
| surface_layer_method | rc |
| response_method | arx_rc |
| response_terms | None（個別処理に委ねる） |

response_terms を指定する場合は正の整数とする。bool、非整数 float、0、負値を拒否する。

## 3. 変換順序

1. parse_all：simulation、nodes、換気枝、熱枝、surfaces、aircon を読み込む。
2. 面の展開：有効なら面から得たノードと熱枝を追加する。
3. 発熱源の展開：heat_generation 枝を追加する。
4. 発湿源の展開：humidity_generation 枝を追加し、発湿先の calc_x を True にする。
5. 湿気容量の予備処理：有効なら元ノードの calc_x を True、無効なら容量関連項目を除去する。
6. 空調機の展開：機器ノードと換気枝を追加する。空調の calc_x は False、calc_c は set（省略時 in）の指定から引き継ぐ。
7. 熱交換換気の展開：専用パーサの結果からノードと換気枝を追加する。
8. 熱容量の展開。
9. 湿気容量の展開。
10. p、t、x、c ごとに、展開後ノードの calc_* が一つでも真なら simulation.calc_flag を真にする。
11. 検証、正規化、未知項目処理及び重複キー処理を行う。
12. 必要な場合だけファイルへ保存する。

この順序により、面が生成した熱容量も第8工程で展開対象になる。raw の値と生成値が混在した中間状態を最終 JSON とみなさない。

## 4. 熱容量変換

体積 V>0 のノードで、空気熱容量 C_a と残余容量 C_f は式 (4-1) とする。

$$ C_a=1.2\times1006\,V,\qquad C_f=\max(0,C_T-C_a). \tag{4-1} $$

C_T は thermal_mass [J/K]、V は m³。C_a>C_T+10^-9 の場合は ValueError。正の容量ごとに次の要素を追加する。

| 容量 | 生成ノード | 生成枝 | conductance |
|---|---|---|---|
| 空気 | key_air、type=capacity、ref_node=key、calc_t=False | key_air→key、subtype=air_capacity | C_a/Δt [W/K] |
| 残余 | key_c、type=capacity、ref_node=key、calc_t=False | key_c→key、subtype=capacity | C_f/Δt [W/K] |

初期温度を生成ノードに引き継ぎ、元ノードの thermal_mass を削除する。V=0 の場合は C_a=0 とし、正の C_T を残余容量として展開する。

## 5. 湿気容量変換

入力容量 C_in の単位が J/(kg/kg') の場合、式 (4-2) により内部容量を求める。kg/(kg/kg) の場合は C_x=C_in とする。

$$ C_x=C_{\rm in}/(2.5\times10^6). \tag{4-2} $$

入力容量及び Δt は有限で正とする。key_mx（calc_x=True、calc_t=False、type=capacity、subtype=moisture、ref_node=key）を追加し、moisture_capacity=C_x とする。初期 x は元ノードの x、省略時0を用いる。

key_mx→key の熱枝を type=conductance、conductance=0、moisture_conductance=C_x/Δt として追加する。元ノードの容量・単位項目を削除する。これは熱容量の「旧温度境界」と同じ処理ではなく、材料側湿度も未知数とする二ノードの湿気結合である。

add_moisture_capacity=False は容量を削除して無効化する。ソルバへ容量を直渡しする指定ではない。

## 6. 保存・例外・制約

output_path の末尾が .gz なら UTF-8 の gzip JSON、それ以外は字下げ4の UTF-8 JSON とする。ファイル出力は検証後に行う。

風量の換算は行わない。raw_config の coupling 等を引き継がないこと、空調湿度を固定境界にすることも現行の変換仕様である。

## 7. 照合例と根拠

C_T=120720 J/K、V=50 m³、Δt=3600 s の場合、空気容量60360、残余容量60360 J/K、両枝の conductance は16.7666667 W/K。湿気容量入力2.5×10^6 J/(kg/kg') は内部容量1、Δt=3600なら湿気コンダクタンス1/3600 kg/sになる。

[変換例ファイル](examples/verify_examples.py)で実際のビルダー出力と照合する。
根拠：[builder.py](../../engine/app/builder/builder.py)、[build_options.py](../../engine/app/builder/build_options.py)、[thermal.py](../../engine/app/builder/thermal.py)、[moisture_capacity.py](../../engine/app/builder/moisture_capacity.py)。
SPEC-04-001：変換順。SPEC-04-002：熱容量分離。SPEC-04-003：湿気容量変換。
2026-09-17：実処理順と容量展開を初稿化。
