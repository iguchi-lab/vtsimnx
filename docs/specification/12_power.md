# 第十二章 電力及び消費電力量の算定

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲及び単位

空調機の運転点から電力・COP を評価する。入力 InputData.Q、Q_S、Q_L はW、T_in・T_exは°C、X_in・X_exはkg/kg(DA)、V_inner・V_outer・V_ventはm³/s。機器仕様の Q・P・P_fan はkWを用い、各モデル内で必要に応じてWへ変換する。COPResult.power はkW、aircon_power はW。

換気ファンの圧力・流量計算だけから電力を自動算定するものではない。HRV の回収熱は消費電力ではない。

## 2. ソルバからの呼出し

停止機の電力・COP 系列値は0のままとする。運転機では顕熱・潜熱を再評価し、totalHeatCapacity が正でなければモデルを呼ばず (0 W, COP=0) を返す。

運転点の V_inner は回路網風量を用いる。現行 buildAcmodelInput の V_outer は25.5/60 m³/s、V_vent は0を設定する。モデル API が任意の外気風量等を受け付けることと、連成ソルバの受渡しが同じではない。

X_in 又は X_ex が正でない場合は JIS 条件で補完する。モデル結果 valid=False は例外を発生させるが、系列を組み立てる calculatePowerOrCOPValues は機器ごとに捕捉して ERROR ログを出し、その要素を初期値0のまま残す。したがって「出力0」は停止・零負荷・推定失敗を値だけでは区別できない。

## 3. RAC

冷房・暖房の電力は、負荷・外気条件から求める四次多項式 f の比に定格電力を乗じる。

```math
f(z,\theta)=\sum_{r=0}^{4}a_r(\theta)z^r,\qquad
a_r(\theta)=p_{r2}\theta^2+p_{r1}\theta+p_{r0}. \tag{12-1}
```

暖房の z は[0,1]に制限、冷房は下限0だけを制限する。p は機器容量と dualcompressor に応じた係数表・補間で決める。

```math
P_{\rm kW}=10^{-3}P_{\rm rtd,W}
\frac{f(z_1,T_{\rm ex})}{f(z_2,T_{\rm ref})}. \tag{12-2}
```

T_ref は冷房35°C、暖房7°C。z₂=q_rtd/q_max。z₁は補正処理負荷 [MJ/h] を q_max,W×0.0036 で除した値。冷房の補正処理負荷は顕熱・潜熱の処理負荷を C_HM_C C_AF_C で除す。暖房はデフロスト等を含む補正比で除す。

出力 COP の分子は、冷房では入力負荷 L_CS+L_CL、暖房では入力 L_H のkW値である。モデル内部で制限した処理負荷と必ず一致すると仮定しない。係数表の全数値及び負荷補正の細目は独立節として追加詳細化する。

## 4. CRIEPI

機器仕様から同定した R(Q)=aQ²+bQ+c と補機相当の P_c を使う。ここで Q、P_c はkW。冷房で蒸発温度 T_e を飽和回避処理後の値とし、M_c を凝縮側空気質量流量、c_p,ex を比熱とする。

```math
A=R(Q)(T_e+273.15),\quad C=1000Q/(M_cc_{p,ex}),\quad
D=T_{\rm ex,adj}-T_e+C,
```

```math
COP=\frac{Q(A-C)}{QD+AP_c}. \tag{12-3}
```

分母・分子が有限、|分母|>1e-12、結果が有限かつ正なら閉形式解を使う。不成立時は初期COP=5、最大100回、変化量1e-3未満を終了条件に固定点反復する。上限時は警告を記録し、最終COPが正なら valid=True とする現行経路がある。警告なしの収束を保証する値ではない。

暖房は閉形式候補を評価して飽和回避が不要なら採用し、必要なら反復へ戻す。係数同定、暖房の閉形式及び飽和回避の全条件は追加詳細化対象。電力は input.Q/1000/COP [kW] とする。

## 5. DUCT_CENTRAL

定格点の熱源効率と理論効率の比 e_r,rtd を[0,1]に制限し、e_r,min=0.65e_r,rtd、e_r,mid=0.95e_r,rtd とする。負荷が最低能力以下では原点から比例、中間点・定格点までは区分線形補間する。定格超過で e_r,rtd>0.4 の場合は低下させ、下限0.4を適用する。それ以外の定格超過は定格比を用いる。

```math
P_{\rm total,kW}=\frac{Q_{\rm W}}{e_{\rm th}e_r}10^{-3}+P_{\rm fan,kW}. \tag{12-4}
```

送風機は、内部風量 v をm³/hに変換して式 (12-5) により求める。

```math
P_{\rm fan,kW}=\max\left(0,
(P_{\rm fan,rtd,W}-F_{\rm SFP}v_{\rm vent})
\frac{v_{\rm supply}-v_{\rm vent}}{v_{\rm design}-v_{\rm vent}}10^{-3}\right). \tag{12-5}
```

給気量・設計風量・定格ファン電力のいずれかが非正、又は分母の絶対値≤1e-9なら0。F_SFP はモデルの定数。理論効率は熱交換器・冷媒計算の結果を使う。ファンは既に総電力に含まれるので別加算しない。

## 6. LATENT_EVALUATE

熱交換器面積を定格能力5600 Wを境に選び、表面温度から冷媒の理論効率を評価する。冷房の q=Q_W/1000 に対する効率比は −0.0316q²+0.2944q。ファン電力は正の能力に対して式 (12-6) とする。

```math
P_{\rm fan,kW}=(1.4675q^3-8.5886q^2+20.217q+50)10^{-3}. \tag{12-6}
```

能力≤0ならファン電力0。圧縮機電力とファン電力を加え、入力処理熱量のkW値との比をCOPとする。熱交換器・冷媒補正式と暖房の詳細は追加詳細化対象。

## 7. 消費電力量の集計

ソルバの確認済み系列は電力[W]であり、本節の電力量は利用者側の後処理として定義する。各行がΔt秒区間を代表する条件で、式 (12-7) を使う。

```math
E_{\rm kWh}=\sum_n P_n\Delta t_n/(3.6\times10^6). \tag{12-7}
```

1000 Wを1800秒ずつ2区間なら1 kWh。時刻ラベルの差だけで最後の一行を落とさない。推定失敗の0、欠測、失敗ランを有効な零消費として集計しない。平均COPは単純平均と熱量加重で意味が異なるため、期間熱量/期間電力量で求める場合はその定義を明記する。

## 8. 検証・根拠

[aircon_controller.cpp](../../engine/solver/aircon/aircon_controller.cpp)、[運転点作成](../../engine/solver/aircon/aircon_latent.cpp)、[RAC](../../engine/acmodel/rac_model.cpp)、[CRIEPI](../../engine/acmodel/criepi_model.cpp)、[DUCT](../../engine/acmodel/duct_central_model.cpp)、[LATENT](../../engine/acmodel/latent_evaluate_model.cpp)。
SPEC-12-001：kW→W。SPEC-12-002：例外捕捉後の0出力。SPEC-12-003：電力量は後処理。
既存 [acmodel テスト](../../engine/solver/tests_cpp/test_acmodel_core.cpp) の全モデル実行は今回未実施。
2026-09-17：主計算式と呼出し境界・失敗時の取扱いを初稿化。
