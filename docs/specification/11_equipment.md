# 第十一章 設備モデル及び制御

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲・入力

空調機と熱交換換気を扱う。空調では吸込 in_node、吹出側、外気 outside_node、制御対象 set_node を区別する。set_node が吸込・吹出の室と異なる遠隔制御を許す。pre_temp は°C、pre_rh は%、ac_spec の能力はkW、風量はm³/sとする。

## 2. 空調の設定値と ON/OFF

requestedSetpoint は要求設定、effectiveSetpoint は能力制限後の実効設定である。ON/OFF と温度固定行で参照する役割を分ける。

温度判定は d=T−T_set、b=max(tolerance,kRestartBandK) とする。|d|≤b では現在の状態を保持する。帯域外では HEATING は d<0、COOLING は d>0、AUTO は真を ON 条件とする。不正モードは例外とする。

運転中で有限な required_heat_w を使える場合、Q_on=max(非負の負荷不感帯, 有効な最低能力) を基準とする。最低能力を使う場合は暖房 Q_req≥Q_on、冷房 Q_req≤−Q_on、AUTO |Q_req|≥Q_on。最低能力を使わない場合は等号を除く。最低能力保持が有効な経路では ON を保持する。required_heat_w を使う条件と待ち時間等は controller 側で評価する。

## 3. 顕熱・潜熱処理

乾き空気の顕熱処理の基礎は ρ₀ c_pa |q| |T_out−T_in| [W] とし、モード方向と実際の制御状態に応じて処理量を評価する。潜熱処理で決まった supplyX を空調ノードの固定湿度へ反映する。

| 潜熱方式 | 現行処理 |
|---|---|
| rh95（既定） | 吹出温度と95%RHを基に supplyX を評価 |
| bf | BFを混合比としてコイル条件を評価。既定0.2、[0,0.99]に制限。吹出RH>100%ならrh95へ戻す |
| coil_aoaf / aoaf / literature | コイル面積と風速を使う式。Af既定0.133 m²、Ao既定4.84 m² |
| none | 潜熱0、吸込湿度を通す |
| 有効な pre_rh | 冷房で吸込湿度が目標を超えると理想除湿経路を先に使う |

理想除湿の目標 x_sp は吸込温度と pre_rh から求める。x_in>x_sp なら supplyX=x_sp、除湿量は ρ₀|q|(x_in−supplyX)、潜熱はこれに吹出温度の蒸発潜熱を乗じる。理想除湿の湿度設定そのものは能力を無視する処理であり、機器の実コイル性能の保証ではない。

湿りエンタルピー有効時は全熱を ρ₀|q|Δh で評価し、顕熱・潜熱に分ける。totalHeatCapacity は max(0,Q_S)+max(0,Q_L)。能力上限判定は顕熱だけでなく全熱を対象とする。

## 4. 能力と風量の反復

機器仕様の Q.<mode>.max、rtd、mid を優先順に参照する。要求を満たせない場合は実効設定温度等を修正して再計算する。能力判定後に DUCT_CENTRAL の固定送風枝を調整する。

目標風量の基本比は clamp(Q_basis/Q_rtd,0,1)。正の負荷が最低能力より小さい場合は最低能力の比を使う。能力制限中・設定未達では Q_max、設定維持中では |required_heat_w| を基準とする経路がある。計測コイル熱を常に基準にする仕様ではない。還気・吹出の固定流量を合わせて更新し、変化時は外側を再計算する。

## 5. 熱交換換気の境界

OA は外気取入、RA は還気元、SA は給気先、EA は排気先とする。短縮指定 outdoor+room は OA=EA=outdoor、SA=RA=room に展開する。

給気境界 H の温湿度は式 (11-1) による。

$$ T_H=T_{OA}+\eta_t(T_{RA}-T_{OA}),\quad
x_H=x_{OA}+\eta_x(x_{RA}-x_{OA}). \tag{11-1} $$

sensible は eta_x=0、total は指定値を使う。C++ readParams は効率を[0,1]へ制限する。生成された排気ジャンクションの温湿度は RA と同じにする。排気側を熱交換後の平衡状態に求め直すモデルではない。

## 6. 回収熱出力

q_S、q_E は接続から得る風量の絶対値。実装の有効風量は式 (11-2) とする。

$$ q_{\rm eff}=\begin{cases}\min(q_S,q_E)&q_E>0\\q_S&q_E\le0.\end{cases}\tag{11-2} $$

$$ Q_{HRV,S}=\rho_0 c_{pa}q_{\rm eff}(T_H-T_{OA}),\quad
Q_{HRV,L}=\rho_0(2.501\times10^6)q_{\rm eff}(x_H-x_{OA}). \tag{11-3} $$

出力単位はW。排気0でも必ず回収熱0になるわけではない。給気温湿度の式 (11-1) に q_eff の比を掛ける処理はない。不均衡風量時の厳密な両側熱収支を保証しない。

バイパス、着霜防止、ファン電力、効率の温湿度依存はこの熱交換換気モデルでは扱わない。

## 7. 出力・検証・根拠

空調の処理熱、電力、COP は機器別系列、HRV は hrv_sensible_heat、hrv_latent_heat。例：OA=0°C、RA=20°C、eta_t=0.7なら給気14°C。q_S=0.1、q_E=0の場合、回収顕熱は1.2×1006×0.1×14=1690.08 W。

根拠：[ON/OFF](../../engine/solver/aircon/aircon_onoff.cpp)、[潜熱](../../engine/solver/aircon/aircon_latent.cpp)、[制御](../../engine/solver/aircon/aircon_controller.cpp)、[HRV](../../engine/solver/hrv/hrv_controller.cpp)。
SPEC-11-001：温度帯内の状態保持。SPEC-11-002：HRV の排気0分岐。SPEC-11-003：理想除湿優先。
待ち時間の全分岐と各潜熱方式の詳細係数は追加詳細化対象。今回 C++ 実行は未実施。
2026-09-17：設備の主要式、制御、制約を初稿化。
