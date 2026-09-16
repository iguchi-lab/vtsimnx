# 第七章 換気計算

対象 commit・版・確認日は[第一章](01_general.md)に同じ。文書版：本文初稿 0.2。\
[全体目次](README.md)

## 1. 適用範囲・入力

calc_p の未知圧力を求め、換気枝の体積流量を更新する。固定圧力ノードは外部リザーバとして扱い、流量収支ゼロを要求しない。流量は m³/s、圧力差は Pa、高さは m、密度は kg/m³ とする。

## 2. 圧力差及び収支

source、target の高さ補正を含む圧力差は式 (7-1) とする。

```math
\Delta p=(p_s-\rho_s g h_{\rm from})-(p_t-\rho_t g h_{\rm to}). \tag{7-1}
```

未知圧力ノード i で、流入を正に集計した体積収支 r_i をゼロにする。温度による密度の影響は高さ補正の評価に入り、流量式の密度定数とは区別する。風圧を使用する場合は境界圧力へ反映した入力を用いる。

## 3. 流路式

s=sign(Δp)、ε=archenv::TOLERANCE_SMALL とする。

開口部は式 (7-2) による。alpha は無次元、A は m²、ρ₀ は実装の乾き空気密度定数。

```math
K=\alpha A\sqrt{2/\rho_0},\qquad
q=\begin{cases}sK\sqrt{|\Delta p|}&|\Delta p|\ge\epsilon\\
K\sqrt{\epsilon}\,\Delta p/\epsilon&|\Delta p|<\epsilon.\end{cases}\tag{7-2}
```

隙間は式 (7-3) による。n=0 が渡された場合、この関数内では n=1 に置換する。

```math
q=\begin{cases}s a|\Delta p|^{1/n}&|\Delta p|\ge\epsilon\\
a\epsilon^{1/n-1}\Delta p&|\Delta p|<\epsilon.\end{cases}\tag{7-3}
```

圧損要素は K=A√(2/(ρ₀ k_total)) を式 (7-2) の K に用いる。k_total が正でない場合、friction_factor>0、length≥0、diameter>0 の条件で k_total=f L/D+ζ を使う。A 又は最終 k_total が正でない場合、流量関数は0を返す。入力検証による拒否とは別の関数内処理である。

## 4. ファン

d=−Δp、τ=archenv::TOLERANCE_MEDIUM、S(d,b)={tanh((d−b)/τ)+1}/2 とする。

```math
\begin{aligned}
w_1&=S(d,p_{\max}+\tau),\\
w_2&=S(d,p_1+\tau)(1-w_1),\\
w_3&=S(d,\tau)(1-S(d,p_1+\tau)),\\
w_4&=1-S(d,\tau),\\
q&=w_2 f_2+w_3 f_3+w_4 q_{\max}.
\end{aligned}\tag{7-4}
```

f₂=q₁(d−p_max)/(p₁−p_max)、ただし p₁=p_max なら q₁とする。f₃=q₁+(q_max−q₁)(d−p₁)/(−p₁)、ただし p₁=0なら q_max とする。単純な区分直線ではなく、上記重みで平滑化した値を返す。

## 5. 有効・固定流量・逆流

統一流量関数は、無効枝なら0、次に固定体積流量枝なら current_vol、その後に型別の流量式を評価する。固定判定は type=fixed_flow、has_prescribed_vol、又は vol 配列が空でない条件のいずれかである。fan 型に vol が存在する場合も固定値の経路が優先される。

逆流は負の q で表し、輸送計算では上流・下流を入れ替える。並列枝を単一枝に潰さず、各枝の流量を収支へ加える。

## 6. 数値解法・合否

非線形圧力計算では Ceres の停止条件と、物理的な流量収支判定を分ける。物理合否は calc_p ノードの収支が全て存在し、有限で、対象数が正、許容値が正、かつ max|r_i|≤ventilationTolerance の条件による。固定境界の収支を判定に加えない。

通常解法で達しない場合は実装のフォールバックへ進み、最終的にも不合格なら停止する。フォールバックの全戦略・選択閾値は本初稿の追加詳細化対象であり、単に「収束するまで反復」とは規定しない。

## 7. 出力・検証・根拠

vent_pressure [Pa]、vent_flow_rate [m³/s] を出力する。開口の非微小差圧では q(−Δp)=−q(Δp)、Δp=0なら0。固定流量は Δp を変えても同じ値とする。

根拠：[flow_calculation.h](../../engine/solver/core/ventilation/flow_calculation.h)、[pressure_balance.h](../../engine/solver/core/ventilation/pressure_balance.h)、[ventilation_network.cpp](../../engine/solver/network/ventilation_network.cpp)。
SPEC-07-001：小差圧の線形化。SPEC-07-002：固定・無効の優先順。SPEC-07-003：体積収支合否。
[既存流量テスト](../../engine/solver/tests_cpp/test_flow_math.cpp)、[並列枝テスト](../../engine/solver/tests_cpp/test_vent_parallel_branch_flow_rates.cpp)。今回 C++ 実行は未実施。
2026-09-17：流量式と合否条件を実装から初稿化。
