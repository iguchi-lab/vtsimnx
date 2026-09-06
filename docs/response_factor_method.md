# 応答係数法：壁体の履歴を用いた非定常熱伝導

対象はパッケージ v1.7.2、実装 commit `db107ac276d9c0a9949bbe4358d06cb8b0595fad`。
入力の全体構成は [surface_usage.md](surface_usage.md)、解法の位置づけは
[建築環境の基礎](building_environment_engineering_basics.md) を参照してください。

## 1. 何を表す方法か

壁には熱容量があるため、現在の両面温度が同じでも、過去の加熱・冷却によって
現在の表面熱流は異なります。応答係数法は、この履歴を係数と過去の値で表します。
壁内部の全温度を建物全体の未知数として解かずに、両表面の熱収支へ接続できます。

一般の応答係数表現は温度履歴の畳み込みであり、過去の熱流を含む有限次数の再帰式は
**伝導伝達関数（CTF: conduction transfer function）**と呼ばれます。
この関係と状態空間からの係数生成は、[EnergyPlus Engineering Reference 24.1](https://bigladdersoftware.com/epx/docs/24-1/engineering-reference/conduction-through-the-walls.html)
にも説明されています。ただし、vtsimnxの空間分割・離散化・初期化は下記の独自実装です。
他ソフトのCTFと同じ係数や精度になるとは限りません。

## 2. 実装の式と符号

両面を source（s）と target（t）、一定時間刻みを $\Delta t$、時刻番号を $k$、
面積を $A$ とします。両端の $q_s,q_t$ はともに**表面から壁体へ入る向きを正**とする熱流 [W] です。

$$
q_s^k=A\sum_{j=0}^{m_s}(a_{s,j}T_s^{k-j}+b_{s,j}T_t^{k-j})
      +\sum_{j=1}^{r_s}c_{s,j}q_s^{k-j}
$$

$$
q_t^k=A\sum_{j=0}^{m_t}(a_{t,j}T_t^{k-j}+b_{t,j}T_s^{k-j})
      +\sum_{j=1}^{r_t}c_{t,j}q_t^{k-j}
$$

| 記号 | 入力配列・単位 | 添字 |
|---|---|---|
| $a_s,b_s$ | `resp_a_src`, `resp_b_src` [W/(m²·K)] | 配列 `[0]` が現在、`[1]` が1ステップ前 |
| $a_t,b_t$ | `resp_a_tgt`, `resp_b_tgt` [W/(m²·K)] | 自面温度が `a`、反対面温度が `b` |
| $c_s,c_t$ | `resp_c_src`, `resp_c_tgt` [無次元] | 配列 `[0]` が1ステップ前の熱流 |
| $T_s,T_t$ | 表面温度 [°C] | 室空気温度とは区別 |
| $A$ | `area` [m²] | 正の値が必要 |

面積は温度項の係数に一度だけ掛けます。実装の熱流履歴は既に W なので、
`resp_c_*` の項へ面積を再度掛けません。
自面の `a` と相互項 `b` は符号を含んだ係数です。
各側の `a` と `b` は同じ非零長、`c` は空配列またはその長さ−1が必要です。

内部発熱のない保存的な壁では、$q_s+q_t$ が壁内蓄熱率です。
定常状態では $q_s=-q_t$ ですが、非定常時には一般に一致しません。
節点熱収支には source 側へ $-q_s$、target 側へ $-q_t$ を加えます。
**現在の共通出力 `heat_rate` は $(q_s+q_t)/2$ です。**
これは貫流熱量ではなく、保存的なモデルなら蓄熱率の半分に相当します。
例えば定常貫流中にも0となるため、壁を通過する熱量の評価にこの値をそのまま用いないでください。
両端熱流は内部の `current_q_src`, `current_q_tgt` に保持されます。

## 3. 係数の生成とRC法との関係

自動生成では物性一定の一次元多層壁を仮定します。層 $\ell$ ごとに中心温度を一つ置き、
単位面積熱容量 $C_\ell=c_{v,\ell}d_\ell$ [J/(m²·K)]、
半層抵抗 $R_{\ell,1/2}=d_\ell/(2\lambda_\ell)$ [m²·K/W] を用います。
隣接中心間の抵抗は両層の半層抵抗の和です。表面熱伝達抵抗は係数へ含めず、別の枝で扱います。

内部温度ベクトル $\mathbf{x}$、表面温度入力 $\mathbf{u}=(T_s,T_t)^\mathsf{T}$、
熱流密度出力 $\mathbf{y}=(q_s/A,q_t/A)^\mathsf{T}$ により、

$$
\dot{\mathbf{x}}=\mathbf{F}\mathbf{x}+\mathbf{B}\mathbf{u},\qquad
\mathbf{y}=\mathbf{C}\mathbf{x}+\mathbf{D}\mathbf{u}
$$

を組み立て、**後退Euler法**で離散化します。

$$
\mathbf{x}^k=\mathbf{F}_d\mathbf{x}^{k-1}+\mathbf{B}_d\mathbf{u}^k,\quad
\mathbf{F}_d=(\mathbf{I}-\Delta t\mathbf{F})^{-1},\quad
\mathbf{B}_d=\mathbf{F}_d\Delta t\mathbf{B}
$$

出力は $\mathbf{y}^k=\mathbf{C}\mathbf{x}^k+\mathbf{D}\mathbf{u}^k$ です。
したがって現在入力への直接係数は $\mathbf{D}+\mathbf{C}\mathbf{B}_d$ を含みます。
$\det(z\mathbf{I}-\mathbf{F}_d)=z^n+d_1z^{n-1}+\cdots+d_n$ から
熱流履歴係数 $c_j=-d_j$ を求め、インパルス応答から温度履歴係数を求めます。
これは連続熱伝導方程式の厳密解ではなく、**離散RCモデルの入出力表現への変換**です。

| 設定 | 実装 | 注意点 |
|---|---|---|
| `response_method="arx_rc"`（既定） | 全層中心の状態をARX（自己回帰・外生入力）形式へ変換 | 通常、n層なら温度係数n+1個、熱流係数n個 |
| `response_method="modal_expsum"` | 離散系の固有モードを寄与と遅さで選択 | `response_terms` は残すモード数。省略時は全数、層数が上限 |
| `layer_method="rc"` | 容量を各層両端へ分けて全体回路網に展開 | 応答係数生成時の層中心配置とは異なる |

全次数ARXと、その生成元のRC状態空間は、履歴・初期状態が整合し丸め誤差が小さければ同じ入出力を表します。
一方、通常の `layer_method="rc"` と応答係数法は容量配置が異なり、同じ層入力だけで過渡応答の完全一致を要求できません。
`response_terms` による次数削減の有無は、生成された係数で確認してください。
`arx_rc` は `response_terms` で次数を減らしません。

## 4. 入力例

次を、節点「室」「外部」を持ち、`simulation.index.timestep` を指定した入力の `surfaces` へ設定します。
数値は手順説明用で、実製品の推奨物性ではありません。

```python
surfaces = [{
    "key": "室->外部", "part": "wall", "area": 10.0,
    "layer_method": "response",
    "response_method": "arx_rc",
    "alpha_i": 4.4, "alpha_o": 20.3,
    "layers": [
        {"name": "蓄熱層", "lambda": 1.4, "t": 0.10, "v_capa": 2.0e6},
        {"name": "断熱層", "lambda": 0.04, "t": 0.05, "v_capa": 3.0e4},
    ],
}]
```

全体の既定は `surface_layer_method="response"` でも指定できます。
壁は両面の2節点と応答係数枝、両側の対流枝へ展開されます。
室内放射・日射は別途表面へ接続します。
明示的な `response` 辞書があれば、自動係数生成より優先します。
このときも展開経路を明確にするため `layers` と `layer_method` を指定してください。
係数に含めた面積・表面抵抗・時間刻みを記録し、二重計上を避けます。

## 5. 初期化、安定性、適用限界

- **履歴初期化**：過去の表面温度はそれぞれの初期表面温度で埋め、過去の熱流は0にします。
  両面温度が異なると定常貫流の履歴とは整合しません。必要な予備計算期間を確保し、
  期間延長による評価区間の変化が小さいことを確認します。
- **履歴更新**：同一ステップの連成・空調反復中は履歴を固定し、ステップ受理時に一度だけ更新します。
  係数は一定の時間刻みに対応します。刻みを変えた計算では係数を再生成し、履歴も再初期化します。
- **定常化への切替**：自動生成で `sum(resp_c_src) > 0.9999` なら、
  $U_w=(\sum d_\ell/\lambda_\ell)^{-1}$ による `a=[Uw]`, `b=[-Uw]`, `c=[]` へ切り替わります。
  これは動的精度を保った安定化ではなく、壁の蓄熱を省く処理です。
  零容量層を含む全次数生成でもこの経路へ入るため、生成係数を確認してください。
- **次数削減**：現在の `modal_expsum` は除去したモードの定常寄与を補償しません。
  等温両面で非零熱流を生じるなど、定常熱収支を損なう場合があります。
  全次数との比較、等温時ゼロ熱流、定常U値、両端熱流の収支を確認せずに採用しないでください。
- **物性・形状**：自動生成には `lambda>0`, `t>0`, `v_capa>=0` が必要です。
  空気層・通気層フラグはこの経路では未対応でエラーになります。
  相変化、温湿度依存物性、壁内水分移動、二次元熱橋を直接表現しません。
- **利用者係数**：配列形状の検査だけでは安定性・受動性・相反性を保証しません。
  自動生成では相互係数を平均してそろえますが、任意入力係数の物理的整合は別に確認が必要です。
  温度原点の変更にも、係数の定常整合性と初期履歴の整合が必要です。

## 6. 確認する項目と実装参照

最低限、(1) 等温両面の熱流、(2) 定常U値、(3) 温度ステップ・周期入力、
(4) 両端熱流と蓄熱の収支、(5) 時間刻み・空間分割・次数・予備計算期間への依存を調べます。
既存のRC/応答係数比較テストは容量なしのケースであり、
多層壁の非定常精度の検証とは区別します。
今回行った係数生成元との数値比較と未解決点は [点検記録](documentation_review_2026-09.md) に記載します。

| 対象 | コード |
|---|---|
| 状態空間の構築 | [`surface_rc.py`](../engine/app/builder/surface_rc.py) |
| 係数生成・次数削減・定常化 | [`surface_response.py`](../engine/app/builder/surface_response.py) |
| 入力係数の検査 | [`branches_parser.cpp`](../engine/solver/parser/branches_parser.cpp) |
| 初期化・履歴確定 | [`thermal_network.cpp`](../engine/solver/network/thermal_network.cpp) |
| 履歴項・両端熱流 | [`thermal_direct_response.h`](../engine/solver/core/thermal/thermal_direct_response.h) |
| 全体収支・共通出力 | [`thermal_edge_physics.h`](../engine/solver/core/thermal/thermal_edge_physics.h) |

詳細入力仕様は [engineの応答係数仕様](../engine/docs/thermal_response_factor.md) も参照してください。
