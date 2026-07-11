from __future__ import annotations

import numpy as np

from .logger import get_logger
from .surface_constants import DEFAULT_ALPHA_I, DEFAULT_ALPHA_O
from .surface_layers import (
    _branch_log_detail,
    _layer_flag,
    _leading_node_key_from_layer_key,
    _node_log_detail,
)
from .surface_rc import _build_rc_continuous_abcd, _layers_to_rc_arrays

logger = get_logger(__name__)


def _auto_response_coefficients_from_layers(
    layers: list[dict],
    time_step: float,
    *,
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> dict:
    """
    layers に含まれる lambda, t, v_capa と time_step[s] から response_conduction 係数を自動生成する。
    v_capa は体積熱容量 [J/(m³·K)]（core vtsimnx/materials の public materials と同じ単位）。

    生成する係数は「熱流密度 q'' [W/m2]」に対するもの（面積Aは solver 側で掛けて [W] にする想定）。
    resp_c_* は無次元（過去の q'' あるいは q に掛ける係数。q = A*q'' なので同じ係数で整合する）として扱う。

    モデル:
    - 各層の中心温度を状態（n層なら n状態）
    - 表面温度 Ts,Tt を入力（2入力）
    - 表面熱流密度 q''_src, q''_tgt を出力（2出力、どちらも「その表面から壁体へ入る向き」を正）
    - 時間離散は Backward Euler: x_k = Ad x_{k-1} + Bd u_k, y_k = C x_k + D u_k

    その離散系を各出力ごとに ARX 形式へ変換:
      y(k) = sum_j b_j * u(k-j) + sum_i c_i * y(k-1-i)

    response_method:
    - "arx_rc"（既定）: RC連鎖（状態数=n=層数）をそのまま ARX 化（次数=n）
    - "modal_expsum": 離散系 Ad を固有分解し、寄与の大きいモードを response_terms 個だけ残して ARX 化
      （実装上は「離散指数項の和（λ^k）」で近似 → 項数を明示的に制御できる）
    """
    if time_step is None:
        raise ValueError("time_step is required for auto response coefficient generation")
    dt = float(time_step)
    if dt <= 0:
        raise ValueError(f"time_step must be positive, got {dt}")

    n, lam, thk, vc, C, R_half, R_between = _layers_to_rc_arrays(layers)
    A, B, Cmat, Dmat = _build_rc_continuous_abcd(n, C, R_half, R_between)

    # 離散化（Backward Euler）: x_k = (I - dt*A)^-1 x_{k-1} + (I - dt*A)^-1 (dt*B) u_k
    eye = np.eye(n, dtype=float)
    M = eye - dt * A
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"failed to invert (I - dt*A): {e}") from e
    Ad = Minv
    Bd = Minv @ (dt * B)

    # ------------------------------------------------------------
    # 係数生成に使う離散系（Ad,Bd,C,D）を、必要ならモードで縮約する
    # ------------------------------------------------------------
    Ad_use = Ad
    Bd_use = Bd
    C_use = Cmat
    D_use = Dmat

    method = str(response_method or "arx_rc").strip().lower()
    terms = response_terms
    if terms is not None:
        try:
            terms = int(terms)
        except Exception:
            raise ValueError(f"response_terms must be int, got {response_terms!r}")
        if terms <= 0:
            raise ValueError(f"response_terms must be positive, got {terms}")

    if method in ("modal_expsum", "expsum", "modal"):
        # terms 未指定ならフル（=従来と同じ）
        m = n if terms is None else min(n, terms)
        if m < n:
            # Ad = V Λ V^{-1} として、選択モードのみ残す（複素モードは現状フォールバック）
            w, V = np.linalg.eig(Ad)
            if np.max(np.abs(np.imag(w))) > 1e-10:
                logger.warning("auto response: complex eigenvalues detected; falling back to full-order ARX")
            else:
                w = np.real(w)
                V = np.real(V)
                try:
                    Vinv = np.linalg.inv(V)
                except np.linalg.LinAlgError:
                    logger.warning("auto response: eigenvector matrix not invertible; falling back to full-order ARX")
                else:
                    # モード寄与の簡易スコア
                    #   contrib ~ |(C V)_mode| * |(V^{-1} B)_mode|
                    Cv = Cmat @ V                 # (2, n)
                    Bin = Vinv @ Bd               # (n, 2)
                    # slow mode優先（|λ|→1 ほど重み増）
                    slow = 1.0 / np.maximum(1e-9, (1.0 - np.minimum(np.abs(w), 0.999999)))
                    weight = (np.sum(np.abs(Cv), axis=0) * np.sum(np.abs(Bin), axis=1)) * slow
                    idx = np.argsort(weight)[::-1][:m]
                    idx = np.array(sorted(idx.tolist()))  # 安定のため昇順にそろえる

                    Ad_use = np.diag(w[idx])
                    Bd_use = Bin[idx, :]
                    C_use = Cv[:, idx]
                    D_use = Dmat

                    # 次数が変わる
                    n = m

    # 伝達関数の分母（共通）: det(zI - Ad_use)
    den = np.poly(Ad_use)  # len=n+1, den[0]=1
    den = np.array(den, dtype=float)

    # AR係数（qの過去項）: y(k) = ... + sum_i c[i]*y(k-1-i), c[i] = -den[i+1]
    c_ar = (-den[1:]).tolist()
    sum_c = float(np.sum(c_ar)) if len(c_ar) else 0.0

    # impulse response h[k] for k=0..n (per output, per input)
    # h[0] = D, h[k] = C * Ad^(k-1) * Bd
    # そこから b_j を生成: b[k] = h[k] + sum_{i=1..k} den[i]*h[k-i]
    # （k<=n なので min(k,n)=k）
    def compute_b_for(output_idx: int, input_idx: int) -> list[float]:
        # impulse response h[k] for k=0..n (u[0]=1, u[k>0]=0, x[-1]=0)
        # x[0] = Bd*u[0], y[0] = C*x[0] + D*u[0] = C*Bd + D
        # x[k] = Ad^k * Bd * u[0], y[k] = C*Ad^k*Bd (k>=1)
        h = np.zeros(n + 1, dtype=float)
        h[0] = float(D_use[output_idx, input_idx] + (C_use[output_idx, :] @ Bd_use[:, input_idx]))
        Apow = np.eye(n, dtype=float)
        for k in range(1, n + 1):
            Apow = Apow @ Ad_use  # Ad^k
            h[k] = float(C_use[output_idx, :] @ (Apow @ Bd_use[:, input_idx]))

        b = np.zeros(n + 1, dtype=float)
        for k in range(0, n + 1):
            s = h[k]
            for i in range(1, k + 1):
                s += den[i] * h[k - i]
            b[k] = s
        return b.tolist()

    # output0=q_src: input0=Ts, input1=Tt
    a_src = compute_b_for(0, 0)
    b_src = compute_b_for(0, 1)
    # output1=q_tgt: input0=Ts, input1=Tt
    b_tgt = compute_b_for(1, 0)  # coefficient on Ts
    a_tgt = compute_b_for(1, 1)  # coefficient on Tt

    # 数値安定性: 地中など極端に遅い系では sum(c) が 1 に非常に近くなり、丸め誤差で発散しやすい。
    # その場合は動的項（resp_c）を使わず、定常U値のみ（メモリなし）にフォールバックする。
    if sum_c > 0.9999:
        R_total = float(np.sum(thk / lam))
        U = 1.0 / R_total if R_total > 0 else 0.0
        return {
            "resp_a_src": [U],
            "resp_b_src": [-U],
            "resp_c_src": [],
            "resp_a_tgt": [U],
            "resp_b_tgt": [-U],
            "resp_c_tgt": [],
        }

    # 重要: 2端子の相互項（Tt -> q_src と Ts -> q_tgt）は、受動・相反系では一致する（Y12=Y21）。
    # 数値誤差でわずかにずれると、ソルバ側の「対称行列前提」の疎直接法で残差が悪化しやすい。
    # そこで相互項は平均して強制的に一致させる。
    b_cross = (np.array(b_src, dtype=float) + np.array(b_tgt, dtype=float)) * 0.5
    b_src = b_cross.tolist()
    b_tgt = b_cross.tolist()

    return {
        "resp_a_src": a_src,
        "resp_b_src": b_src,
        "resp_c_src": c_ar,
        "resp_a_tgt": a_tgt,
        "resp_b_tgt": b_tgt,
        "resp_c_tgt": c_ar,
    }


def _process_surface_response_method(
    surface: dict,
    start_node: str,
    end_node: str,
    i_prefix: str,
    o_prefix: str,
    a: float,
    initial_t_by_node_key: dict[str, float] | None,
    time_step: float | None,
    response_method: str,
    response_terms: int | None,
    verbose: bool = True,
) -> tuple[list, list]:
    """応答係数法: 両端の表面ノードのみ生成し、内部は response_conduction ブランチで表現。"""
    surface_key = surface.get("key", "?")
    for idx, layer in enumerate(surface.get("layers", [])):
        if not isinstance(layer, dict):
            continue
        if _layer_flag(layer, "air_layer") or _layer_flag(layer, "ventilated_air_layer"):
            raise ValueError(
                f"surface {surface_key}: layer[{idx}] has hollow/ventilated flag, "
                "which is supported only when layer_method='rc'"
            )

    resp = surface.get("response")
    if resp is None:
        rm = surface.get("response_method", response_method)
        rt = surface.get("response_terms", response_terms)
        resp = _auto_response_coefficients_from_layers(
            surface["layers"],
            time_step=time_step,
            response_method=str(rm) if rm is not None else "arx_rc",
            response_terms=rt,
        )
    if not isinstance(resp, dict):
        raise ValueError(f"surface {surface_key}: layer_method='response' requires 'response' dict")
    for kreq in ("resp_a_src", "resp_b_src", "resp_a_tgt", "resp_b_tgt"):
        if kreq not in resp:
            raise ValueError(f"surface {surface_key}: response missing required '{kreq}'")

    i_surface = f"{i_prefix}_s"
    o_surface = f"{o_prefix}_s"
    node_names = [i_surface, o_surface]
    node_types = ["surface", "surface"]
    alpha_i = surface.get("alpha_i", DEFAULT_ALPHA_I)
    alpha_o = surface.get("alpha_o", DEFAULT_ALPHA_O)

    nodes: list = []
    for i, node in enumerate(node_names):
        if verbose:
            logger.info(f"　ノード【{node}】 を追加します。{_node_log_detail(None, node_types[i])}")
        node_dict = {
            "key": node,
            "calc_t": True,
            "type": "layer",
            "subtype": node_types[i],
        }
        if initial_t_by_node_key:
            lead = _leading_node_key_from_layer_key(node)
            if lead in initial_t_by_node_key:
                node_dict["t"] = initial_t_by_node_key[lead]
        nodes.append(node_dict)

    thermal_branches: list = [
        {"key": f"{start_node}->{i_surface}", "conductance": a * alpha_i, "subtype": "convection"},
        {"key": f"{o_surface}->{end_node}", "conductance": a * alpha_o, "subtype": "convection"},
        {
            "key": f"{i_surface}->{o_surface}",
            "type": "response_conduction",
            "subtype": "conduction",
            "area": float(surface["area"]),
            **resp,
        },
    ]
    if verbose:
        logger.info(f"　応答係数熱ブランチ【{thermal_branches[-1]['key']}】を追加します。{_branch_log_detail(thermal_branches[-1])}")
    return nodes, thermal_branches
