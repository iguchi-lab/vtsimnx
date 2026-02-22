from __future__ import annotations

import numpy as np

from .logger import get_logger
from .utils import CHAIN_DELIMITER, ensure_timeseries
from .validate import ConfigFileError

logger = get_logger(__name__)

# ------------------------------
# 表面の種類の対応関係 と 物性定数
# ------------------------------
SURFACE_PAIR = {
    "wall": "wall",
    "floor": "ceiling",
    "ceiling": "floor",
    "glass": "glass",
}

DEFAULT_ALPHA_I = 4.4   # 室内側表面の対流熱伝達率
DEFAULT_ALPHA_O = 20.3  # 室外側表面の対流熱伝達率
DEFAULT_ALPHA_R = 4.7   # 放射熱伝達率
DEFAULT_ETA_LW = 0.9    # 長波放射の吸収率（室内放射回路）


def _scalar_initial_temperature(value):
    """
    ノード設定の `t`（スカラー or 時系列）から「初期値（スカラー）」を取り出す。

    背景:
    - solver 側は `t` が配列だと timestep ごとに `current_t` を更新する。
      `calc_t=True` のノードでは、その後に計算結果で上書きされるため、
      配列 `t` は「境界条件」というより「各ステップの初期推定値」を与える意味合いになる。
    - ここでは “表面分割で自動生成した層ノード” の初期値を素直に設定したいだけなので、
      時系列が来ても先頭要素（初期値）だけを採用する。
    """
    if value is None:
        return None
    # numpy/pandas は builder 側で list に正規化される想定だが、念のため対応
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return None


def _leading_node_key_from_layer_key(layer_key: str) -> str:
    """
    生成層ノードの key（例: 'A-B_wall_s'）から、先頭に現れるノード名（例: 'A'）を返す。
    """
    return str(layer_key).split("-", 1)[0]


def get_node_prefix(surface: dict) -> tuple[str, str, str, str]:
    key = surface.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ConfigFileError(f"surface.key must be a non-empty string, got {key!r}")
    parts = key.split(CHAIN_DELIMITER)
    if len(parts) < 2:
        raise ConfigFileError(
            f"surface.key must contain '{CHAIN_DELIMITER}' (e.g. 'RoomA{CHAIN_DELIMITER}Outdoor'), got {key!r}"
        )
    start_node = parts[0]
    end_node = parts[1]
    start_part     = surface["part"]
    end_part       = SURFACE_PAIR[start_part]
    comment        = surface.get("comment", "").strip()
    comment_suffix = f"({comment})" if comment else ""
    i_prefix       = f"{start_node}-{end_node}{comment_suffix}_{start_part}"
    o_prefix       = f"{end_node}-{start_node}{comment_suffix}_{end_part}"
    return start_node, end_node, i_prefix, o_prefix


def _auto_response_coefficients_from_layers(
    layers: list[dict],
    time_step: float,
    *,
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> dict:
    """
    layers に含まれる lambda, t, v_capa と time_step[s] から response_conduction 係数を自動生成する。

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
    if not layers:
        raise ValueError("layers is empty")

    # 物性 -> 抵抗/容量（単位: R[m2K/W], C[J/m2K]）
    n = len(layers)
    lam = np.array([float(x["lambda"]) for x in layers], dtype=float)
    thk = np.array([float(x["t"]) for x in layers], dtype=float)
    vc  = np.array([float(x["v_capa"]) for x in layers], dtype=float)
    if np.any(lam <= 0) or np.any(thk <= 0) or np.any(vc < 0):
        raise ValueError("invalid layer properties: lambda>0, t>0, v_capa>=0 required")

    # 各層中心の熱容量（面積1m2あたり）
    C = vc * thk  # [J/m2K]

    # 各層の半抵抗
    R_half = (thk / 2.0) / lam  # [m2K/W]

    # 連結抵抗（中心-中心）
    R_between = np.zeros(max(n - 1, 0), dtype=float)
    for i in range(n - 1):
        R_between[i] = R_half[i] + R_half[i + 1]

    # 連続系: x_dot = A x + B u, u=[Ts, Tt], y=[q''src, q''tgt]
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, 2), dtype=float)

    def invR(r):
        return 0.0 if r == 0 else 1.0 / r

    # src側境界
    g_s = invR(R_half[0])
    if n == 1:
        # C1 dT1/dt = g_s*(Ts - T1) + g_t*(Tt - T1)
        g_t = invR(R_half[0])  # same layer half to target
        A[0, 0] = -(g_s + g_t) / C[0] if C[0] > 0 else 0.0
        B[0, 0] = g_s / C[0] if C[0] > 0 else 0.0
        B[0, 1] = g_t / C[0] if C[0] > 0 else 0.0
    else:
        g_12 = invR(R_between[0])
        A[0, 0] = -(g_s + g_12) / C[0] if C[0] > 0 else 0.0
        A[0, 1] = (g_12) / C[0] if C[0] > 0 else 0.0
        B[0, 0] = g_s / C[0] if C[0] > 0 else 0.0

        # interior
        for i in range(1, n - 1):
            g_im1 = invR(R_between[i - 1])
            g_ip1 = invR(R_between[i])
            if C[i] > 0:
                A[i, i - 1] = g_im1 / C[i]
                A[i, i] = -(g_im1 + g_ip1) / C[i]
                A[i, i + 1] = g_ip1 / C[i]
            # Bは0

        # tgt側
        g_t = invR(R_half[-1])
        g_nm1 = invR(R_between[-1])
        if C[-1] > 0:
            A[-1, -2] = g_nm1 / C[-1]
            A[-1, -1] = -(g_nm1 + g_t) / C[-1]
            B[-1, 1] = g_t / C[-1]

    # 出力: q''src = (Ts - T1)/R_half0, q''tgt = (Tt - Tn)/R_halfN
    Cmat = np.zeros((2, n), dtype=float)
    Dmat = np.zeros((2, 2), dtype=float)
    Cmat[0, 0] = -invR(R_half[0])
    Dmat[0, 0] = invR(R_half[0])
    Cmat[1, -1] = -invR(R_half[-1])
    Dmat[1, 1] = invR(R_half[-1])

    # 離散化（Backward Euler）: x_k = (I - dt*A)^-1 x_{k-1} + (I - dt*A)^-1 (dt*B) u_k
    I = np.eye(n, dtype=float)
    M = I - dt * A
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


def process_surface(
    surface: dict,
    initial_t_by_node_key: dict[str, float] | None = None,
    *,
    time_step: float | None = None,
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> tuple[list, list]:
    nodes: list = []
    thermal_branches: list = []

    start_node, end_node, i_prefix, o_prefix = get_node_prefix(surface)

    a = surface["area"]
    alpha_i = surface.get("alpha_i", DEFAULT_ALPHA_I)
    alpha_o = surface.get("alpha_o", DEFAULT_ALPHA_O)

    layer_method = surface.get("layer_method", "rc")  # "rc"(従来) or "response"(応答係数)

    if "layers" in surface and layer_method == "response":
        # 応答係数法: 両端の表面ノードのみ生成し、内部は response_conduction ブランチで表現
        # - response が無ければ layers + time_step から自動生成する（q'' [W/m2] 系列）
        resp = surface.get("response")
        if resp is None:
            # surface 個別の上書きを許可（無ければ builder 既定値）
            rm = surface.get("response_method", response_method)
            rt = surface.get("response_terms", response_terms)
            resp = _auto_response_coefficients_from_layers(
                surface["layers"],
                time_step=time_step,
                response_method=str(rm) if rm is not None else "arx_rc",
                response_terms=rt,
            )
        if not isinstance(resp, dict):
            raise ValueError(f"surface {surface.get('key','?')}: layer_method='response' requires 'response' dict")
        for kreq in ("resp_a_src", "resp_b_src", "resp_a_tgt", "resp_b_tgt"):
            if kreq not in resp:
                raise ValueError(f"surface {surface.get('key','?')}: response missing required '{kreq}'")

        # 表面ノード（室内側/室外側）
        i_surface = f"{i_prefix}_s"
        o_surface = f"{o_prefix}_s"
        node_names = [i_surface, o_surface]
        node_types = ["surface", "surface"]

        for i, node in enumerate(node_names):
            logger.info(f"　ノード【{node}】 を追加します。")
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

        # 対流（室内側/室外側）
        alpha_i = surface.get("alpha_i", DEFAULT_ALPHA_I)
        alpha_o = surface.get("alpha_o", DEFAULT_ALPHA_O)
        thermal_branches.append(
            {"key": f"{start_node}->{i_surface}", "conductance": a * alpha_i, "subtype": "convection"}
        )
        thermal_branches.append(
            {"key": f"{o_surface}->{end_node}", "conductance": a * alpha_o, "subtype": "convection"}
        )

        # 応答係数ブランチ（壁体内部）
        tb = {
            "key": f"{i_surface}->{o_surface}",
            "type": "response_conduction",
            "subtype": "conduction",
            "area": float(surface["area"]),  # q''[W/m2] を solver 側で q[W]=A*q'' にするため
            **resp,
        }
        logger.info(f"　応答係数熱ブランチ【{tb['key']}】を追加します。")
        thermal_branches.append(tb)

        return nodes, thermal_branches

    if "layers" in surface:
        n = len(surface["layers"])
        q = [a * layer["lambda"] / layer["t"] for layer in surface["layers"]]
        c = [a * layer["v_capa"] * layer["t"] for layer in surface["layers"]]
    else:
        n = 1
        q = [a * surface["u_value"]]
        c = [a * surface.get("a_capacity", 0.0)]

    thermal_mass = (
        [c[0] / 2] + [c[i] / 2 + c[i + 1] / 2 for i in range(n - 1)] + [c[-1] / 2]
    )
    node_types = ["surface"] + ["internal"] * (n - 1) + ["surface"]
    conductance = [a * alpha_i] + [q[i] for i in range(n)] + [a * alpha_o]
    branch_types = ["convection"] + ["conduction"] * (n) + ["convection"]

    node_names = (
        [f"{i_prefix}_s"]
        + [f"{i_prefix}_{i+1}-{i+2}" for i in range(n - 1)]
        + [f"{o_prefix}_s"]
    )

    thermal_node_chain = [start_node] + node_names + [end_node]
    thermal_branch_names = [
        f"{thermal_node_chain[i]}->{thermal_node_chain[i+1]}" for i in range(n + 2)
    ]

    for i, node in enumerate(node_names):
        logger.info(f"　ノード【{node}】 を追加します。")
        node_dict = {
            "key": node,
            "calc_t": True,
            "thermal_mass": thermal_mass[i],
            "type": "layer",
            "subtype": node_types[i],
        }
        # 生成ノードの初期温度: ノード key の先頭に現れるノード（例: A-B... なら A）の初期温度をコピー
        if initial_t_by_node_key:
            lead = _leading_node_key_from_layer_key(node)
            if lead in initial_t_by_node_key:
                node_dict["t"] = initial_t_by_node_key[lead]
        nodes.append(node_dict)

    for i, branch in enumerate(thermal_branch_names):
        logger.info(f"　熱ブランチ【{branch}】を追加します。")
        thermal_branches.append(
            {"key": branch, "conductance": conductance[i], "subtype": branch_types[i]}
        )

    return nodes, thermal_branches


def process_wall_solar(surface: dict, sim_length: int) -> list:
    thermal_branches: list = []
    _, _, _, o_prefix = get_node_prefix(surface)

    heat_generation = surface["area"] * surface.get("eta", 0.8) * np.array(surface["solar"])
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    logger.info(f"　外壁日射熱ブランチ【{branch_key}】を追加します。")
    thermal_branches.append(
        {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
    )
    return thermal_branches


def process_wall_nocturnal(surface: dict, sim_length: int) -> list:
    """
    夜間放射（長波放射）による熱損失を、void から表面ノードへの heat_generation として表現する。

    注意:
    - 熱ブランチは target が必ず実在ノードである必要があるため、`surface->void` は作れない。
      代わりに `void->surface` を作り、heat_generation を負にして「表面から void へ流出」を表す。
    - nocturnal は [W/m2]（または入力系列と同単位）を想定し、面積を掛けて [W] の系列にする。
    """
    thermal_branches: list = []
    _, _, _, o_prefix = get_node_prefix(surface)

    # 設定キーは "nocturnal" を推奨。互換で "night_radiation" も受ける。
    noct = surface.get("nocturnal", surface.get("night_radiation"))
    if noct is None:
        return thermal_branches

    # 表面->void への流出なので負符号
    heat_generation = -surface["area"] * surface.get("epsilon", 0.9) * np.array(noct)
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    logger.info(f"　外壁夜間放射熱ブランチ【{branch_key}】を追加します。")
    thermal_branches.append(
        {"key": branch_key, "heat_generation": heat_generation, "subtype": "nocturnal_loss"}
    )
    return thermal_branches


def process_glass_solar(surface: dict, surfaces: list, sim_length: int) -> list:
    thermal_branches: list = []

    # NOTE:
    # - 同一室（start_node が同じ）の表面に対して配分する。
    # - startswith だと "Room1" と "Room10" のような前方一致で誤って混ざるため、
    #   CHAIN_DELIMITER で分割した先頭ノードの「完全一致」で判定する。
    node = str(surface["key"]).split(CHAIN_DELIMITER, 1)[0]
    surfaces_of_target_node = [
        s
        for s in surfaces
        if str(s.get("key", "")).split(CHAIN_DELIMITER, 1)[0] == node
    ]
    area_ceiling = sum([s["area"] for s in surfaces_of_target_node if s["part"] == "ceiling"])
    area_wall = sum([s["area"] for s in surfaces_of_target_node if s["part"] == "wall"])
    area_ceiling_wall = area_ceiling + area_wall
    area_floor = sum([s["area"] for s in surfaces_of_target_node if s["part"] == "floor"])

    # ガラス透過日射の配分:
    # - 床/床以外（壁・天井）: eta の代わりに SCR を掛けて表面ノードへ投入
    # - 室空間（空気ノード）   : SCC を掛けて投入（追加ブランチ）
    #
    # 互換: 既存入力が eta のみの場合は SCR のデフォルトとして eta を使用する。
    scr = surface.get("SCR", surface.get("scr", surface.get("eta", 0.9)))
    scc = surface.get("SCC", surface.get("scc", 0.0))

    base = np.array(surface["solar"]) * surface["area"]
    heat_generation_floor        = base * 0.50 * scr
    heat_generation_ceiling_wall = base * 0.50 * scr
    heat_generation_space        = base * scc

    heat_generation_floor        = ensure_timeseries(heat_generation_floor,        sim_length)
    heat_generation_ceiling_wall = ensure_timeseries(heat_generation_ceiling_wall, sim_length)
    heat_generation_space        = ensure_timeseries(heat_generation_space,        sim_length)

    for s in surfaces_of_target_node:
        _, _, i_prefix, _ = get_node_prefix(s)
        branch_key = f"void->{i_prefix}_s"
        # 室内側の各面での「日射吸収」を表すため、受け側表面の eta を掛ける
        # （外壁日射の process_wall_solar と同様の扱い）
        eta_abs = float(s.get("eta", 0.8))
        if s["part"] == "floor":
            if area_floor <= 0:
                continue
            heat_generation = (
                np.array(heat_generation_floor) * eta_abs * s["area"] / area_floor
            ).tolist()
        elif s["part"] == "ceiling" or s["part"] == "wall":
            if area_ceiling_wall <= 0:
                continue
            heat_generation = (
                np.array(heat_generation_ceiling_wall) * eta_abs * s["area"] / area_ceiling_wall
            ).tolist()
        else:
            continue
        logger.info(f"　ガラス透過日射熱ブランチ【{branch_key}】を追加します。")
        thermal_branches.append(
            {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
        )

    # 室空間（ノード）へ SCC 分を追加投入
    # key は一旦 "void->{node}" として生成し、重複があれば validation 側で (01),(02)... にリネームされる。
    if any(v != 0.0 for v in heat_generation_space):
        branch_key = f"void->{node}"
        logger.info(f"　ガラス透過日射（室空間SCC）熱ブランチ【{branch_key}】を追加します。")
        thermal_branches.append(
            {
                "key": branch_key,
                "heat_generation": list(heat_generation_space),
                "subtype": "solar_gain",
                "comment": "glass_solar_space(SCC)",
            }
        )

    return thermal_branches


def process_radiation(node: str, surface_nodes: list[tuple[str, float]]) -> list:
    thermal_branches: list = []
    if len(surface_nodes) < 2:
        return thermal_branches
    sum_area = float(sum(a for _, a in surface_nodes))
    if sum_area <= 0.0:
        return thermal_branches

    for i, node1 in enumerate(surface_nodes):
        for j, node2 in enumerate(surface_nodes[i + 1 :], start=i + 1):
            node1_key, area1 = node1
            node2_key, area2 = node2
            branch_key = f"{node1_key}->{node2_key}"
            conductance = DEFAULT_ALPHA_R * DEFAULT_ETA_LW * area1 * area2 / sum_area
            logger.info(f"　室内放射熱ブランチ【{branch_key}】を追加します。")
            thermal_branches.append(
                {"key": branch_key, "conductance": conductance, "subtype": "radiation"}
            )

    return thermal_branches


def process_surfaces(
    surface_config: list,
    sim_length: int,
    node_config: list | None = None,
    add_solar: bool = True,
    add_nocturnal: bool = True,
    add_radiation: bool = True,
    radiation_exclude_glass: bool = False,
    layer_method: str = "rc",
    time_step: float | None = None,
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> tuple[list, list]:
    """
    builder から呼び出す統合処理。
    - 各面の要素分割ノード/熱ブランチを生成
    - 日射（壁/床/天井・ガラス）の熱ブランチを追加（add_solar が True の場合）
    - 室内放射の熱ブランチを追加（add_radiation が True の場合）
    戻り値は (add_nodes, add_thermal_branches)。
    """
    if not surface_config:
        return [], []

    nodes: list = []
    thermal_branches: list = []

    surface_data = surface_config
    initial_t_by_node_key: dict[str, float] = {}
    if node_config:
        for n in node_config:
            if not isinstance(n, dict):
                continue
            k = n.get("key")
            if not k:
                continue
            if "t" not in n:
                continue
            init_t = _scalar_initial_temperature(n.get("t"))
            if init_t is None:
                continue
            initial_t_by_node_key[str(k)] = init_t

    # 表面の分解
    logger.info("表面の解析を開始します。")
    for s in surface_data:
        # 全体指定（引数）をデフォルトとして surface ごとに持たせる
        if isinstance(s, dict) and "layer_method" not in s:
            s["layer_method"] = layer_method
        add_nodes, add_tb = process_surface(
            s,
            initial_t_by_node_key=initial_t_by_node_key,
            time_step=time_step,
            response_method=response_method,
            response_terms=response_terms,
        )
        nodes.extend(add_nodes)
        thermal_branches.extend(add_tb)
    logger.info("表面の解析が完了しました。")

    # 日射
    if add_solar:
        logger.info("日射の解析を開始します。")
        for s in (x for x in surface_data if "solar" in x):
            if s["part"] in ["wall", "floor", "ceiling"]:
                thermal_branches.extend(process_wall_solar(s, sim_length))
            elif s["part"] == "glass":
                thermal_branches.extend(process_glass_solar(s, surface_data, sim_length))
        logger.info("日射の解析が完了しました。")
    else:
        logger.info("日射の解析をスキップします。")

    # 夜間放射（外部への放射損失）
    if add_nocturnal:
        logger.info("夜間放射の解析を開始します。")
        for s in (x for x in surface_data if ("nocturnal" in x or "night_radiation" in x)):
            if s["part"] in ["wall", "floor", "ceiling", "glass"]:
                thermal_branches.extend(process_wall_nocturnal(s, sim_length))
        logger.info("夜間放射の解析が完了しました。")
    else:
        logger.info("夜間放射の解析をスキップします。")

    # 室内放射
    if add_radiation:
        logger.info("室内放射の解析を開始します。")
        # 基準室（start側）ごとに、その室に接する全表面（start/end の両側）を放射対象として集約する。
        # これにより「X->LD」のような surface でも LD 側表面ノードを LD 室の放射ネットワークへ含められる。
        start_nodes = {str(s.get("key", "")).split(CHAIN_DELIMITER, 1)[0] for s in surface_data}
        for node in start_nodes:
            node_area_map: dict[str, float] = {}
            for s in surface_data:
                if radiation_exclude_glass and str(s.get("part", "")).lower() == "glass":
                    continue
                try:
                    start_node, end_node, i_prefix, o_prefix = get_node_prefix(s)
                except Exception:
                    continue
                try:
                    area = float(s.get("area", 0.0))
                except Exception:
                    continue
                if area <= 0.0:
                    continue
                if str(start_node) == node:
                    key_i = f"{i_prefix}_s"
                    node_area_map[key_i] = node_area_map.get(key_i, 0.0) + area
                if str(end_node) == node:
                    key_o = f"{o_prefix}_s"
                    node_area_map[key_o] = node_area_map.get(key_o, 0.0) + area
            node_surfaces = list(node_area_map.items())
            thermal_branches.extend(process_radiation(node, node_surfaces))
        logger.info("室内放射の解析が完了しました。")
    else:
        logger.info("室内放射の解析をスキップします。")

    return nodes, thermal_branches


