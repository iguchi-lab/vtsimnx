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
SURFACE_PART_ALIASES = {
    "window": "glass",
}
SOLAR_TARGET_PARTS = frozenset(["wall", "floor", "ceiling"])
NOCTURNAL_TARGET_PARTS = frozenset(["wall", "floor", "ceiling", "glass"])

DEFAULT_ALPHA_I = 4.4   # 室内側表面の対流熱伝達率
DEFAULT_ALPHA_O = 20.3  # 室外側表面の対流熱伝達率
DEFAULT_ALPHA_R = 4.7   # 室内表面間の放射熱伝達率 [W/m2/K]（両面の長波放射率0.9は既に含む。0.9/0.8は掛けない）
DEFAULT_ETA_SW = 0.8   # 短波（日射）の吸収率（外壁日射・ガラス透過日射の床・壁への吸収）
DEFAULT_ETA_LW = 0.9    # 長波の吸収率（夜間放射・発熱の放射配分で表面が吸収するとき。室内表面間の4.7には不要）
DEFAULT_EPSILON_LW = 0.9  # 長波放射率（夜間放射の放出側。室内表面間の4.7には既に含まれる）
# 空気の体積熱容量 ρ·c_p [J/(m³·K)]。archenv の定数（ρ≈1.2 kg/m³, c_p=1005 J/(kg·K)）と同じ。
DEFAULT_AIR_V_CAPA = 1.2 * 1005  # 1206.0


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


def _layer_flag(layer: dict, *names: str) -> bool:
    for n in names:
        v = layer.get(n)
        if isinstance(v, bool):
            return v
    return False


def _layer_float(layer: dict, *names: str, default: float | None = None) -> float | None:
    for n in names:
        if n in layer:
            try:
                return float(layer[n])
            except Exception:
                raise ValueError(f"invalid numeric value for layer.{n}: {layer.get(n)!r}")
    return default


def _branch_log_detail(branch: dict) -> str:
    """熱ブランチのログ用に conductance / subtype / heat_generation を文字列化する（単位付き）。"""
    parts: list[str] = []
    if "conductance" in branch:
        try:
            g = float(branch["conductance"])
            parts.append(f"conductance={g:.6g} [W/K]")
        except (TypeError, ValueError):
            parts.append(f"conductance={branch['conductance']!r}")
    if "subtype" in branch:
        parts.append(f"subtype={branch['subtype']!r}")
    if "heat_generation" in branch:
        hg = branch["heat_generation"]
        if hasattr(hg, "__len__"):
            n = len(hg)
            parts.append(f"heat_generation=timeseries(len={n}) [W]")
        else:
            parts.append("heat_generation=(scalar) [W]")
    if "area" in branch:
        try:
            parts.append(f"area={float(branch['area']):.6g} [m²]")
        except (TypeError, ValueError):
            pass
    return " " + ", ".join(parts) if parts else ""


def _node_log_detail(thermal_mass: float | None, subtype: str | None) -> str:
    """ノードのログ用に thermal_mass / subtype を文字列化する（単位付き）。"""
    parts: list[str] = []
    if thermal_mass is not None:
        parts.append(f"thermal_mass={thermal_mass:.6g} [J/K]")
    if subtype is not None:
        parts.append(f"subtype={subtype!r}")
    return " " + ", ".join(parts) if parts else ""


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
    start_part = _get_surface_part(surface)
    end_part = SURFACE_PAIR[start_part]
    comment        = surface.get("comment", "").strip()
    comment_suffix = f"({comment})" if comment else ""
    i_prefix       = f"{start_node}-{end_node}{comment_suffix}_{start_part}"
    o_prefix       = f"{end_node}-{start_node}{comment_suffix}_{end_part}"
    return start_node, end_node, i_prefix, o_prefix


def _get_surface_part(surface: dict) -> str:
    part_raw = surface.get("part")
    if not isinstance(part_raw, str):
        raise ConfigFileError(f"surface.part must be a non-empty string, got {part_raw!r}")
    part = part_raw.strip().lower()
    if not part:
        raise ConfigFileError(f"surface.part must be a non-empty string, got {part_raw!r}")
    part = SURFACE_PART_ALIASES.get(part, part)
    if part not in SURFACE_PAIR:
        supported = ", ".join(sorted(SURFACE_PAIR.keys()))
        raise ConfigFileError(f"surface.part must be one of [{supported}], got {part_raw!r}")
    return part


def collect_room_side_surfaces(
    room: str,
    surfaces: list[dict],
    *,
    exclude_glass: bool = False,
) -> list[tuple[dict, str, str, float]]:
    """
    基準室 `room` に接する表面（start/end 両側）を収集する。
    戻り値: [(surface_dict, room_side_node_key, room_side_part, area), ...]
    """
    out: list[tuple[dict, str, str, float]] = []
    room_s = str(room)
    for s in surfaces or []:
        if not isinstance(s, dict):
            continue
        try:
            part_start = _get_surface_part(s)
        except Exception:
            continue
        if exclude_glass and part_start == "glass":
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

        part_end = SURFACE_PAIR.get(part_start)

        if str(start_node) == room_s:
            out.append((s, f"{i_prefix}_s", part_start, area))
        # A->A の自己ループ面は二重計上しない
        if str(end_node) == room_s and str(end_node) != str(start_node) and part_end is not None:
            out.append((s, f"{o_prefix}_s", part_end, area))
    return out


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
        for idx, layer in enumerate(surface.get("layers", [])):
            if not isinstance(layer, dict):
                continue
            is_hollow = _layer_flag(layer, "air_layer")
            is_ventilated = _layer_flag(layer, "ventilated_air_layer")
            if is_hollow or is_ventilated:
                raise ValueError(
                    f"surface {surface.get('key','?')}: layer[{idx}] has hollow/ventilated flag, "
                    "which is supported only when layer_method='rc'"
                )
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
        logger.info(f"　応答係数熱ブランチ【{tb['key']}】を追加します。{_branch_log_detail(tb)}")
        thermal_branches.append(tb)

        return nodes, thermal_branches

    if "layers" in surface:
        layers = surface["layers"]
        n = len(layers)
        node_names = (
            [f"{i_prefix}_s"]
            + [f"{i_prefix}_{i+1}-{i+2}" for i in range(n - 1)]
            + [f"{o_prefix}_s"]
        )
        node_types = ["surface"] + ["internal"] * (n - 1) + ["surface"]
        node_thermal_mass: dict[str, float] = {k: 0.0 for k in node_names}
        extra_nodes: list[tuple[str, str, float]] = []

        # 室内側/室外側の対流
        thermal_branches.append(
            {"key": f"{start_node}->{node_names[0]}", "conductance": a * alpha_i, "subtype": "convection"}
        )

        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"surface {surface.get('key','?')}: layers[{idx}] must be dict")
            left = node_names[idx]
            right = node_names[idx + 1]

            is_hollow = _layer_flag(layer, "air_layer")
            is_ventilated = _layer_flag(layer, "ventilated_air_layer")
            if is_hollow and is_ventilated:
                raise ValueError(
                    f"surface {surface.get('key','?')}: layers[{idx}] cannot have both "
                    "air_layer and ventilated_air_layer"
                )

            if is_ventilated:
                alpha_c1 = _layer_float(layer, "alpha_c1", default=DEFAULT_ALPHA_I)
                alpha_c2 = _layer_float(layer, "alpha_c2", default=DEFAULT_ALPHA_I)
                alpha_r = _layer_float(layer, "alpha_r", default=DEFAULT_ALPHA_R)
                thickness = _layer_float(layer, "t")
                if thickness is None or thickness <= 0.0:
                    raise ValueError(
                        f"surface {surface.get('key','?')}: ventilated layer[{idx}] requires positive 't'"
                    )
                air_v_capa = _layer_float(
                    layer, "air_v_capa", "v_capa_air", "v_capa", default=DEFAULT_AIR_V_CAPA
                )
                if air_v_capa is None or air_v_capa < 0.0:
                    raise ValueError(
                        f"surface {surface.get('key','?')}: ventilated layer[{idx}] has invalid air heat capacity"
                    )
                # 通気層: 空気の熱容量は全て中心のみ。左境界＝左側建材の半分（隣接層で付与）、右境界＝右側建材の半分（後段で付与）。
                capa_vent = a * thickness * air_v_capa
                center = f"{i_prefix}_{idx+1}_vent"
                extra_nodes.append((center, "internal", capa_vent))
                thermal_branches.append(
                    {"key": f"{left}->{center}", "conductance": a * alpha_c1, "subtype": "convection"}
                )
                thermal_branches.append(
                    {"key": f"{center}->{right}", "conductance": a * alpha_c2, "subtype": "convection"}
                )
                thermal_branches.append(
                    {"key": f"{left}->{right}", "conductance": a * alpha_r, "subtype": "radiation"}
                )
                continue

            if is_hollow:
                # 中空層: thermal_resistance（または r_value / r）で抵抗を指定。中心ノードは設けず、設定された抵抗値で 1 本の伝導のみ。厚さ t は使用しない。
                r_layer = _layer_float(layer, "thermal_resistance", "r_value", "r")
                if r_layer is None:
                    raise ValueError(
                        f"surface {surface.get('key','?')}: hollow layer[{idx}] requires "
                        "'thermal_resistance' (or 'r_value'/'r')"
                    )
                if r_layer <= 0.0:
                    raise ValueError(
                        f"surface {surface.get('key','?')}: hollow layer[{idx}] resistance must be positive"
                    )
                thermal_branches.append(
                    {"key": f"{left}->{right}", "conductance": a / r_layer, "subtype": "conduction"}
                )
                continue

            lam = _layer_float(layer, "lambda")
            thickness = _layer_float(layer, "t")
            v_capa = _layer_float(layer, "v_capa")
            if lam is None or thickness is None or v_capa is None:
                raise ValueError(
                    f"surface {surface.get('key','?')}: normal layer[{idx}] requires lambda, t, v_capa"
                )
            if lam <= 0.0 or thickness <= 0.0 or v_capa < 0.0:
                raise ValueError(
                    f"surface {surface.get('key','?')}: normal layer[{idx}] must satisfy "
                    "lambda>0, t>0, v_capa>=0"
                )
            # 熱容量 [J/K] = 面積 [m²] × 体積熱容量 [J/(m³·K)] × 厚さ [m]
            # 注意: t は [m]、v_capa は [J/(m³·K)]。t を [mm] で渡すと熱容量が約1000倍になる。
            # core materials は公開時 v_capa を既に [J/(m³·K)] にしているのでそのまま使う。
            c_layer = a * v_capa * thickness
            node_thermal_mass[left] += c_layer / 2.0
            node_thermal_mass[right] += c_layer / 2.0
            thermal_branches.append(
                {"key": f"{left}->{right}", "conductance": a * lam / thickness, "subtype": "conduction"}
            )

        thermal_branches.append(
            {"key": f"{node_names[-1]}->{end_node}", "conductance": a * alpha_o, "subtype": "convection"}
        )

        base_nodes = [(name, node_types[i], node_thermal_mass[name]) for i, name in enumerate(node_names)]
        for node, subtype, thermal_mass in base_nodes + extra_nodes:
            logger.info(f"　ノード【{node}】 を追加します。{_node_log_detail(thermal_mass, subtype)}")
            node_dict = {
                "key": node,
                "calc_t": True,
                "thermal_mass": thermal_mass,
                "type": "layer",
                "subtype": subtype,
            }
            # 生成ノードの初期温度: ノード key の先頭に現れるノード（例: A-B... なら A）の初期温度をコピー
            if initial_t_by_node_key:
                lead = _leading_node_key_from_layer_key(node)
                if lead in initial_t_by_node_key:
                    node_dict["t"] = initial_t_by_node_key[lead]
            nodes.append(node_dict)

        for branch in thermal_branches:
            logger.info(f"　熱ブランチ【{branch['key']}】を追加します。{_branch_log_detail(branch)}")
    else:
        node_names = [f"{i_prefix}_s", f"{o_prefix}_s"]
        node_types = ["surface", "surface"]
        c = [a * surface.get("a_capacity", 0.0)]
        thermal_mass = [c[0] / 2, c[0] / 2]
        conductance = [a * alpha_i, a * surface["u_value"], a * alpha_o]
        branch_types = ["convection", "conduction", "convection"]
        thermal_node_chain = [start_node] + node_names + [end_node]
        thermal_branch_names = [
            f"{thermal_node_chain[i]}->{thermal_node_chain[i+1]}" for i in range(3)
        ]

        for i, node in enumerate(node_names):
            logger.info(f"　ノード【{node}】 を追加します。{_node_log_detail(thermal_mass[i], node_types[i])}")
            node_dict = {
                "key": node,
                "calc_t": True,
                "thermal_mass": thermal_mass[i],
                "type": "layer",
                "subtype": node_types[i],
            }
            if initial_t_by_node_key:
                lead = _leading_node_key_from_layer_key(node)
                if lead in initial_t_by_node_key:
                    node_dict["t"] = initial_t_by_node_key[lead]
            nodes.append(node_dict)

        for i, branch in enumerate(thermal_branch_names):
            b = {"key": branch, "conductance": conductance[i], "subtype": branch_types[i]}
            logger.info(f"　熱ブランチ【{branch}】を追加します。{_branch_log_detail(b)}")
            thermal_branches.append(b)

    return nodes, thermal_branches


def process_wall_solar(surface: dict, sim_length: int) -> list:
    thermal_branches: list = []
    _, _, _, o_prefix = get_node_prefix(surface)

    heat_generation = surface["area"] * surface.get("eta", DEFAULT_ETA_SW) * np.array(surface["solar"])
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    b = {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
    logger.info(f"　外壁日射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
    thermal_branches.append(b)
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

    # 表面->void への流出なので負符号。夜間放射は長波なので epsilon=0.9
    heat_generation = -surface["area"] * surface.get("epsilon", DEFAULT_EPSILON_LW) * np.array(noct)
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    b = {"key": branch_key, "heat_generation": heat_generation, "subtype": "nocturnal_loss"}
    logger.info(f"　外壁夜間放射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
    thermal_branches.append(b)
    return thermal_branches


def process_glass_solar(surface: dict, surfaces: list, sim_length: int) -> list:
    thermal_branches: list = []

    # NOTE:
    # - 同一室（start_node が同じ）の表面に対して配分する。
    # - startswith だと "Room1" と "Room10" のような前方一致で誤って混ざるため、
    #   CHAIN_DELIMITER で分割した先頭ノードの「完全一致」で判定する。
    node = str(surface["key"]).split(CHAIN_DELIMITER, 1)[0]
    # 基準室 node に接する表面（start/end 両側）を収集する。
    # これにより "X->LD" のような面でも LD 側表面ノードを配分対象に含められる。
    room_side_surfaces = collect_room_side_surfaces(node, surfaces)

    area_ceiling = sum([area for _s, _node_key, part, area in room_side_surfaces if part == "ceiling"])
    area_wall = sum([area for _s, _node_key, part, area in room_side_surfaces if part == "wall"])
    area_ceiling_wall = area_ceiling + area_wall
    area_floor = sum([area for _s, _node_key, part, area in room_side_surfaces if part == "floor"])

    # ガラス透過日射の配分:
    # - 床/床以外（壁・天井）: eta の代わりに SCR を掛けて表面ノードへ投入
    # - 室空間（空気ノード）   : SCC を掛けて投入（追加ブランチ）
    #
    # 互換: 既存入力が eta のみの場合は SCR のデフォルトとして eta を使用する。日射は短波なので 0.8。
    scr = surface.get("SCR", surface.get("scr", surface.get("eta", DEFAULT_ETA_SW)))
    scc = surface.get("SCC", surface.get("scc", 0.0))

    base = np.array(surface["solar"]) * surface["area"]
    heat_generation_floor        = base * 0.50 * scr
    heat_generation_ceiling_wall = base * 0.50 * scr
    heat_generation_space        = base * scc

    heat_generation_floor        = ensure_timeseries(heat_generation_floor,        sim_length)
    heat_generation_ceiling_wall = ensure_timeseries(heat_generation_ceiling_wall, sim_length)
    heat_generation_space        = ensure_timeseries(heat_generation_space,        sim_length)

    for s, room_node_key, part, area in room_side_surfaces:
        branch_key = f"void->{room_node_key}"
        # 室内側の各面での「日射吸収」を表すため、受け側表面の eta を掛ける（短波なので 0.8）
        eta_abs = float(s.get("eta", DEFAULT_ETA_SW))
        if part == "floor":
            if area_floor <= 0:
                continue
            heat_generation = (
                np.array(heat_generation_floor) * eta_abs * area / area_floor
            ).tolist()
        elif part == "ceiling" or part == "wall":
            if area_ceiling_wall <= 0:
                continue
            heat_generation = (
                np.array(heat_generation_ceiling_wall) * eta_abs * area / area_ceiling_wall
            ).tolist()
        else:
            continue
        b = {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
        logger.info(f"　ガラス透過日射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
        thermal_branches.append(b)

    # 室空間（ノード）へ SCC 分を追加投入
    # key は一旦 "void->{node}" として生成し、重複があれば validation 側で (01),(02)... にリネームされる。
    if any(v != 0.0 for v in heat_generation_space):
        branch_key = f"void->{node}"
        b = {
            "key": branch_key,
            "heat_generation": list(heat_generation_space),
            "subtype": "solar_gain",
            "comment": "glass_solar_space(SCC)",
        }
        logger.info(f"　ガラス透過日射（室空間SCC）熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
        thermal_branches.append(b)

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
            # 4.7 には既に両面の放射率0.9が含まれるため、室内表面間では eta を掛けない
            conductance = DEFAULT_ALPHA_R * area1 * area2 / sum_area
            b = {"key": branch_key, "conductance": conductance, "subtype": "radiation"}
            logger.info(f"　室内放射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
            thermal_branches.append(b)

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
            part = _get_surface_part(s)
            if part in SOLAR_TARGET_PARTS:
                thermal_branches.extend(process_wall_solar(s, sim_length))
            elif part == "glass":
                thermal_branches.extend(process_glass_solar(s, surface_data, sim_length))
        logger.info("日射の解析が完了しました。")
    else:
        logger.info("日射の解析をスキップします。")

    # 夜間放射（外部への放射損失）
    if add_nocturnal:
        logger.info("夜間放射の解析を開始します。")
        for s in (x for x in surface_data if ("nocturnal" in x or "night_radiation" in x)):
            part = _get_surface_part(s)
            if part in NOCTURNAL_TARGET_PARTS:
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
            for _s, node_key, _part, area in collect_room_side_surfaces(
                node, surface_data, exclude_glass=radiation_exclude_glass
            ):
                node_area_map[node_key] = node_area_map.get(node_key, 0.0) + area
            node_surfaces = list(node_area_map.items())
            thermal_branches.extend(process_radiation(node, node_surfaces))
        logger.info("室内放射の解析が完了しました。")
    else:
        logger.info("室内放射の解析をスキップします。")

    return nodes, thermal_branches


