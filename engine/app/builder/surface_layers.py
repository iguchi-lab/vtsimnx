from __future__ import annotations

import numpy as np

from .logger import get_logger
from .surface_constants import (
    DEFAULT_AIR_V_CAPA,
    DEFAULT_ALPHA_I,
    DEFAULT_ALPHA_O,
    DEFAULT_ALPHA_R,
    SURFACE_PAIR,
    SURFACE_PART_ALIASES,
)
from .utils import CHAIN_DELIMITER
from .validate import ConfigFileError

logger = get_logger(__name__)


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


def _get_air_v_capa(layer: dict) -> float:
    """層の空気体積熱容量を取得。未指定・不正時は DEFAULT_AIR_V_CAPA を返す。"""
    v = _layer_float(layer, "air_v_capa", "v_capa_air", "v_capa", default=DEFAULT_AIR_V_CAPA)
    if v is None or v < 0.0:
        return DEFAULT_AIR_V_CAPA
    return float(v)


def _build_layer_node_dict(
    key: str,
    subtype: str,
    thermal_mass: float,
    initial_t_by_node_key: dict[str, float] | None,
) -> dict:
    """層ノード用の辞書を組み立てる（key, calc_t, type, subtype, thermal_mass, 必要なら t）。"""
    node_dict = {
        "key": key,
        "calc_t": True,
        "type": "layer",
        "subtype": subtype,
        "thermal_mass": thermal_mass,
    }
    if initial_t_by_node_key:
        lead = _leading_node_key_from_layer_key(key)
        if lead in initial_t_by_node_key:
            node_dict["t"] = initial_t_by_node_key[lead]
    return node_dict


def _apply_ventilated_layer(
    layer: dict, idx: int, left: str, right: str, a: float, i_prefix: str, surface_key: str
) -> tuple[list[tuple[str, str, float]], list[dict]]:
    """
    通気層を処理する。返り値は (extra_nodes: [(name, subtype, thermal_mass)], branches)。
    """
    alpha_c1 = _layer_float(layer, "alpha_c1", default=DEFAULT_ALPHA_I)
    alpha_c2 = _layer_float(layer, "alpha_c2", default=DEFAULT_ALPHA_I)
    alpha_r = _layer_float(layer, "alpha_r", default=DEFAULT_ALPHA_R)
    thickness = _layer_float(layer, "t")
    if thickness is None or thickness <= 0.0:
        raise ValueError(
            f"surface {surface_key}: ventilated layer[{idx}] requires positive 't'"
        )
    air_v_capa = _get_air_v_capa(layer)
    if air_v_capa < 0.0:
        raise ValueError(
            f"surface {surface_key}: ventilated layer[{idx}] has invalid air heat capacity"
        )
    capa_vent = a * thickness * air_v_capa
    center = f"{i_prefix}_{idx+1}_vent"
    extra_nodes = [(center, "internal", capa_vent)]
    branches = [
        {"key": f"{left}->{center}", "conductance": a * alpha_c1, "subtype": "convection"},
        {"key": f"{center}->{right}", "conductance": a * alpha_c2, "subtype": "convection"},
        {"key": f"{left}->{right}", "conductance": a * alpha_r, "subtype": "radiation"},
    ]
    return extra_nodes, branches


def _apply_hollow_layer(
    layer: dict, idx: int, left: str, right: str, a: float, i_prefix: str, surface_key: str
) -> tuple[float, float, list[dict]]:
    """
    中空層を処理する。返り値は (add_thermal_mass_left, add_thermal_mass_right, branches)。
    """
    r_layer = _layer_float(layer, "thermal_resistance", "r_value", "r")
    if r_layer is None:
        raise ValueError(
            f"surface {surface_key}: hollow layer[{idx}] requires "
            "'thermal_resistance' (or 'r_value'/'r')"
        )
    if r_layer <= 0.0:
        raise ValueError(
            f"surface {surface_key}: hollow layer[{idx}] resistance must be positive"
        )
    thickness = _layer_float(layer, "t")
    if thickness is None or thickness <= 0.0:
        raise ValueError(
            f"surface {surface_key}: hollow layer[{idx}] requires positive 't'"
        )
    air_v_capa = _get_air_v_capa(layer)
    capa_air = a * thickness * air_v_capa
    add_left = capa_air / 2.0
    add_right = capa_air / 2.0
    branches = [
        {"key": f"{left}->{right}", "conductance": a / r_layer, "subtype": "conduction"}
    ]
    return add_left, add_right, branches


def _apply_normal_layer(
    layer: dict, idx: int, left: str, right: str, a: float, surface_key: str
) -> tuple[float, float, list[dict]]:
    """
    通常層（lambda, t, v_capa）を処理する。返り値は (add_thermal_mass_left, add_thermal_mass_right, branches)。
    """
    lam = _layer_float(layer, "lambda")
    thickness = _layer_float(layer, "t")
    v_capa = _layer_float(layer, "v_capa")
    if lam is None or thickness is None or v_capa is None:
        raise ValueError(
            f"surface {surface_key}: normal layer[{idx}] requires lambda, t, v_capa"
        )
    if lam <= 0.0 or thickness <= 0.0 or v_capa < 0.0:
        raise ValueError(
            f"surface {surface_key}: normal layer[{idx}] must satisfy "
            "lambda>0, t>0, v_capa>=0"
        )
    c_layer = a * v_capa * thickness
    add_left = c_layer / 2.0
    add_right = c_layer / 2.0
    branches = [
        {
            "key": f"{left}->{right}",
            "conductance": a * lam / thickness,
            "subtype": "conduction",
        }
    ]
    return add_left, add_right, branches


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


def process_surface(
    surface: dict,
    initial_t_by_node_key: dict[str, float] | None = None,
    *,
    time_step: float | None = None,
    response_method: str = "arx_rc",
    response_terms: int | None = None,
    verbose: bool = True,
) -> tuple[list, list]:
    nodes: list = []
    thermal_branches: list = []

    start_node, end_node, i_prefix, o_prefix = get_node_prefix(surface)
    a = surface["area"]
    alpha_i = surface.get("alpha_i", DEFAULT_ALPHA_I)
    alpha_o = surface.get("alpha_o", DEFAULT_ALPHA_O)
    layer_method = surface.get("layer_method", "rc")

    if "layers" in surface and layer_method == "response":
        from .surface_response import _process_surface_response_method

        return _process_surface_response_method(
            surface, start_node, end_node, i_prefix, o_prefix, a,
            initial_t_by_node_key, time_step, response_method, response_terms, verbose,
        )

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

        surface_key = surface.get("key", "?")
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"surface {surface_key}: layers[{idx}] must be dict")
            left = node_names[idx]
            right = node_names[idx + 1]

            is_hollow = _layer_flag(layer, "air_layer")
            is_ventilated = _layer_flag(layer, "ventilated_air_layer")
            if is_hollow and is_ventilated:
                raise ValueError(
                    f"surface {surface_key}: layers[{idx}] cannot have both "
                    "air_layer and ventilated_air_layer"
                )

            if is_ventilated:
                extra, br = _apply_ventilated_layer(
                    layer, idx, left, right, a, i_prefix, surface_key
                )
                extra_nodes.extend(extra)
                thermal_branches.extend(br)
                continue

            if is_hollow:
                add_left, add_right, br = _apply_hollow_layer(
                    layer, idx, left, right, a, i_prefix, surface_key
                )
                node_thermal_mass[left] += add_left
                node_thermal_mass[right] += add_right
                thermal_branches.extend(br)
                continue

            add_left, add_right, br = _apply_normal_layer(
                layer, idx, left, right, a, surface_key
            )
            node_thermal_mass[left] += add_left
            node_thermal_mass[right] += add_right
            thermal_branches.extend(br)

        thermal_branches.append(
            {"key": f"{node_names[-1]}->{end_node}", "conductance": a * alpha_o, "subtype": "convection"}
        )

        base_nodes = [(name, node_types[i], node_thermal_mass[name]) for i, name in enumerate(node_names)]
        for node, subtype, thermal_mass in base_nodes + extra_nodes:
            if verbose:
                logger.info(f"　ノード【{node}】 を追加します。{_node_log_detail(thermal_mass, subtype)}")
            nodes.append(
                _build_layer_node_dict(node, subtype, thermal_mass, initial_t_by_node_key)
            )

        for branch in thermal_branches:
            if verbose:
                logger.info(f"　熱ブランチ【{branch['key']}】を追加します。{_branch_log_detail(branch)}")
    else:
        return _process_surface_u_value(
            surface, start_node, end_node, i_prefix, o_prefix, a, alpha_i, alpha_o,
            initial_t_by_node_key, verbose,
        )

    return nodes, thermal_branches


def _process_surface_u_value(
    surface: dict,
    start_node: str,
    end_node: str,
    i_prefix: str,
    o_prefix: str,
    a: float,
    alpha_i: float,
    alpha_o: float,
    initial_t_by_node_key: dict[str, float] | None,
    verbose: bool = True,
) -> tuple[list, list]:
    """u_value のみ指定された表面: 2 ノードと 3 ブランチ（対流・伝導・対流）。"""
    node_names = [f"{i_prefix}_s", f"{o_prefix}_s"]
    node_types = ["surface", "surface"]
    c = a * surface.get("a_capacity", 0.0)
    thermal_mass = [c / 2, c / 2]
    conductance = [a * alpha_i, a * surface["u_value"], a * alpha_o]
    branch_types = ["convection", "conduction", "convection"]
    thermal_node_chain = [start_node] + node_names + [end_node]
    thermal_branch_names = [
        f"{thermal_node_chain[i]}->{thermal_node_chain[i+1]}" for i in range(3)
    ]

    nodes: list = []
    for i, node in enumerate(node_names):
        if verbose:
            logger.info(f"　ノード【{node}】 を追加します。{_node_log_detail(thermal_mass[i], node_types[i])}")
        nodes.append(
            _build_layer_node_dict(node, node_types[i], thermal_mass[i], initial_t_by_node_key)
        )

    thermal_branches: list = []
    for i, branch_key in enumerate(thermal_branch_names):
        b = {"key": branch_key, "conductance": conductance[i], "subtype": branch_types[i]}
        if verbose:
            logger.info(f"　熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
        thermal_branches.append(b)

    return nodes, thermal_branches
