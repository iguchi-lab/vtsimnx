"""builder オプションの解決（関数引数 / builder JSON / トップレベル / 既定値）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


_BOOL_FLAGS_DEFAULT_TRUE = (
    "add_surface",
    "add_aircon",
    "add_capacity",
    "add_moisture_capacity",
    "add_surface_solar",
    "add_surface_nocturnal",
    "add_surface_radiation",
)

_BOOL_FLAGS_DEFAULT_FALSE = (
    "add_surface_radiation_exclude_glass",
)


def _pick_bool(obj: dict[str, Any], name: str) -> bool | None:
    v = obj.get(name)
    return v if isinstance(v, bool) else None


@dataclass(frozen=True)
class BuildOptions:
    """展開処理の有無と表面モデル設定（解決済みの最終値）。"""

    add_surface: bool = True
    add_aircon: bool = True
    add_capacity: bool = True
    add_moisture_capacity: bool = True
    add_surface_solar: bool = True
    add_surface_nocturnal: bool = True
    add_surface_radiation: bool = True
    add_surface_radiation_exclude_glass: bool = False
    surface_layer_method: str = "rc"
    response_method: str = "arx_rc"
    response_terms: int | None = None

    @classmethod
    def resolve(
        cls,
        raw: Dict[str, Any],
        *,
        add_surface: bool | None = None,
        add_aircon: bool | None = None,
        add_capacity: bool | None = None,
        add_moisture_capacity: bool | None = None,
        add_surface_solar: bool | None = None,
        add_surface_nocturnal: bool | None = None,
        add_surface_radiation: bool | None = None,
        add_surface_radiation_exclude_glass: bool | None = None,
        surface_layer_method: str = "rc",
        response_method: str = "arx_rc",
        response_terms: int | None = None,
    ) -> "BuildOptions":
        """
        優先順位:
          1. 関数引数（None 以外）
          2. raw["builder"] 内の同名キー
          3. raw トップレベルの同名キー
          4. 既定値
        """
        pending: dict[str, bool | None] = {
            "add_surface": add_surface,
            "add_aircon": add_aircon,
            "add_capacity": add_capacity,
            "add_moisture_capacity": add_moisture_capacity,
            "add_surface_solar": add_surface_solar,
            "add_surface_nocturnal": add_surface_nocturnal,
            "add_surface_radiation": add_surface_radiation,
            "add_surface_radiation_exclude_glass": add_surface_radiation_exclude_glass,
        }

        builder_opt = raw.get("builder")
        if isinstance(builder_opt, dict):
            for name in pending:
                if pending[name] is None:
                    pending[name] = _pick_bool(builder_opt, name)

        for name in pending:
            if pending[name] is None and isinstance(raw.get(name), bool):
                pending[name] = raw.get(name)  # type: ignore[assignment]

        resolved_bools: dict[str, bool] = {}
        for name in _BOOL_FLAGS_DEFAULT_TRUE:
            v = pending[name]
            resolved_bools[name] = True if v is None else bool(v)
        for name in _BOOL_FLAGS_DEFAULT_FALSE:
            v = pending[name]
            resolved_bools[name] = False if v is None else bool(v)

        # surface_layer_method / response_* は「関数引数 surface_layer_method が既定 'rc' のときだけ」JSON を反映
        # （従来 _resolve_builder_options と同じ条件）
        if surface_layer_method == "rc":
            if isinstance(builder_opt, dict):
                v = builder_opt.get("surface_layer_method")
                if isinstance(v, str) and v:
                    surface_layer_method = v
                rm = builder_opt.get("response_method")
                if response_method == "arx_rc" and isinstance(rm, str) and rm:
                    response_method = rm
                rt = builder_opt.get("response_terms")
                if response_terms is None and rt is not None:
                    try:
                        response_terms = int(rt)
                    except Exception as e:
                        raise ValueError(f"builder.response_terms must be int, got {rt!r}") from e

            v2 = raw.get("surface_layer_method")
            if isinstance(v2, str) and v2:
                surface_layer_method = v2
            if response_method == "arx_rc":
                rm2 = raw.get("response_method")
                if isinstance(rm2, str) and rm2:
                    response_method = rm2
            if response_terms is None:
                rt2 = raw.get("response_terms")
                if rt2 is not None:
                    try:
                        response_terms = int(rt2)
                    except Exception as e:
                        raise ValueError(f"response_terms must be int, got {rt2!r}") from e

        return cls(
            **resolved_bools,
            surface_layer_method=surface_layer_method,
            response_method=response_method,
            response_terms=response_terms,
        )


__all__ = ["BuildOptions"]
