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

_DEFAULT_SURFACE_LAYER_METHOD = "rc"
_DEFAULT_RESPONSE_METHOD = "arx_rc"


def _pick_bool(obj: dict[str, Any], name: str) -> bool | None:
    v = obj.get(name)
    return v if isinstance(v, bool) else None


def _pick_str(obj: dict[str, Any], name: str) -> str | None:
    v = obj.get(name)
    return v if isinstance(v, str) and v else None


def _pick_optional_int(obj: dict[str, Any], name: str) -> int | None:
    v = obj.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"{name} must be int, got {v!r}") from e


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
    surface_layer_method: str = _DEFAULT_SURFACE_LAYER_METHOD
    response_method: str = _DEFAULT_RESPONSE_METHOD
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
        surface_layer_method: str | None = None,
        response_method: str | None = None,
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

        builder_opt = raw.get("builder") if isinstance(raw.get("builder"), dict) else None

        if builder_opt is not None:
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

        resolved_layer = surface_layer_method
        resolved_rm = response_method
        resolved_rt = response_terms

        if resolved_layer is None and builder_opt is not None:
            resolved_layer = _pick_str(builder_opt, "surface_layer_method")
        if resolved_layer is None:
            resolved_layer = _pick_str(raw, "surface_layer_method")
        if resolved_layer is None:
            resolved_layer = _DEFAULT_SURFACE_LAYER_METHOD

        if resolved_rm is None and builder_opt is not None:
            resolved_rm = _pick_str(builder_opt, "response_method")
        if resolved_rm is None:
            resolved_rm = _pick_str(raw, "response_method")
        if resolved_rm is None:
            resolved_rm = _DEFAULT_RESPONSE_METHOD

        if resolved_rt is None and builder_opt is not None:
            try:
                resolved_rt = _pick_optional_int(builder_opt, "response_terms")
            except ValueError as e:
                raise ValueError(f"builder.response_terms must be int, got {builder_opt.get('response_terms')!r}") from e
        if resolved_rt is None and "response_terms" in raw:
            resolved_rt = _pick_optional_int(raw, "response_terms")

        return cls(
            **resolved_bools,
            surface_layer_method=resolved_layer,
            response_method=resolved_rm,
            response_terms=resolved_rt,
        )


__all__ = ["BuildOptions"]
