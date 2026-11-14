"""Air-conditioning input contract shared by API and builder."""
import math
from typing import Any


def validate_aircon_spec(node: dict[str, Any]) -> None:
    spec = node.get("ac_spec") or {}
    control = node.get("control_mode", "temperature_priority")
    default = "rh95" if control == "temperature_priority" else "ideal_coil"
    method = spec.get("latent_method", default)
    if not isinstance(method, str) or method.lower() not in {
        "rh95", "rh100", "none", "bf", "coil_aoaf", "aoaf", "literature", "ideal_coil",
    }:
        raise ValueError(f"ac_spec.latent_method is invalid: {method!r}")
    method = method.lower()
    if control != "temperature_priority" and method not in {"none", "rh100", "ideal_coil"}:
        raise ValueError("humidistat latent_method must be ideal_coil (none/rh100 aliases); bf/rh95/coil_aoaf are unsupported")
    if control == "temperature_priority" and method == "ideal_coil":
        raise ValueError("latent_method=ideal_coil requires a room humidity control mode; use rh100 for temperature priority")
    for key in ("bf", "BF", "bypass_factor", "Af", "Ao", "coil_face_area", "coil_surface_area"):
        if key not in spec:
            continue
        value = spec[key]
        finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        valid = finite and (0 <= value <= 0.99 if key in {"bf", "BF", "bypass_factor"} else value > 0)
        if not valid:
            raise ValueError(f"ac_spec.{key} is outside its physical range")
    if str(node.get("model", "")).upper() != "DUCT_CENTRAL":
        return
    modes = node.get("mode", "AUTO")
    if not isinstance(modes, list):
        modes = [modes]
    names = {str(m).upper() for m in modes}
    active = set()
    if names & {"HEATING", "1", "AUTO", "3"}:
        active.add("heating")
    if names & {"COOLING", "2", "AUTO", "3"}:
        active.add("cooling")
    for mode in active:
        for group, keys in {"Q": ("min", "mid", "rtd"), "P": ("rtd",),
                            "P_fan": ("rtd",), "V_inner": ("mid", "rtd", "dsgn")}.items():
            for key in keys:
                value = spec.get(group, {}).get(mode, {}).get(key)
                nonnegative = group == "P_fan" or (group == "Q" and key == "min")
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or (value < 0 if nonnegative else value <= 0)):
                    raise ValueError(f"DUCT_CENTRAL requires valid ac_spec.{group}.{mode}.{key}")
        q = spec["Q"][mode]
        if not q["min"] < q["mid"] < q["rtd"]:
            raise ValueError(f"ac_spec.Q.{mode} requires min < mid < rtd")
        if spec["P"][mode]["rtd"] <= spec["P_fan"][mode]["rtd"]:
            raise ValueError(f"ac_spec.P.{mode}.rtd must exceed P_fan.{mode}.rtd")
    if names & {"OFF", "0"} and node.get("vol_zero", 0) > 0:
        mode = "cooling" if "cooling" in spec.get("V_inner", {}) else "heating"
        for group, key in (("P_fan", "rtd"), ("V_inner", "dsgn")):
            value = spec.get(group, {}).get(mode, {}).get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or (value < 0 if group == "P_fan" else value <= 0)):
                raise ValueError(f"OFF ventilation requires ac_spec.{group}.{mode}.{key}")
