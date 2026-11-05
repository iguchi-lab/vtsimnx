"""計算せずに入力を検査・展開する。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.builder.intent_lint import lint_built_config, lint_raw_config
from app.builder.validate import ConfigFileError, ValidationError
from app.schemas.config import UnknownFieldError, prepare_raw_config
from app.schemas.request import SimulationRequest
from app.services import simulation as sim_svc


def _issue(
    *,
    code: str,
    message: str,
    path: List[Any] | None = None,
    severity: str = "warning",
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "code": str(code),
        "message": str(message),
        "path": list(path or []),
        "severity": severity,
    }
    if hint:
        out["hint"] = hint
    return out


def _path_from_loc(loc: Any) -> List[Any]:
    path = [p for p in list(loc or []) if p not in ("body", "config")]
    return path


def issues_from_unknown_details(details: List[Dict[str, Any]], *, as_error: bool) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for item in details:
        path = item.get("path")
        if not isinstance(path, list):
            path = _path_from_loc(item.get("loc"))
            field = item.get("field")
            if field and field not in path:
                path = list(path) + [field]
        msg = item.get("message") or item.get("msg") or ""
        if as_error:
            msg = str(msg).replace("無視しました。", "拒否します。")
            issues.append(
                _issue(
                    code="unknown_field",
                    message=msg,
                    path=path,
                    severity="error",
                    hint="未定義キーを削除してください。",
                )
            )
        else:
            issues.append(
                _issue(
                    code=str(item.get("code") or "unknown_field_stripped"),
                    message=str(msg),
                    path=path,
                    severity="warning",
                    hint="未知キーは strip 時に無視されます。",
                )
            )
    return issues


def issues_from_unknown_field_error(exc: UnknownFieldError) -> List[Dict[str, Any]]:
    return issues_from_unknown_details(exc.details or [], as_error=True) or [
        _issue(
            code="unknown_field",
            message=str(exc),
            path=[],
            severity="error",
            hint="未定義キーを削除してください。",
        )
    ]


def _path_from_builder_message(message: str) -> List[Any]:
    text = message or ""
    if "heat_source" in text:
        return ["heat_source"]
    if "humidity_source" in text:
        return ["humidity_source"]
    if "換気ブランチ" in text or "ventilation" in text:
        return ["ventilation_branches"]
    if "熱ブランチ" in text:
        return ["thermal_branches"]
    if "ノード" in text:
        return ["nodes"]
    if "simulation" in text.lower():
        return ["simulation"]
    if "aircon" in text or "空調" in text:
        return ["aircon"]
    if "surface" in text or "表面" in text:
        return ["surfaces"]
    return []


def issues_from_exception(exc: Exception) -> List[Dict[str, Any]]:
    raw = str(exc).strip() or type(exc).__name__
    parts = [p.strip() for p in raw.split("\n") if p.strip()] or [raw]
    hint = None
    if "存在しません" in raw:
        hint = "nodes に参照先ノードを追加するか、参照先の key を既存ノード名に合わせてください。"
    code = "invalid_config"
    if isinstance(exc, KeyError):
        code = "invalid_config_missing_field"
    return [
        _issue(
            code=code,
            message=part,
            path=_path_from_builder_message(part),
            severity="error",
            hint=hint,
        )
        for part in parts
    ]


def issues_from_builder_warning_details(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for item in details:
        path = item.get("path")
        if not isinstance(path, list):
            path = []
            section = item.get("section")
            if section:
                path.append(section)
            field = item.get("field")
            if field:
                path.append(field)
        issues.append(
            _issue(
                code=str(item.get("code") or "builder_warning"),
                message=str(item.get("message") or ""),
                path=path,
                severity=str(item.get("severity") or "warning"),
                hint=item.get("hint"),
            )
        )
    return issues


def _builder_kwargs(req: SimulationRequest) -> Dict[str, Any]:
    return {
        "output_path": None,
        "add_surface": req.add_surface,
        "add_aircon": req.add_aircon,
        "add_capacity": req.add_capacity,
        "add_moisture_capacity": req.add_moisture_capacity,
        "add_surface_solar": req.add_surface_solar,
        "add_surface_nocturnal": req.add_surface_nocturnal,
        "add_surface_radiation": req.add_surface_radiation,
        "add_surface_radiation_exclude_glass": req.add_surface_radiation_exclude_glass,
        "unknown_keys": req.unknown_keys,
    }


def _has_error(issues: List[Dict[str, Any]]) -> bool:
    return any(i.get("severity") == "error" for i in issues)


def inspect_config(req: SimulationRequest) -> Dict[str, Any]:
    """スキーマ・未知キー・builder 検証・意図警告。ソルバは呼ばない。"""
    issues: List[Dict[str, Any]] = []
    built: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, int]] = None
    try:
        prepared, _warnings, details = prepare_raw_config(req.config, unknown_keys="strip")
    except UnknownFieldError as exc:
        return {
            "ok": False,
            "issues": issues_from_unknown_field_error(exc),
            "config": None,
            "stats": None,
        }

    issues.extend(issues_from_unknown_details(details, as_error=req.unknown_keys == "error"))
    issues.extend(lint_raw_config(prepared))

    try:
        built, _warnings, warning_details = sim_svc.build_config_with_warning_details(
            prepared, **_builder_kwargs(req)
        )
    except UnknownFieldError as exc:
        issues.extend(issues_from_unknown_field_error(exc))
        return {"ok": False, "issues": issues, "config": None, "stats": None}
    except (ValidationError, ConfigFileError, ValueError, KeyError, TypeError) as exc:
        issues.extend(issues_from_exception(exc))
        return {"ok": False, "issues": issues, "config": None, "stats": None}

    issues.extend(issues_from_builder_warning_details(warning_details))
    issues.extend(lint_built_config(built))
    stats = {
        "nodes": len(built.get("nodes") or []),
        "thermal_branches": len(built.get("thermal_branches") or []),
        "ventilation_branches": len(built.get("ventilation_branches") or []),
    }
    ok = not _has_error(issues)
    return {
        "ok": ok,
        "issues": issues,
        "config": built if ok else None,
        "stats": stats if ok else None,
    }


def build_config_only(req: SimulationRequest) -> Dict[str, Any]:
    """展開後 JSON と生成件数を返す。ソルバは呼ばない。"""
    result = inspect_config(req)
    return {
        "ok": result["ok"],
        "issues": result["issues"],
        "stats": result.get("stats"),
        "config": result.get("config"),
    }


__all__ = [
    "build_config_only",
    "inspect_config",
    "issues_from_exception",
    "issues_from_unknown_field_error",
]
