"""統一 API エラー形式。

レスポンス例::

    {
      "error": {
        "code": "invalid_config",
        "message": "...",
        "path": ["nodes", 0, "key"],
        "hint": "...",
        "run_id": "..."
      }
    }
"""
from __future__ import annotations

from typing import Any, Mapping, NoReturn, Optional, Sequence

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response


def error_body(
    *,
    code: str,
    message: str,
    hint: str | None = None,
    path: Sequence[Any] | None = None,
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {
        "code": str(code),
        "message": str(message),
    }
    if hint:
        err["hint"] = str(hint)
    if path is not None:
        err["path"] = list(path)
    if run_id:
        err["run_id"] = str(run_id)
    if extra:
        for k, v in extra.items():
            if k in err or v is None:
                continue
            err[k] = v
    return {"error": err}


def error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    hint: str | None = None,
    path: Sequence[Any] | None = None,
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=int(status_code),
        content=error_body(
            code=code,
            message=message,
            hint=hint,
            path=path,
            run_id=run_id,
            extra=extra,
        ),
    )


def raise_api_error(
    status_code: int,
    *,
    code: str,
    message: str,
    hint: str | None = None,
    path: Sequence[Any] | None = None,
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> NoReturn:
    """HTTPException with structured detail (normalized by exception handlers)."""
    detail = error_body(
        code=code,
        message=message,
        hint=hint,
        path=path,
        run_id=run_id,
        extra=extra,
    )["error"]
    raise HTTPException(status_code=status_code, detail=detail)


def normalize_error_payload(detail: Any) -> dict[str, Any]:
    """Convert various detail shapes into a single error object."""
    if isinstance(detail, dict):
        if "error" in detail and isinstance(detail["error"], dict):
            return detail["error"]
        if "code" in detail or "message" in detail:
            out = dict(detail)
            out.setdefault("code", "error")
            out.setdefault("message", str(detail.get("message") or detail.get("code") or "error"))
            return out
        return {
            "code": "error",
            "message": str(detail),
        }
    if isinstance(detail, list):
        # Pydantic / unknown_field list
        first = detail[0] if detail else {}
        path: list[Any] = []
        message = "validation failed"
        code = "validation_error"
        if isinstance(first, dict):
            loc = first.get("loc")
            if isinstance(loc, (list, tuple)):
                path = list(loc)
            msg = first.get("msg")
            if isinstance(msg, str) and msg:
                message = msg
            typ = first.get("type")
            if typ == "unknown_field":
                code = "unknown_field"
            elif isinstance(typ, str) and typ:
                code = "validation_error"
        return {
            "code": code,
            "message": message,
            "path": path,
            "hint": "リクエスト JSON の型・必須フィールド・未知キーを確認してください。",
            "details": detail,
        }
    return {
        "code": "error",
        "message": str(detail) if detail is not None else "error",
    }


def build_bad_request_detail(e: Exception) -> dict[str, Any]:
    """入力不正(400)の error オブジェクトを返す。"""
    message = str(e)
    code = "invalid_config"
    hint: str | None = None
    path: list[Any] | None = None

    if isinstance(e, KeyError):
        missing = str(e.args[0]) if getattr(e, "args", None) else message
        code = "invalid_config_missing_field"
        message = f"必須フィールド '{missing}' が不足しています。"
        hint = f"入力JSONに '{missing}' を追加してください。"
        path = [missing]

    if "ノード" in message and "存在しません" in message:
        hint = "nodes に参照先ノードを追加するか、参照先の key を既存ノード名に合わせてください。"
    if "ventilated layer" in message and "requires positive 't'" in message:
        hint = "ventilated_air_layer=true の層には正の厚さ t（例: 0.04）を指定してください。"

    out: dict[str, Any] = {"code": code, "message": message}
    if hint:
        out["hint"] = hint
    if path is not None:
        out["path"] = path
    return out


def build_internal_error_detail(e: Exception, *, run_id: Optional[str] = None) -> dict[str, Any]:
    """内部エラー(500)の error オブジェクトを返す。"""
    detail: dict[str, Any] = {
        "code": "internal_error",
        "message": str(e),
    }
    if run_id:
        detail["run_id"] = run_id

    if isinstance(e, FileNotFoundError) and "vtsimnx_solver" in str(e):
        detail["code"] = "solver_binary_not_found"
        detail["hint"] = "サーバ上で C++ solver 実行ファイル build/vtsimnx_solver をビルドしてください。"
    elif isinstance(e, RuntimeError) and str(e).startswith("solver timed out"):
        detail["code"] = "solver_timeout"
        detail["hint"] = "VTSIMNX_SOLVER_TIMEOUT を延ばすか、計算条件（期間・分割・連成）を見直してください。"
    elif isinstance(e, RuntimeError) and str(e).startswith("solver failed:"):
        detail["code"] = "solver_execution_failed"
        detail["hint"] = "solver.log と入力JSONを確認し、設定値や境界条件の不整合を見直してください。"

    return detail


async def http_exception_handler(_request: Request, exc: HTTPException) -> Response:
    err = normalize_error_payload(exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": err})


async def request_validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> Response:
    err = normalize_error_payload(exc.errors())
    err["code"] = "validation_error"
    if not err.get("message"):
        err["message"] = "request validation failed"
    return JSONResponse(status_code=422, content={"error": err})


def register_exception_handlers(app) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
