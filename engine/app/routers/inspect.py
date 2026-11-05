"""計算せずに入力を検査・展開する HTTP API。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api_auth import audit_log
from app.schemas.request import SimulationRequest
from app.schemas.response import BuildResponse, ValidateResponse
from app.services import inspect as inspect_svc

router = APIRouter(tags=["inspect"])


def _key_id(request: Request | None) -> str | None:
    if request is None:
        return None
    return request.scope.get("vtsimnx_key_id")


@router.post("/validate", response_model=ValidateResponse)
def validate_config(req: SimulationRequest, request: Request):
    """
    入力 JSON を検査する（ソルバは起動しない）。
    スキーマ、未知キー、builder 検証、意図警告を同一形式の issues で返す。
    """
    result = inspect_svc.inspect_config(req)
    audit_log(
        "validate",
        key_id=_key_id(request),
        path="POST /validate",
        status=200,
        extra={"ok": result["ok"], "issues": len(result["issues"])},
    )
    return {"ok": result["ok"], "issues": result["issues"]}


@router.post("/build", response_model=BuildResponse)
def build_config(req: SimulationRequest, request: Request):
    """
    入力 JSON を builder で展開する（ソルバは起動しない）。
    成功時は展開後 JSON とノード・枝の件数を返す。
    """
    result = inspect_svc.build_config_only(req)
    audit_log(
        "build",
        key_id=_key_id(request),
        path="POST /build",
        status=200,
        extra={"ok": result["ok"], "issues": len(result["issues"])},
    )
    return result
