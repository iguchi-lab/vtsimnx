"""同期 /run と非同期 /runs。"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from app.api_auth import audit_log
from app.builder.validate import ConfigFileError, ValidationError
from app.errors import build_bad_request_detail, build_internal_error_detail, raise_api_error
from app.jobs import get_run_manager
from app.schemas import SimulationRequest, SimulationResponse, UnknownFieldError
from app.services import simulation as sim_svc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["runs"])


def _key_id(request: Request | None) -> str | None:
    if request is None:
        return None
    return request.scope.get("vtsimnx_key_id")


@router.post("/run", response_model=SimulationResponse)
def run_simulation(req: SimulationRequest, request: Request):
    """
    入力 JSON（config）を C++ ソルバに渡して、結果 JSON を返す（同期・互換API）。
    長時間計算は POST /runs を推奨。
    """
    run_id: str | None = None
    key_id = _key_id(request)
    try:
        result = sim_svc.run_from_request(req, owner_key_id=key_id)
        run_id = None
        if isinstance(result, dict):
            # SimulationResponse may wrap result
            pass
        out = result if isinstance(result, SimulationResponse) else result
        artifact_dir = None
        if isinstance(out, SimulationResponse) and isinstance(out.result, dict):
            artifact_dir = out.result.get("artifact_dir")
        audit_log(
            "run_completed",
            key_id=key_id,
            path="POST /run",
            artifact_dir=str(artifact_dir) if artifact_dir else None,
            status=200,
        )
        return result
    except UnknownFieldError as e:
        raise HTTPException(status_code=422, detail=e.details or str(e))
    except (ValidationError, ConfigFileError, ValueError, KeyError, TypeError) as e:
        logger.info("validation/config error in /run: %s", e)
        raise HTTPException(status_code=400, detail=build_bad_request_detail(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("internal error in /run")
        raise HTTPException(status_code=500, detail=build_internal_error_detail(e, run_id=run_id))


@router.post("/runs")
def create_run(req: SimulationRequest, response: Response, request: Request):
    """非同期ジョブを投入し、run_id を即時返す。重複入力は既存 run_id を 200 で返す。"""
    manager = get_run_manager()
    key_id = _key_id(request)
    try:
        raw_config, pre_warnings, pre_details = sim_svc.submit_run_payload(req)
    except UnknownFieldError as e:
        raise HTTPException(status_code=422, detail=e.details or str(e))
    request_payload = req.model_dump(mode="python")
    request_payload["config"] = raw_config
    request_payload["initial_warnings"] = pre_warnings
    request_payload["initial_warning_details"] = pre_details
    request_payload["owner_key_id"] = key_id
    job, is_duplicate = manager.submit(request_payload)
    response.status_code = 200 if is_duplicate else 202
    audit_log(
        "run_submitted",
        key_id=key_id,
        run_id=job.run_id,
        path="POST /runs",
        status=response.status_code,
        extra={"duplicate": is_duplicate},
    )
    return {
        "run_id": job.run_id,
        "status": job.status,
        "input_hash": job.input_hash,
    }


@router.get("/runs/{run_id}")
def get_run_status(run_id: str, request: Request):
    job = get_run_manager().get(run_id)
    if job is None:
        raise_api_error(404, code="run_not_found", message="run not found", run_id=run_id)
    owner = (job.request or {}).get("owner_key_id")
    kid = _key_id(request)
    if owner and kid and owner != kid:
        raise_api_error(403, code="forbidden_run", message="run access denied", run_id=run_id)
    return job.to_status_dict()


@router.get("/runs/{run_id}/result", response_model=SimulationResponse)
def get_run_result(run_id: str, request: Request):
    job = get_run_manager().get(run_id)
    if job is None:
        raise_api_error(404, code="run_not_found", message="run not found", run_id=run_id)
    owner = (job.request or {}).get("owner_key_id")
    kid = _key_id(request)
    if owner and kid and owner != kid:
        raise_api_error(403, code="forbidden_run", message="run access denied", run_id=run_id)
    if job.status == "succeeded" and isinstance(job.result, dict):
        return SimulationResponse(**job.result)
    if job.status in ("queued", "running"):
        raise_api_error(
            409,
            code="not_ready",
            message=f"run status is {job.status}",
            run_id=run_id,
        )
    if job.status == "cancelled":
        extra = dict(job.error or {})
        raise_api_error(
            409,
            code=str(extra.get("code") or "cancelled"),
            message=str(extra.get("message") or "run was cancelled"),
            run_id=run_id,
            hint=extra.get("hint"),
            extra={k: v for k, v in extra.items() if k not in ("code", "message", "hint", "run_id")},
        )
    err = job.error or {"code": "internal_error", "message": "run failed"}
    raise HTTPException(status_code=500, detail={**err, "run_id": run_id})


@router.delete("/runs/{run_id}")
def cancel_run(run_id: str, request: Request):
    job = get_run_manager().get(run_id)
    if job is None:
        raise_api_error(404, code="run_not_found", message="run not found", run_id=run_id)
    owner = (job.request or {}).get("owner_key_id")
    kid = _key_id(request)
    if owner and kid and owner != kid:
        raise_api_error(403, code="forbidden_run", message="run access denied", run_id=run_id)
    job = get_run_manager().cancel(run_id)
    if job is None:
        raise_api_error(404, code="run_not_found", message="run not found", run_id=run_id)
    audit_log("run_cancelled", key_id=kid, run_id=run_id, path="DELETE /runs/{run_id}")
    return job.to_status_dict()
