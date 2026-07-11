from __future__ import annotations

import gzip
import json
import time
from typing import Any, Dict, Optional

import requests


API_KEY_HEADER = "X-API-Key"


class RunCalcAPIError(RuntimeError):
    """
    シミュレーション API の HTTP エラーを、code/message/hint が読める形で伝える。
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Any = None,
        response: Optional[requests.Response] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.response = response


def _api_key_headers(api_key: Optional[str]) -> Dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {API_KEY_HEADER: key}


def _format_http_error(resp: requests.Response, *, endpoint: str = "/runs") -> RunCalcAPIError:
    status = resp.status_code
    detail: Any = None
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = body.get("detail", body)
        else:
            detail = body
    except Exception:
        detail = None

    if isinstance(detail, dict):
        parts: list[str] = []
        code = detail.get("code")
        if isinstance(code, str) and code:
            parts.append(f"[{code}]")
        message = detail.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())
        elif not parts:
            parts.append(str(detail))
        hint = detail.get("hint")
        if isinstance(hint, str) and hint.strip():
            parts.append(f"hint: {hint.strip()}")
        artifact_dir = detail.get("artifact_dir")
        if isinstance(artifact_dir, str) and artifact_dir:
            parts.append(f"artifact_dir={artifact_dir}")
        run_id = detail.get("run_id")
        if isinstance(run_id, str) and run_id:
            parts.append(f"run_id={run_id}")
        msg = " ".join(parts)
        log_tail = detail.get("log_tail")
        if isinstance(log_tail, str) and log_tail.strip():
            msg = f"{msg}\n--- log_tail ---\n{log_tail.strip()}"
    elif isinstance(detail, str) and detail.strip():
        msg = detail.strip()
    else:
        text = (resp.text or "").strip()
        msg = text[:800] if text else f"HTTP {status}"

    return RunCalcAPIError(
        f"{endpoint} failed with HTTP {status}: {msg}",
        status_code=status,
        detail=detail,
        response=resp,
    )


def _post_json(
    url: str,
    *,
    payload: Dict[str, Any],
    compress_request: bool,
    timeout: float,
    api_key: Optional[str] = None,
) -> tuple[requests.Response, Dict[str, Any]]:
    """POST JSON（任意 gzip）し、(response, timing_profile) を返す。"""
    auth_headers = _api_key_headers(api_key)
    req_raw: bytes | None = None
    req_gz: bytes | None = None
    t_serialize_ms = 0.0
    t_gzip_ms = 0.0
    t_http_ms = 0.0

    if compress_request:
        t0 = time.perf_counter()
        req_raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        t1 = time.perf_counter()
        req_gz = gzip.compress(req_raw)
        t2 = time.perf_counter()
        t_serialize_ms = (t1 - t0) * 1000.0
        t_gzip_ms = (t2 - t1) * 1000.0

        t3 = time.perf_counter()
        resp = requests.post(
            url,
            data=req_gz,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Accept": "application/json",
                **auth_headers,
            },
            timeout=timeout,
        )
        t4 = time.perf_counter()
        t_http_ms = (t4 - t3) * 1000.0
    else:
        t0 = time.perf_counter()
        req_raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        t1 = time.perf_counter()
        t_serialize_ms = (t1 - t0) * 1000.0

        t3 = time.perf_counter()
        resp = requests.post(url, json=payload, headers=auth_headers or None, timeout=timeout)
        t4 = time.perf_counter()
        t_http_ms = (t4 - t3) * 1000.0

    profile = {
        "request_serialize_ms": t_serialize_ms,
        "request_gzip_ms": t_gzip_ms,
        "http_roundtrip_ms": t_http_ms,
        "request_payload_bytes_raw": len(req_raw) if req_raw is not None else 0,
        "request_payload_bytes_sent": len(req_gz) if req_gz is not None else (len(req_raw) if req_raw is not None else 0),
        "response_bytes": len(resp.content),
        "compress_request": bool(compress_request),
    }
    return resp, profile


def _post_run(
    base_url: str,
    *,
    payload: Dict[str, Any],
    compress_request: bool,
    timeout: float,
    api_key: Optional[str] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    同期 POST /run を叩いて JSON(dict) を返す。HTTPエラーは例外。
    """
    url = base_url.rstrip("/") + "/run"
    resp, profile = _post_json(
        url,
        payload=payload,
        compress_request=compress_request,
        timeout=timeout,
        api_key=api_key,
    )

    if resp.status_code >= 400:
        raise _format_http_error(resp, endpoint="/run")

    t5 = time.perf_counter()
    out = resp.json()
    t6 = time.perf_counter()

    if not isinstance(out, dict):
        raise TypeError(f"/run response.json() must be dict, got {type(out).__name__}")

    if profile_out is not None:
        profile_out.clear()
        profile_out.update(profile)
        profile_out["response_json_decode_ms"] = (t6 - t5) * 1000.0
    return out


def _submit_and_wait(
    base_url: str,
    *,
    payload: Dict[str, Any],
    compress_request: bool,
    timeout: float,
    api_key: Optional[str] = None,
    poll_interval: float = 1.0,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    POST /runs → GET /runs/{id} ポーリング → GET /runs/{id}/result。
    timeout はポーリング打ち切り時間（秒）。
    """
    root = base_url.rstrip("/")
    auth_headers = _api_key_headers(api_key)
    poll_interval = max(0.05, float(poll_interval))
    deadline = time.perf_counter() + float(timeout)

    resp, submit_profile = _post_json(
        root + "/runs",
        payload=payload,
        compress_request=compress_request,
        timeout=min(timeout, 60.0),
        api_key=api_key,
    )
    if resp.status_code >= 400:
        raise _format_http_error(resp, endpoint="/runs")

    submit_body = resp.json()
    if not isinstance(submit_body, dict):
        raise TypeError(f"/runs response.json() must be dict, got {type(submit_body).__name__}")
    run_id = submit_body.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TypeError(f"/runs response missing run_id: {submit_body!r}")

    poll_count = 0
    status_url = f"{root}/runs/{run_id}"
    result_url = f"{root}/runs/{run_id}/result"
    last_status = submit_body.get("status")

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RunCalcAPIError(
                f"/runs polling timed out after {timeout}s (run_id={run_id}, last_status={last_status!r})",
                status_code=None,
                detail={"code": "timeout", "run_id": run_id, "status": last_status},
            )

        t_poll0 = time.perf_counter()
        status_resp = requests.get(
            status_url,
            headers=auth_headers or None,
            timeout=min(remaining, 60.0),
        )
        t_poll1 = time.perf_counter()
        poll_count += 1
        if status_resp.status_code >= 400:
            raise _format_http_error(status_resp, endpoint=f"/runs/{run_id}")

        status_body = status_resp.json()
        if not isinstance(status_body, dict):
            raise TypeError(f"/runs/{{id}} response must be dict, got {type(status_body).__name__}")
        status = status_body.get("status")
        last_status = status

        if status == "succeeded":
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RunCalcAPIError(
                    f"/runs polling timed out after {timeout}s (run_id={run_id}, last_status={last_status!r})",
                    status_code=None,
                    detail={"code": "timeout", "run_id": run_id, "status": last_status},
                )
            t_res0 = time.perf_counter()
            result_resp = requests.get(
                result_url,
                headers=auth_headers or None,
                timeout=min(remaining, 60.0),
            )
            t_res1 = time.perf_counter()
            if result_resp.status_code >= 400:
                raise _format_http_error(result_resp, endpoint=f"/runs/{run_id}/result")
            out = result_resp.json()
            if not isinstance(out, dict):
                raise TypeError(f"/runs/{{id}}/result must be dict, got {type(out).__name__}")
            if profile_out is not None:
                profile_out.clear()
                profile_out.update(submit_profile)
                profile_out["poll_count"] = poll_count
                profile_out["last_poll_ms"] = (t_poll1 - t_poll0) * 1000.0
                profile_out["result_fetch_ms"] = (t_res1 - t_res0) * 1000.0
                profile_out["run_id"] = run_id
                profile_out["response_bytes"] = len(result_resp.content)
            return out

        if status in ("failed", "cancelled"):
            err = status_body.get("error") if isinstance(status_body.get("error"), dict) else {}
            code = err.get("code") if isinstance(err, dict) else None
            message = err.get("message") if isinstance(err, dict) else None
            parts = [f"run {status}"]
            if isinstance(code, str) and code:
                parts.insert(0, f"[{code}]")
            if isinstance(message, str) and message.strip():
                parts.append(message.strip())
            raise RunCalcAPIError(
                f"/runs/{run_id} {status}: {' '.join(parts)}",
                status_code=None,
                detail=status_body,
            )

        if status not in ("queued", "running", None):
            raise RunCalcAPIError(
                f"/runs/{run_id} unexpected status: {status!r}",
                detail=status_body,
            )

        sleep_for = min(poll_interval, max(0.0, deadline - time.perf_counter()))
        if sleep_for <= 0:
            raise RunCalcAPIError(
                f"/runs polling timed out after {timeout}s (run_id={run_id}, last_status={last_status!r})",
                status_code=None,
                detail={"code": "timeout", "run_id": run_id, "status": last_status},
            )
        time.sleep(sleep_for)


__all__ = [
    "API_KEY_HEADER",
    "RunCalcAPIError",
    "_api_key_headers",
    "_post_run",
    "_submit_and_wait",
]
