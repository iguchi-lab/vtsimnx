from __future__ import annotations

import gzip
import json
import time
from typing import Any, Dict, Optional

import requests

API_KEY_HEADER = "X-API-Key"


class RunCalcAPIError(RuntimeError):
    """
    /run の HTTP エラーを、code/message/hint が読める形で伝える。
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


def _format_http_error(resp: requests.Response) -> RunCalcAPIError:
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
        f"/run failed with HTTP {status}: {msg}",
        status_code=status,
        detail=detail,
        response=resp,
    )


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
    /run を叩いて JSON(dict) を返す。HTTPエラーは例外。
    """
    url = base_url.rstrip("/") + "/run"
    auth_headers = _api_key_headers(api_key)
    req_raw: bytes | None = None
    req_gz: bytes | None = None
    t_serialize_ms = 0.0
    t_gzip_ms = 0.0
    t_http_ms = 0.0
    t_resp_json_ms = 0.0

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

    if resp.status_code >= 400:
        raise _format_http_error(resp)

    t5 = time.perf_counter()
    out = resp.json()
    t6 = time.perf_counter()
    t_resp_json_ms = (t6 - t5) * 1000.0

    if not isinstance(out, dict):
        raise TypeError(f"/run response.json() must be dict, got {type(out).__name__}")

    if profile_out is not None:
        profile_out.clear()
        profile_out.update(
            {
                "request_serialize_ms": t_serialize_ms,
                "request_gzip_ms": t_gzip_ms,
                "http_roundtrip_ms": t_http_ms,
                "response_json_decode_ms": t_resp_json_ms,
                "request_payload_bytes_raw": len(req_raw) if req_raw is not None else 0,
                "request_payload_bytes_sent": len(req_gz) if req_gz is not None else (len(req_raw) if req_raw is not None else 0),
                "response_bytes": len(resp.content),
                "compress_request": bool(compress_request),
            }
        )
    return out


__all__ = ["API_KEY_HEADER", "RunCalcAPIError", "_api_key_headers", "_post_run"]
