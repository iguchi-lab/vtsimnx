from __future__ import annotations

import gzip
import json
import time
from typing import Any, Dict, Optional

import requests

API_KEY_HEADER = "X-API-Key"


def _api_key_headers(api_key: Optional[str]) -> Dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {API_KEY_HEADER: key}


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

    resp.raise_for_status()
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


__all__ = ["API_KEY_HEADER", "_api_key_headers", "_post_run"]


