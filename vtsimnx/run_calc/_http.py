from __future__ import annotations

import gzip
import json
from typing import Any, Dict

import requests


def _post_run(
    base_url: str,
    *,
    payload: Dict[str, Any],
    compress_request: bool,
    timeout: float,
) -> Dict[str, Any]:
    """
    /run を叩いて JSON(dict) を返す。HTTPエラーは例外。
    """
    url = base_url.rstrip("/") + "/run"
    if compress_request:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        gz = gzip.compress(raw)
        resp = requests.post(
            url,
            data=gz,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
    else:
        resp = requests.post(url, json=payload, timeout=timeout)

    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, dict):
        raise TypeError(f"/run response.json() must be dict, got {type(out).__name__}")
    return out


__all__ = ["_post_run"]


