"""HTTP API 認証ヘッダ（run_calc / artifacts 共用）。"""
from __future__ import annotations

from typing import Dict, Optional

API_KEY_HEADER = "X-API-Key"


def api_key_headers(api_key: Optional[str]) -> Dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {API_KEY_HEADER: key}


__all__ = ["API_KEY_HEADER", "api_key_headers"]
