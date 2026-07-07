from __future__ import annotations

import os

from starlette.responses import JSONResponse

API_KEY_HEADER = "X-API-Key"


def _split_api_keys(raw: str) -> set[str]:
    keys: set[str] = set()
    for part in raw.replace("\n", ",").split(","):
        key = part.strip()
        if key:
            keys.add(key)
    return keys


def expected_api_keys() -> set[str]:
    keys = _split_api_keys(os.getenv("VTSIMNX_API_KEY", ""))
    keys.update(_split_api_keys(os.getenv("VTSIMNX_API_KEYS", "")))
    return keys


class ApiKeyMiddleware:
    """
    VTSIMNX_API_KEY / VTSIMNX_API_KEYS が設定されているときだけ API キー認証を有効化する。
    VTSIMNX_API_KEYS はカンマまたは改行区切りで複数キーを指定できる。
    未設定の場合は従来どおり認証なし（ローカル開発用）。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        expected = expected_api_keys()
        if not expected:
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        provided = headers.get(API_KEY_HEADER.lower().encode("ascii"), b"").decode("latin1").strip()
        if provided not in expected:
            resp = JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
            return await resp(scope, receive, send)

        return await self.app(scope, receive, send)
