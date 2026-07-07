from __future__ import annotations

import os

from starlette.responses import JSONResponse

API_KEY_HEADER = "X-API-Key"


def expected_api_key() -> str:
    return os.getenv("VTSIMNX_API_KEY", "").strip()


class ApiKeyMiddleware:
    """
    VTSIMNX_API_KEY が設定されているときだけ API キー認証を有効化する。
    未設定の場合は従来どおり認証なし（ローカル開発用）。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        expected = expected_api_key()
        if not expected:
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        provided = headers.get(API_KEY_HEADER.lower().encode("ascii"), b"").decode("latin1").strip()
        if provided != expected:
            resp = JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
            return await resp(scope, receive, send)

        return await self.app(scope, receive, send)
