"""gzip リクエスト展開ミドルウェア。"""
from __future__ import annotations

import gzip
import os

from app.errors import error_response


class GZipRequestMiddleware:
    """
    Content-Encoding: gzip のとき、リクエストボディを展開して下流へ渡す。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        enc_raw = headers.get(b"content-encoding", b"").decode("latin1").lower()
        enc_tokens = [t.strip() for t in enc_raw.split(",") if t.strip()]
        if "gzip" not in enc_tokens and "x-gzip" not in enc_tokens:
            return await self.app(scope, receive, send)

        max_compressed = int(os.getenv("VTSIMNX_MAX_GZIP_BODY_BYTES", str(64 * 1024 * 1024)))
        max_decompressed = int(os.getenv("VTSIMNX_MAX_JSON_BODY_BYTES", str(256 * 1024 * 1024)))
        body = b""
        more_body = True
        try:
            while more_body:
                msg = await receive()
                if msg["type"] != "http.request":
                    continue
                body += msg.get("body", b"")
                if max_compressed > 0 and len(body) > max_compressed:
                    resp = error_response(
                        413,
                        code="gzip_too_large",
                        message="gzip body too large",
                        hint="VTSIMNX_MAX_GZIP_BODY_BYTES を見直すか、入力を分割してください。",
                    )
                    return await resp(scope, receive, send)
                more_body = msg.get("more_body", False)
            decompressed = gzip.decompress(body)
            if max_decompressed > 0 and len(decompressed) > max_decompressed:
                resp = error_response(
                    413,
                    code="body_too_large",
                    message="decompressed body too large",
                    hint="VTSIMNX_MAX_JSON_BODY_BYTES を見直すか、入力を分割してください。",
                )
                return await resp(scope, receive, send)
        except Exception:
            resp = error_response(
                400,
                code="invalid_gzip",
                message="invalid gzip body",
                hint="Content-Encoding: gzip のボディが壊れているか、gzip ではありません。",
            )
            return await resp(scope, receive, send)

        async def receive2():
            return {"type": "http.request", "body": decompressed, "more_body": False}

        new_headers = []
        for k, v in (scope.get("headers") or []):
            if k in (b"content-encoding", b"content-length"):
                continue
            new_headers.append((k, v))
        new_headers.append((b"content-length", str(len(decompressed)).encode("ascii")))
        scope["headers"] = new_headers

        return await self.app(scope, receive2, send)
