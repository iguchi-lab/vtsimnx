"""gzip リクエスト展開ミドルウェア。"""
from __future__ import annotations

import os
import zlib

from app.errors import error_response


class _BodyTooLarge(Exception):
    pass


class _GzipBody:
    """Bound every output allocation, including concatenated gzip members."""

    def __init__(self, limit: int):
        self.limit = limit
        self.output = bytearray()
        self.decoder: zlib._Decompress | None = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.completed = False

    def feed(self, data: bytes) -> None:
        while data:
            if self.decoder is None:
                data = data.lstrip(b"\0")  # gzip permits padding after a member
                if not data:
                    return
                self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            remaining = self.limit - len(self.output) + 1 if self.limit > 0 else 65536
            self.output.extend(self.decoder.decompress(data, min(65536, remaining)))
            if self.limit > 0 and len(self.output) > self.limit:
                raise _BodyTooLarge
            if self.decoder.eof:
                data = self.decoder.unused_data
                self.decoder = None
                self.completed = True
            else:
                data = self.decoder.unconsumed_tail

    def finish(self) -> bytes:
        if not self.completed or self.decoder is not None:
            raise EOFError("incomplete gzip member")
        return bytes(self.output)


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
        compressed_size = 0
        decoder = _GzipBody(max_decompressed)
        more_body = True
        try:
            while more_body:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
                if msg["type"] != "http.request":
                    continue
                chunk = msg.get("body", b"")
                compressed_size += len(chunk)
                if max_compressed > 0 and compressed_size > max_compressed:
                    resp = error_response(
                        413,
                        code="gzip_too_large",
                        message="gzip body too large",
                        hint="VTSIMNX_MAX_GZIP_BODY_BYTES を見直すか、入力を分割してください。",
                    )
                    return await resp(scope, receive, send)
                decoder.feed(chunk)
                more_body = msg.get("more_body", False)
            decompressed = decoder.finish()
        except _BodyTooLarge:
            resp = error_response(
                413, code="body_too_large", message="decompressed body too large",
                hint="VTSIMNX_MAX_JSON_BODY_BYTES を見直すか、入力を分割してください。",
            )
            return await resp(scope, receive, send)
        except (zlib.error, EOFError):
            resp = error_response(
                400,
                code="invalid_gzip",
                message="invalid gzip body",
                hint="Content-Encoding: gzip のボディが壊れているか、gzip ではありません。",
            )
            return await resp(scope, receive, send)

        sent = False

        async def receive2():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": decompressed, "more_body": False}

        new_headers = []
        for k, v in (scope.get("headers") or []):
            if k in (b"content-encoding", b"content-length"):
                continue
            new_headers.append((k, v))
        new_headers.append((b"content-length", str(len(decompressed)).encode("ascii")))
        return await self.app({**scope, "headers": new_headers}, receive2, send)
