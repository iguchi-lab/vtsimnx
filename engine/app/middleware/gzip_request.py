"""圧縮・非圧縮リクエストのサイズ制限と gzip 展開。"""
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
    JSONを下流が読む前にサイズを検査し、gzip のときは展開する。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        enc_raw = headers.get(b"content-encoding", b"").decode("latin1").lower()
        enc_tokens = [t.strip() for t in enc_raw.split(",") if t.strip()]
        is_gzip = "gzip" in enc_tokens or "x-gzip" in enc_tokens

        max_compressed = int(os.getenv("VTSIMNX_MAX_GZIP_BODY_BYTES", str(64 * 1024 * 1024)))
        max_decompressed = int(os.getenv("VTSIMNX_MAX_JSON_BODY_BYTES", str(256 * 1024 * 1024)))
        wire_limit = max_compressed if is_gzip else max_decompressed
        wire_size = 0
        decoder = _GzipBody(max_decompressed) if is_gzip else None
        body = bytearray()

        async def reject_wire_size():
            resp = error_response(
                413,
                code="gzip_too_large" if is_gzip else "body_too_large",
                message="gzip body too large" if is_gzip else "request body too large",
                hint=("VTSIMNX_MAX_GZIP_BODY_BYTES" if is_gzip else "VTSIMNX_MAX_JSON_BODY_BYTES")
                     + " を見直すか、入力を分割してください。",
            )
            return await resp(scope, receive, send)

        try:
            declared_size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared_size = 0
        if wire_limit > 0 and declared_size > wire_limit:
            return await reject_wire_size()

        more_body = True
        try:
            while more_body:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
                if msg["type"] != "http.request":
                    continue
                chunk = msg.get("body", b"")
                wire_size += len(chunk)
                if wire_limit > 0 and wire_size > wire_limit:
                    return await reject_wire_size()
                if decoder is not None:
                    decoder.feed(chunk)
                else:
                    body.extend(chunk)
                more_body = msg.get("more_body", False)
            decompressed = decoder.finish() if decoder is not None else bytes(body)
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
            if k == b"content-length" or (is_gzip and k == b"content-encoding"):
                continue
            new_headers.append((k, v))
        new_headers.append((b"content-length", str(len(decompressed)).encode("ascii")))
        return await self.app({**scope, "headers": new_headers}, receive2, send)
