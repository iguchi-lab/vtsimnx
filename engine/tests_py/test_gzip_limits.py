"""Bound memory before expansion finishes; exercise real ASGI chunk boundaries."""
import asyncio
import gzip
import json
import tracemalloc

import pytest

from app.middleware.gzip_request import GZipRequestMiddleware, _GzipBody, _BodyTooLarge


def invoke(monkeypatch, chunks, *, limit=100, compressed_limit=10000, disconnect=False, headers=None):
    monkeypatch.setenv("VTSIMNX_MAX_JSON_BODY_BYTES", str(limit))
    monkeypatch.setenv("VTSIMNX_MAX_GZIP_BODY_BYTES", str(compressed_limit))
    sent, received = [], []
    if headers is None:
        headers = [(b"content-encoding", b"gzip")]
    messages = [{"type": "http.request", "body": chunk, "more_body": i < len(chunks)-1 or disconnect}
                for i, chunk in enumerate(chunks)]
    if disconnect:
        messages.append({"type": "http.disconnect"})

    async def receive():
        assert messages, "Middleware read past the end of the request"
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    async def application(scope, receive, send):
        message = await receive()
        received.append(message["body"])
        assert dict(scope["headers"])[b"content-length"] == str(len(message["body"])).encode()
        if b"gzip" in dict(headers).get(b"content-encoding", b""):
            assert b"content-encoding" not in dict(scope["headers"])
        else:
            assert dict(scope["headers"]).get(b"content-encoding") == dict(headers).get(b"content-encoding")
        await send({"type": "http.response.start", "status": 200})

    asyncio.run(GZipRequestMiddleware(application)(
        {"type": "http", "headers": headers}, receive, send))
    return sent, received


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_gzip_members_share_one_limit(monkeypatch, chunk_size):
    compressed = gzip.compress(b"a"*50) + b"\0\0" + gzip.compress(b"b"*50) + b"\0"
    chunks = [compressed[i:i+chunk_size] for i in range(0, len(compressed), chunk_size)]
    sent, received = invoke(monkeypatch, chunks)
    assert sent[0]["status"] == 200 and received == [b"a"*50+b"b"*50]
    sent, received = invoke(monkeypatch, chunks, limit=99)
    assert sent[0]["status"] == 413 and not received
    assert json.loads(sent[1]["body"])["error"]["code"] == "body_too_large"


@pytest.mark.parametrize("content", [b"bad", gzip.compress(b"ok")[:-1],
                                     gzip.compress(b"ok") + b"garbage",
                                     gzip.compress(b"ok") + gzip.compress(b"second")[:-3]])
def test_invalid_or_truncated_members_rejected(monkeypatch, content):
    sent, received = invoke(monkeypatch, [content])
    assert sent[0]["status"] == 400 and not received


def test_large_expansion_does_not_allocate_the_full_output():
    compressed = gzip.compress(b"a" * (8 * 1024 * 1024))
    tracemalloc.start()
    try:
        decoder = _GzipBody(32768)
        with pytest.raises(_BodyTooLarge):
            decoder.feed(compressed)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 1024 * 1024
    finally:
        tracemalloc.stop()


def test_compressed_limit_and_disconnect(monkeypatch):
    content = gzip.compress(b"ok")
    sent, received = invoke(monkeypatch, [content], compressed_limit=len(content)-1)
    assert sent[0]["status"] == 413 and not received
    sent, received = invoke(monkeypatch, [content[:5]], disconnect=True)
    assert not sent and not received


def test_disabled_output_limit_and_empty_member(monkeypatch):
    assert invoke(monkeypatch, [gzip.compress(b"x"*1000)], limit=0)[1] == [b"x"*1000]
    assert invoke(monkeypatch, [gzip.compress(b"")])[1] == [b""]


@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")], [(b"content-encoding", b"identity")]])
def test_plain_body_limit_counts_actual_bytes(monkeypatch, chunk_size, headers):
    body = b"a" * 100
    chunks = [body[i:i+chunk_size] for i in range(0, len(body), chunk_size)]
    sent, received = invoke(monkeypatch, chunks, headers=headers, compressed_limit=1)
    assert sent[0]["status"] == 200 and received == [body]
    sent, received = invoke(monkeypatch, chunks, limit=99, headers=headers)
    assert sent[0]["status"] == 413 and not received
    assert json.loads(sent[1]["body"])["error"]["code"] == "body_too_large"


@pytest.mark.parametrize("headers,code", [
    ([(b"content-length", b"101")], "body_too_large"),
    ([(b"content-length", b"10001"), (b"content-encoding", b"gzip")], "gzip_too_large"),
])
def test_declared_oversized_body_is_rejected_before_reading(monkeypatch, headers, code):
    sent, received = invoke(monkeypatch, [], headers=headers)
    assert sent[0]["status"] == 413 and not received
    assert json.loads(sent[1]["body"])["error"]["code"] == code


def test_plain_disconnect_empty_and_disabled_limit(monkeypatch):
    assert invoke(monkeypatch, [b"abc"], headers=[], disconnect=True) == ([], [])
    assert invoke(monkeypatch, [b""], headers=[])[1] == [b""]
    assert invoke(monkeypatch, [b"a" * 1000], headers=[], limit=0)[1] == [b"a" * 1000]


def test_oversized_plain_chunk_is_not_copied(monkeypatch):
    body = b"x" * (8 * 1024 * 1024)
    tracemalloc.start()
    try:
        sent, received = invoke(monkeypatch, [body], headers=[])
        _, peak = tracemalloc.get_traced_memory()
        assert sent[0]["status"] == 413 and not received
        assert peak < 1024 * 1024
    finally:
        tracemalloc.stop()
