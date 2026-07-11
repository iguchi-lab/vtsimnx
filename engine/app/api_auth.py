"""API キー認証・レート制限・監査ログ。

環境変数:
  VTSIMNX_API_KEY          単一キー（後方互換）
  VTSIMNX_API_KEYS         カンマ/改行区切りの複数キー
  VTSIMNX_API_KEYS_JSON    JSON 配列: [{"id":"ops","key":"...","revoked":false}, ...]
  VTSIMNX_RATE_LIMIT_PER_MIN  キーあたりの分間リクエスト上限（0 で無効、既定 120）

公開運用では TLS 終端（リバースプロキシ等）を前提とします。平文 HTTP での
API キー送信は推奨しません。
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.errors import error_response

API_KEY_HEADER = "X-API-Key"
logger = logging.getLogger("vtsimnx.audit")

# 認証不要（プローブ / 公開メタ）
AUTH_EXEMPT_PATHS = frozenset(
    {
        "/ping",
        "/health/live",
        "/health/ready",
        "/version",
    }
)


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    secret: str
    revoked: bool = False


def _split_api_keys(raw: str) -> list[str]:
    keys: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        key = part.strip()
        if key:
            keys.append(key)
    return keys


def load_api_key_records() -> list[ApiKeyRecord]:
    """設定から API キー一覧を読み込む（秘密はログに出さない）。"""
    records: list[ApiKeyRecord] = []
    seen_secrets: set[str] = set()

    raw_json = os.getenv("VTSIMNX_API_KEYS_JSON", "").strip()
    if raw_json:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.error("VTSIMNX_API_KEYS_JSON is invalid JSON (keys not loaded from JSON)")
            data = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                secret = str(item.get("key") or "").strip()
                if not secret:
                    continue
                key_id = str(item.get("id") or f"json-{i}").strip() or f"json-{i}"
                revoked = bool(item.get("revoked", False))
                records.append(ApiKeyRecord(key_id=key_id, secret=secret, revoked=revoked))
                seen_secrets.add(secret)

    for idx, secret in enumerate(_split_api_keys(os.getenv("VTSIMNX_API_KEY", ""))):
        if secret in seen_secrets:
            continue
        records.append(ApiKeyRecord(key_id=f"legacy-{idx}", secret=secret, revoked=False))
        seen_secrets.add(secret)

    for idx, secret in enumerate(_split_api_keys(os.getenv("VTSIMNX_API_KEYS", ""))):
        if secret in seen_secrets:
            continue
        # id:secret 形式を許可
        if ":" in secret and not secret.startswith("sk-"):
            kid, sep, rest = secret.partition(":")
            if sep and kid.strip() and rest.strip():
                records.append(ApiKeyRecord(key_id=kid.strip(), secret=rest.strip(), revoked=False))
                seen_secrets.add(rest.strip())
                continue
        records.append(ApiKeyRecord(key_id=f"key-{idx}", secret=secret, revoked=False))
        seen_secrets.add(secret)

    return records


def expected_api_keys() -> set[str]:
    """後方互換: 有効な秘密キー集合。"""
    return {r.secret for r in load_api_key_records() if not r.revoked}


def match_api_key(provided: str, records: list[ApiKeyRecord] | None = None) -> ApiKeyRecord | None:
    """定数時間比較でキーを照合する。一致しても revoked なら None。"""
    if not provided:
        return None
    recs = records if records is not None else load_api_key_records()
    provided_b = provided.encode("utf-8")
    matched: ApiKeyRecord | None = None
    # 全件を走査してタイミング差を抑える（最初の一致で break しない）
    for rec in recs:
        secret_b = rec.secret.encode("utf-8")
        if len(provided_b) != len(secret_b):
            # 長さが違う場合もダミー比較
            hmac.compare_digest(provided_b, provided_b)
            continue
        if hmac.compare_digest(provided_b, secret_b):
            matched = rec
    if matched is None or matched.revoked:
        return None
    return matched


class _RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key_id: str, *, limit_per_min: int) -> bool:
        if limit_per_min <= 0:
            return True
        now = time.monotonic()
        window = 60.0
        with self._lock:
            bucket = self._hits.setdefault(key_id, [])
            cutoff = now - window
            bucket[:] = [t for t in bucket if t >= cutoff]
            if len(bucket) >= limit_per_min:
                return False
            bucket.append(now)
            return True


_rate_limiter = _RateLimiter()


def _client_ip(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return "-"


def audit_log(
    event: str,
    *,
    key_id: str | None = None,
    path: str | None = None,
    run_id: str | None = None,
    artifact_dir: str | None = None,
    status: int | None = None,
    client_ip: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """構造化監査ログ。API キー秘密は絶対に出さない。"""
    payload: dict[str, Any] = {
        "event": event,
        "key_id": key_id or "-",
        "path": path or "-",
        "run_id": run_id or "-",
        "artifact_dir": artifact_dir or "-",
        "status": status,
        "client_ip": client_ip or "-",
    }
    if extra:
        for k, v in extra.items():
            if k in ("key", "api_key", "secret", "authorization"):
                continue
            payload[k] = v
    logger.info("audit %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))


class ApiKeyMiddleware:
    """
    API キー認証 + レート制限。
    キー未設定時は認証オフ（ローカル開発用）。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path") or ""
        method = scope.get("method") or ""
        client_ip = _client_ip(scope)

        records = load_api_key_records()
        active = [r for r in records if not r.revoked]
        if not active:
            scope["vtsimnx_key_id"] = None
            return await self.app(scope, receive, send)

        if path in AUTH_EXEMPT_PATHS:
            scope["vtsimnx_key_id"] = None
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        provided = headers.get(API_KEY_HEADER.lower().encode("ascii"), b"").decode("latin1").strip()
        matched = match_api_key(provided, records)
        if matched is None:
            audit_log(
                "auth_failed",
                path=f"{method} {path}",
                status=401,
                client_ip=client_ip,
            )
            resp = error_response(
                401,
                code="unauthorized",
                message="invalid or missing API key",
                hint="X-API-Key ヘッダに有効な API キーを指定してください（TLS 前提）。",
            )
            return await resp(scope, receive, send)

        try:
            limit = int(os.getenv("VTSIMNX_RATE_LIMIT_PER_MIN", "120"))
        except ValueError:
            limit = 120
        if not _rate_limiter.allow(matched.key_id, limit_per_min=limit):
            audit_log(
                "rate_limited",
                key_id=matched.key_id,
                path=f"{method} {path}",
                status=429,
                client_ip=client_ip,
            )
            resp = error_response(
                429,
                code="rate_limited",
                message="rate limit exceeded",
                hint="リクエスト頻度を下げてください（VTSIMNX_RATE_LIMIT_PER_MIN）。",
            )
            return await resp(scope, receive, send)

        scope["vtsimnx_key_id"] = matched.key_id
        audit_log(
            "auth_ok",
            key_id=matched.key_id,
            path=f"{method} {path}",
            client_ip=client_ip,
        )
        return await self.app(scope, receive, send)
