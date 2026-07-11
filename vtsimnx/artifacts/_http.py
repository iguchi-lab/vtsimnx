"""成果物 API への HTTP アクセス。"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from vtsimnx._http_auth import api_key_headers

from .errors import ArtifactHTTPError
from ._schema import NormalizedManifest


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key is not None:
        key = api_key.strip()
        return key or None
    env_key = os.getenv("VTSIMNX_API_KEY", "").strip()
    return env_key or None


def headers(api_key: Optional[str]) -> Dict[str, str]:
    return api_key_headers(resolve_api_key(api_key))


def fetch_manifest(
    base_url: str,
    artifact_dir: str,
    *,
    timeout: float,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + f"/artifacts/{artifact_dir}/manifest"
    try:
        resp = requests.get(url, headers=headers(api_key), timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        raise ArtifactHTTPError(f"manifest GET failed: {e}", status_code=status) from e
    except requests.RequestException as e:
        raise ArtifactHTTPError(f"manifest GET failed: {e}") from e
    obj = resp.json()
    if not isinstance(obj, dict):
        raise TypeError(f"manifest response must be dict, got {type(obj).__name__}")
    return obj


def download_by_key(
    base_url: str,
    artifact_dir: str,
    key: str,
    *,
    timeout: float,
    api_key: Optional[str] = None,
) -> bytes:
    url = base_url.rstrip("/") + f"/artifacts/{artifact_dir}/download/{key}"
    try:
        resp = requests.get(url, headers=headers(api_key), timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        raise ArtifactHTTPError(f"download GET failed for key={key!r}: {e}", status_code=status) from e
    except requests.RequestException as e:
        raise ArtifactHTTPError(f"download GET failed for key={key!r}: {e}") from e
    return resp.content


def fetch_normalized_manifest(
    base_url: str,
    artifact_dir: str,
    *,
    timeout: float,
    api_key: Optional[str] = None,
) -> NormalizedManifest:
    raw = fetch_manifest(base_url, artifact_dir, timeout=timeout, api_key=api_key)
    return NormalizedManifest.from_dict(raw)


__all__ = [
    "resolve_api_key",
    "headers",
    "fetch_manifest",
    "download_by_key",
    "fetch_normalized_manifest",
]
