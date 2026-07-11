"""artifact ディレクトリ解決と manifest / ファイル参照。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from app.errors import raise_api_error
from app.services.artifact_policy import read_owner_key_id
from app.solver_runner import resolve_artifact_path


def resolve_artifact_dir(artifact_dir: str) -> Path:
    """パストラバーサル防止付きで artifact ディレクトリを解決する。"""
    if "/" in artifact_dir or "\\" in artifact_dir or ".." in artifact_dir:
        raise_api_error(
            400,
            code="invalid_artifact_dir",
            message="invalid artifact_dir",
            hint="artifact_dir はディレクトリ名のみを指定してください（パス区切りや .. は不可）。",
            path=["artifact_dir"],
        )
    p = resolve_artifact_path(artifact_dir)
    if p is None:
        raise_api_error(
            404,
            code="artifact_not_found",
            message="artifact_dir not found",
            hint="run の結果に含まれる artifact_dir を確認するか、TTL で削除されていないか確認してください。",
            path=["artifact_dir"],
        )
    return p


def assert_artifact_access(artifact_path: Path, *, requester_key_id: str | None) -> None:
    """
    所有者キーがある成果物は、同一 key_id のみアクセス可。
    認証無効（requester_key_id is None かつサーバも認証オフ）のときは許可。
    """
    owner = read_owner_key_id(artifact_path)
    if owner is None:
        return
    if requester_key_id is None:
        # 認証オフ運用、または exempt 経路からの呼び出し
        from app.api_auth import expected_api_keys

        if not expected_api_keys():
            return
        raise_api_error(
            403,
            code="forbidden_artifact",
            message="artifact access denied",
            hint="この成果物は別の API キー所有者に紐づいています。",
        )
    if requester_key_id != owner:
        raise_api_error(
            403,
            code="forbidden_artifact",
            message="artifact access denied",
            hint="この成果物は別の API キー所有者に紐づいています。",
        )


def load_manifest(artifact_path: Path) -> Dict[str, Any]:
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise_api_error(
            404,
            code="manifest_not_found",
            message="manifest.json not found",
            hint="先に /run または /runs を実行して成果物を生成してください。",
        )
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise_api_error(
            500,
            code="manifest_read_failed",
            message=f"failed to read manifest.json: {e}",
        )


def artifact_file_from_key(manifest: Dict[str, Any], key: str) -> Tuple[str, str]:
    """
    manifest(output.json相当) からキーに対応するファイル名を返す。
    戻り値: (filename, media_type)
    """
    out = manifest.get("output", {})
    if not isinstance(out, dict):
        raise_api_error(500, code="invalid_manifest", message="invalid manifest format")

    if key == "log":
        name = out.get("log_file")
        if not isinstance(name, str) or not name:
            raise_api_error(404, code="file_not_found", message="log_file not available", path=["log"])
        return name, "text/plain"

    if key == "builder_log":
        name = out.get("builder_log_file")
        if not isinstance(name, str) or not name:
            raise_api_error(
                404, code="file_not_found", message="builder_log_file not available", path=["builder_log"]
            )
        return name, "text/plain"

    if key == "manifest":
        return "manifest.json", "application/json"

    result_files = out.get("result_files", {})
    if not isinstance(result_files, dict):
        raise_api_error(404, code="file_not_found", message="result_files not available")
    name = result_files.get(key)
    if not isinstance(name, str) or not name:
        raise_api_error(
            404,
            code="unknown_file_key",
            message=f"unknown file key: {key}",
            path=["key"],
            hint="/artifacts/{artifact_dir}/files で取得したキーのみ指定できます。",
        )

    if name.endswith(".json"):
        return name, "application/json"
    if name.endswith(".log"):
        return name, "text/plain"
    return name, "application/octet-stream"


def list_download_keys(manifest: Dict[str, Any]) -> list[str]:
    out = manifest.get("output", {})
    result_files = out.get("result_files", {}) if isinstance(out, dict) else {}
    keys: list[str] = []
    if isinstance(result_files, dict):
        keys.extend(sorted([k for k, v in result_files.items() if isinstance(v, str) and v]))
    if isinstance(out, dict) and isinstance(out.get("log_file"), str) and out.get("log_file"):
        keys.append("log")
    if isinstance(out, dict) and isinstance(out.get("builder_log_file"), str) and out.get("builder_log_file"):
        keys.append("builder_log")
    keys.append("manifest")
    return keys
