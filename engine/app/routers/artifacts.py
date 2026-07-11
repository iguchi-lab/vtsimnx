"""artifact 取得 API。"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.api_auth import audit_log
from app.errors import raise_api_error
from app.services import artifact_store

router = APIRouter(tags=["artifacts"])


def _requester_key_id(request: Request) -> str | None:
    return getattr(request.scope, "get", lambda _k, _d=None: None)("vtsimnx_key_id")  # type: ignore[misc]


def _key_id(request: Request) -> str | None:
    return request.scope.get("vtsimnx_key_id")


@router.get("/artifacts/{artifact_dir}/manifest")
def get_artifact_manifest(artifact_dir: str, request: Request):
    """artifact_dir 配下の manifest.json を返す。"""
    artifact_path = artifact_store.resolve_artifact_dir(artifact_dir)
    artifact_store.assert_artifact_access(artifact_path, requester_key_id=_key_id(request))
    audit_log(
        "artifact_access",
        key_id=_key_id(request),
        artifact_dir=artifact_dir,
        path="GET /artifacts/.../manifest",
    )
    return artifact_store.load_manifest(artifact_path)


@router.get("/artifacts/{artifact_dir}/files")
def list_artifact_files(artifact_dir: str, request: Request):
    """ダウンロード可能なファイルキー一覧を返す（ホワイトリスト）。"""
    artifact_path = artifact_store.resolve_artifact_dir(artifact_dir)
    artifact_store.assert_artifact_access(artifact_path, requester_key_id=_key_id(request))
    manifest = artifact_store.load_manifest(artifact_path)
    keys = artifact_store.list_download_keys(manifest)
    return {"artifact_dir": artifact_dir, "keys": keys}


@router.get("/artifacts/{artifact_dir}/download/{key}")
def download_artifact_file(artifact_dir: str, key: str, request: Request):
    """
    ファイル本体を返す（巨大データは FileResponse によりストリーミング送信される）。
    key は /artifacts/{artifact_dir}/files で得たもののみ許可する。
    """
    artifact_path = artifact_store.resolve_artifact_dir(artifact_dir)
    artifact_store.assert_artifact_access(artifact_path, requester_key_id=_key_id(request))
    manifest = artifact_store.load_manifest(artifact_path)
    filename, media_type = artifact_store.artifact_file_from_key(manifest, key)

    if "/" in filename or "\\" in filename or ".." in filename:
        raise_api_error(
            400,
            code="invalid_filename",
            message="invalid filename in manifest",
            path=["filename"],
        )

    file_path = (artifact_path / filename).resolve()
    if artifact_path not in file_path.parents:
        raise_api_error(400, code="invalid_file_path", message="invalid file path")
    if not file_path.exists() or not file_path.is_file():
        raise_api_error(404, code="file_not_found", message="file not found", path=["key"])

    audit_log(
        "artifact_download",
        key_id=_key_id(request),
        artifact_dir=artifact_dir,
        path=f"GET /artifacts/.../download/{key}",
        extra={"file_key": key},
    )
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)
