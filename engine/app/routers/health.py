"""ヘルスチェックとバージョン情報。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from fastapi import APIRouter

from app.solver_runner import BASE_DIR, SOLVER_EXE
from app.versioning import SCHEMA_FORMAT_VERSION, get_package_version

router = APIRouter(tags=["health"])

# FastAPI app.version / OpenAPI 用。正本は pyproject.toml（get_package_version）。
API_VERSION = get_package_version()


@router.get("/ping")
def ping():
    """後方互換の軽量ヘルスチェック（/health/live と同等）。"""
    return {"status": "ok"}


@router.get("/health/live")
def health_live():
    """プロセス生存確認（liveness probe）。"""
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready():
    """依存の準備確認（readiness probe）。"""
    checks: dict[str, object] = {}
    ready = True

    solver_ok = Path(SOLVER_EXE).is_file() and os.access(SOLVER_EXE, os.X_OK)
    checks["solver_binary"] = {"ok": solver_ok, "path": str(SOLVER_EXE)}
    ready = ready and solver_ok

    work_dir = BASE_DIR / "work"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        probe = work_dir / ".ready_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        work_ok = True
    except OSError as e:
        work_ok = False
        checks["work_dir"] = {"ok": False, "path": str(work_dir), "error": str(e)}
        ready = False
    if work_ok:
        checks["work_dir"] = {"ok": True, "path": str(work_dir)}

    # 必要ライブラリ（Python 側の代表依存）
    for mod in ("fastapi", "pydantic", "numpy"):
        found = importlib.util.find_spec(mod) is not None
        checks[f"python:{mod}"] = {"ok": found}
        ready = ready and found

    status = "ok" if ready else "not_ready"
    body = {"status": status, "checks": checks}
    if not ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=body)
    return body


@router.get("/version")
def version():
    """client / API / solver / schema バージョン情報。"""
    pkg = get_package_version()
    client_version = None
    try:
        from importlib.metadata import version as pkg_version

        client_version = pkg_version("vtsimnx")
    except Exception:
        client_version = None

    solver_path = Path(SOLVER_EXE)
    return {
        "api_version": pkg,
        "client_version": client_version or pkg,
        "solver": {
            "path": str(solver_path),
            "present": solver_path.is_file(),
        },
        "schema_format_version": SCHEMA_FORMAT_VERSION,
    }
