"""
FastAPI アプリケーション入口。

- ルータ / ミドルウェア / lifespan の組み立て
- CLI は `python -m app.main`
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path as _Path

if __name__ == "__main__" and (globals().get("__package__") in (None, "")):
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from fastapi import FastAPI

from app.api_auth import ApiKeyMiddleware
from app.errors import register_exception_handlers
from app.jobs import RunManager, set_run_manager
from app.middleware import GZipRequestMiddleware
from app.routers import artifacts, health, inspect, runs
from app.routers.health import API_VERSION
from app.services.artifact_policy import cleanup_artifacts, get_artifact_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    manager = RunManager(state_dir=WORK_DIR / "jobs")
    set_run_manager(manager)
    try:
        # 起動時に TTL / 容量ポリシーで掃除
        try:
            stats = cleanup_artifacts(get_artifact_store(WORK_DIR))
            logger.info("artifact cleanup on startup: %s", stats)
        except Exception:
            logger.exception("artifact cleanup on startup failed")
        yield
    finally:
        try:
            cleanup_artifacts(get_artifact_store(WORK_DIR))
        except Exception:
            logger.exception("artifact cleanup on shutdown failed")
        manager.shutdown(wait=False)
        set_run_manager(None)


app = FastAPI(title="VTSimNX API", version=API_VERSION, lifespan=_lifespan)
register_exception_handlers(app)
app.add_middleware(GZipRequestMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.include_router(health.router)
app.include_router(runs.router)
app.include_router(inspect.router)
app.include_router(artifacts.router)

BASE_DIR = _Path(__file__).resolve().parent.parent
WORK_DIR = BASE_DIR / "work"


if __name__ == "__main__":
    from app.cli import main as cli_main

    raise SystemExit(cli_main())
