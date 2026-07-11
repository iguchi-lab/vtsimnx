"""Artifact ストレージ抽象とローカル実装・保持ポリシー。

環境変数:
  VTSIMNX_ARTIFACT_TTL_SEC                  成果物 TTL（秒）。0 で無効（既定 604800=7日）
  VTSIMNX_ARTIFACT_MAX_BYTES_PER_RUN        run あたり最大バイト（0 で無効、既定 2GiB）
  VTSIMNX_ARTIFACT_MAX_TOTAL_BYTES          work 全体上限（0 で無効、既定 50GiB）
  VTSIMNX_ARTIFACT_CLEANUP_MIN_INTERVAL_SEC 稼働中 cleanup の最短間隔（既定 300）
  VTSIMNX_ARTIFACT_STORE                    local（既定）。将来 s3 等を追加予定
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Protocol

logger = logging.getLogger(__name__)

_cleanup_lock = threading.RLock()
_active_runs: set[str] = set()
_active_lock = threading.Lock()
_last_cleanup_ts = 0.0

# 容量超過時に残す診断用ファイル
_DIAGNOSTIC_NAMES = frozenset(
    {
        "manifest.json",
        "owner.json",
        "error.json",
        "solver.log",
        "builder.log",
    }
)


def mark_run_active(run_id: str) -> None:
    with _active_lock:
        _active_runs.add(run_id)


def mark_run_inactive(run_id: str) -> None:
    with _active_lock:
        _active_runs.discard(run_id)


def active_run_ids() -> set[str]:
    with _active_lock:
        return set(_active_runs)


@dataclass(frozen=True)
class ArtifactPolicy:
    ttl_sec: int
    max_bytes_per_run: int
    max_total_bytes: int
    cleanup_min_interval_sec: int = 300

    @classmethod
    def from_env(cls) -> "ArtifactPolicy":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            ttl_sec=_int("VTSIMNX_ARTIFACT_TTL_SEC", 7 * 24 * 3600),
            max_bytes_per_run=_int("VTSIMNX_ARTIFACT_MAX_BYTES_PER_RUN", 2 * 1024**3),
            max_total_bytes=_int("VTSIMNX_ARTIFACT_MAX_TOTAL_BYTES", 50 * 1024**3),
            cleanup_min_interval_sec=_int("VTSIMNX_ARTIFACT_CLEANUP_MIN_INTERVAL_SEC", 300),
        )


class ArtifactStore(Protocol):
    def resolve(self, artifact_dir: str) -> Optional[Path]: ...

    def iter_artifacts(self) -> Iterator[tuple[str, Path, float]]:
        """Yield (artifact_dir_name, path, mtime_epoch)."""
        ...

    def delete(self, artifact_dir: str) -> bool: ...

    def total_bytes(self) -> int: ...

    def dir_bytes(self, path: Path) -> int: ...


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        return total
    return total


def _is_diagnostic_file(path: Path) -> bool:
    name = path.name
    if name in _DIAGNOSTIC_NAMES:
        return True
    if path.suffix.lower() == ".log":
        return True
    return False


def trim_run_artifacts_to_diagnostics(
    artifact_path: Path,
    *,
    reason: str,
) -> dict[str, int | str]:
    """
    巨大結果を削除し、ログ・manifest・owner・error.json だけ残す。
    ディスク保護と診断性のバランス用。
    """
    removed_files = 0
    freed_bytes = 0
    if not artifact_path.is_dir():
        return {"removed_files": 0, "freed_bytes": 0, "reason": reason}

    try:
        for p in sorted(artifact_path.rglob("*"), reverse=True):
            if not p.is_file():
                continue
            if _is_diagnostic_file(p):
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                sz = 0
            try:
                p.unlink()
                removed_files += 1
                freed_bytes += sz
            except OSError as e:
                logger.warning("failed to trim artifact file %s: %s", p, e)

        # 空ディレクトリ掃除（artifact 直下以外）
        for p in sorted(artifact_path.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    next(p.iterdir())
                except StopIteration:
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                except OSError:
                    pass

        err_path = artifact_path / "error.json"
        payload = {
            "code": "artifact_quota_exceeded",
            "message": reason,
            "trimmed": True,
            "removed_files": removed_files,
            "freed_bytes": freed_bytes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            err_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            logger.warning("failed to write error.json: %s", e)
    except OSError as e:
        logger.warning("trim_run_artifacts_to_diagnostics failed: %s", e)

    logger.info(
        "artifact trim audit: path=%s removed_files=%s freed_bytes=%s reason=%s",
        artifact_path,
        removed_files,
        freed_bytes,
        reason,
    )
    return {"removed_files": removed_files, "freed_bytes": freed_bytes, "reason": reason}


class LocalArtifactStore:
    """ローカル work/ 配下の成果物ストア。"""

    def __init__(self, work_root: Path):
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)

    def resolve(self, artifact_dir: str) -> Optional[Path]:
        if "/" in artifact_dir or "\\" in artifact_dir or ".." in artifact_dir:
            return None
        direct = (self.work_root / artifact_dir).resolve()
        try:
            if self.work_root.resolve() in direct.parents or direct == self.work_root.resolve():
                if direct.is_dir():
                    return direct
        except OSError:
            pass
        runs = self.work_root / "runs"
        if runs.is_dir():
            for candidate in runs.glob(f"*/{artifact_dir}"):
                if candidate.is_dir():
                    return candidate.resolve()
        return None

    def iter_artifacts(self) -> Iterator[tuple[str, Path, float]]:
        # work/artifacts.* と work/runs/*/artifacts.*
        seen: set[Path] = set()
        patterns = [self.work_root.glob("artifacts.*"), self.work_root.glob("runs/*/artifacts.*")]
        for group in patterns:
            for p in group:
                if not p.is_dir():
                    continue
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                yield p.name, p, mtime

    def delete(self, artifact_dir: str) -> bool:
        path = self.resolve(artifact_dir)
        if path is None or not path.exists():
            return False
        # 実行中 run 配下は触らない
        try:
            parts = path.parts
            if "runs" in parts:
                idx = parts.index("runs")
                if idx + 1 < len(parts) and parts[idx + 1] in active_run_ids():
                    logger.info("skip delete of active run artifact: %s", artifact_dir)
                    return False
        except Exception:
            pass
        try:
            shutil.rmtree(path)
            return True
        except OSError as e:
            logger.warning("failed to delete artifact %s: %s", artifact_dir, e)
            return False

    def total_bytes(self) -> int:
        return sum(_dir_size(p) for _, p, _ in self.iter_artifacts())

    def dir_bytes(self, path: Path) -> int:
        return _dir_size(path)


def get_artifact_store(work_root: Path | None = None) -> LocalArtifactStore:
    backend = (os.getenv("VTSIMNX_ARTIFACT_STORE") or "local").strip().lower()
    if work_root is None:
        from app.solver_runner import BASE_DIR

        work_root = BASE_DIR / "work"
    if backend != "local":
        logger.warning("unknown VTSIMNX_ARTIFACT_STORE=%s; falling back to local", backend)
    return LocalArtifactStore(work_root)


def enforce_run_size_limit(artifact_path: Path, *, policy: ArtifactPolicy | None = None) -> None:
    """
    run 完了後に容量超過なら巨大ファイルを trim し、RuntimeError を送出する。
    ログ / manifest / owner / error.json は残す。
    """
    pol = policy or ArtifactPolicy.from_env()
    if pol.max_bytes_per_run <= 0:
        return
    size = _dir_size(artifact_path)
    if size > pol.max_bytes_per_run:
        reason = f"artifact exceeds per-run limit: {size} > {pol.max_bytes_per_run} bytes"
        trim_run_artifacts_to_diagnostics(artifact_path, reason=reason)
        raise RuntimeError(reason)


def cleanup_artifacts(
    store: ArtifactStore | None = None,
    *,
    policy: ArtifactPolicy | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """
    TTL 超過と全体上限超過の成果物を削除する。
    実行中 run の成果物はスキップ。排他はプロセス内ロック。
    """
    pol = policy or ArtifactPolicy.from_env()
    st = store or get_artifact_store()
    now_ts = time.time() if now is None else now
    deleted_ttl = 0
    deleted_quota = 0
    bytes_before = 0
    bytes_after = 0

    with _cleanup_lock:
        items = list(st.iter_artifacts())
        bytes_before = sum(st.dir_bytes(p) for _, p, _ in items)
        # TTL
        if pol.ttl_sec > 0:
            for name, path, mtime in items:
                if now_ts - mtime > pol.ttl_sec:
                    if st.delete(name):
                        deleted_ttl += 1

        # 全体上限: 古い順に削除
        if pol.max_total_bytes > 0:
            items = list(st.iter_artifacts())
            sized = []
            total = 0
            for name, path, mtime in items:
                sz = st.dir_bytes(path)
                sized.append((mtime, name, sz))
                total += sz
            if total > pol.max_total_bytes:
                sized.sort(key=lambda x: x[0])  # oldest first
                for _mtime, name, sz in sized:
                    if total <= pol.max_total_bytes:
                        break
                    if st.delete(name):
                        total -= sz
                        deleted_quota += 1

        bytes_after = sum(st.dir_bytes(p) for _, p, _ in st.iter_artifacts())

    stats = {
        "deleted_ttl": deleted_ttl,
        "deleted_quota": deleted_quota,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }
    if deleted_ttl or deleted_quota:
        logger.info(
            "artifact cleanup audit: deleted_ttl=%s deleted_quota=%s bytes_before=%s bytes_after=%s",
            deleted_ttl,
            deleted_quota,
            bytes_before,
            bytes_after,
        )
    return stats


def maybe_cleanup_artifacts(
    store: ArtifactStore | None = None,
    *,
    policy: ArtifactPolicy | None = None,
    force: bool = False,
) -> dict[str, int | bool]:
    """
    稼働中の軽量 cleanup。最短間隔未満ならスキップする（debounce）。
    run 完了後などから呼ぶ。
    """
    global _last_cleanup_ts
    pol = policy or ArtifactPolicy.from_env()
    interval = max(0, int(pol.cleanup_min_interval_sec))
    now = time.time()
    with _cleanup_lock:
        if not force and interval > 0 and (now - _last_cleanup_ts) < interval:
            return {
                "skipped": True,
                "deleted_ttl": 0,
                "deleted_quota": 0,
                "bytes_before": 0,
                "bytes_after": 0,
            }
        _last_cleanup_ts = now

    stats = cleanup_artifacts(store, policy=pol, now=now)
    return {"skipped": False, **stats}


def reset_cleanup_debounce_for_tests() -> None:
    """テスト用に debounce 時刻をリセットする。"""
    global _last_cleanup_ts
    with _cleanup_lock:
        _last_cleanup_ts = 0.0


def write_owner_metadata(artifact_path: Path, *, key_id: str | None, run_id: str | None) -> None:
    meta = {
        "owner_key_id": key_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (artifact_path / "owner.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("failed to write owner.json: %s", e)


def read_owner_key_id(artifact_path: Path) -> str | None:
    p = artifact_path / "owner.json"
    if not p.exists():
        # 旧成果物: 所有者未設定 → 認可スキップ（後方互換）
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    kid = data.get("owner_key_id")
    return str(kid) if kid else None
