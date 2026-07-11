"""ジョブ失敗判定・prune・artifact trim / debounce のテスト。"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.jobs import RunManager, failure_from_output, output_indicates_failure  # noqa: E402
from app.main import app  # noqa: E402
from app.services.artifact_policy import (  # noqa: E402
    ArtifactPolicy,
    LocalArtifactStore,
    enforce_run_size_limit,
    maybe_cleanup_artifacts,
    reset_cleanup_debounce_for_tests,
    trim_run_artifacts_to_diagnostics,
)


MIN_CFG = {
    "simulation": {
        "index": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T01:00:00Z",
            "timestep": 60,
            "length": 60,
        }
    },
    "nodes": [{"key": "N1"}],
    "ventilation_branches": [],
    "thermal_branches": [],
}


def test_output_indicates_failure_helpers():
    assert output_indicates_failure({"status": "error", "error": "boom"})
    assert output_indicates_failure({"status": "ok", "error": "still bad"})
    assert not output_indicates_failure({"status": "ok"})
    err = failure_from_output({"status": "error", "error": "bad", "artifact_dir": "artifacts.x"}, run_id="r1")
    assert err["code"] == "solver_error"
    assert err["artifact_dir"] == "artifacts.x"
    assert err["run_id"] == "r1"


def test_runs_marks_failed_when_solver_returns_error_status():
    import app.services.simulation as sim_svc

    sim_svc.run_solver = lambda _cfg, **_kw: {
        "status": "error",
        "error": "solver failed: singular matrix",
        "artifact_dir": "artifacts.err",
        "result_files": {},
    }
    with TestClient(app) as client:
        resp = client.post("/runs", json={"config": MIN_CFG})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        deadline = time.time() + 10
        status = None
        body = None
        while time.time() < deadline:
            st = client.get(f"/runs/{run_id}")
            assert st.status_code == 200
            body = st.json()
            status = body["status"]
            if status in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert status == "failed"
        assert body["error"]["code"] == "solver_error"
        assert "singular" in body["error"]["message"]
        result = client.get(f"/runs/{run_id}/result")
        assert result.status_code == 500


def test_job_prune_by_ttl_and_max_records():
    mgr = RunManager(max_workers=1, job_ttl_sec=3600, max_job_records=2)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=2)).isoformat()
    recent = now.isoformat()

    # inject finished jobs directly
    from app.jobs import JobRecord

    with mgr._lock:
        for i, finished in enumerate([old, old, recent]):
            rid = f"job{i}"
            mgr._jobs[rid] = JobRecord(
                run_id=rid,
                status="succeeded",
                input_hash=f"h{i}",
                created_at=finished,
                finished_at=finished,
                request={"config": {"big": "x" * 100}},
                result={"result": {"status": "ok"}},
            )
        removed = mgr._prune_jobs_unlocked(now=now.timestamp())
    assert removed >= 2  # two expired by TTL
    assert mgr.job_count <= 2
    assert mgr.get("job2") is not None  # recent kept if within max


def test_job_prune_max_records_keeps_running():
    mgr = RunManager(max_workers=1, job_ttl_sec=0, max_job_records=1)
    from app.jobs import JobRecord

    with mgr._lock:
        mgr._jobs["done"] = JobRecord(
            run_id="done",
            status="succeeded",
            input_hash="h0",
            created_at="2020-01-01T00:00:00+00:00",
            finished_at="2020-01-01T00:00:00+00:00",
            request={},
        )
        mgr._jobs["run"] = JobRecord(
            run_id="run",
            status="running",
            input_hash="h1",
            created_at="2020-01-02T00:00:00+00:00",
            request={},
        )
        mgr._prune_jobs_unlocked()
    assert mgr.get("run") is not None
    assert mgr.get("done") is None


def test_trim_oversized_run_keeps_diagnostics(tmp_path):
    art = tmp_path / "artifacts.big"
    art.mkdir()
    (art / "solver.log").write_text("log\n", encoding="utf-8")
    (art / "manifest.json").write_text("{}", encoding="utf-8")
    (art / "owner.json").write_text("{}", encoding="utf-8")
    big = art / "thermal_temperature.csv"
    big.write_bytes(b"x" * 10_000)
    (art / "nested").mkdir()
    (art / "nested" / "huge.bin").write_bytes(b"y" * 5_000)

    stats = trim_run_artifacts_to_diagnostics(art, reason="over limit")
    assert stats["removed_files"] >= 2
    assert (art / "solver.log").exists()
    assert (art / "manifest.json").exists()
    assert (art / "owner.json").exists()
    assert (art / "error.json").exists()
    assert not big.exists()
    assert not (art / "nested" / "huge.bin").exists()


def test_enforce_run_size_limit_trims_and_raises(tmp_path):
    art = tmp_path / "artifacts.q"
    art.mkdir()
    (art / "solver.log").write_text("ok\n", encoding="utf-8")
    (art / "data.csv").write_bytes(b"z" * 2000)
    policy = ArtifactPolicy(ttl_sec=0, max_bytes_per_run=500, max_total_bytes=0, cleanup_min_interval_sec=0)
    with pytest.raises(RuntimeError, match="per-run limit"):
        enforce_run_size_limit(art, policy=policy)
    assert (art / "solver.log").exists()
    assert (art / "error.json").exists()
    assert not (art / "data.csv").exists()


def test_maybe_cleanup_debounces(tmp_path, monkeypatch):
    reset_cleanup_debounce_for_tests()
    work = tmp_path / "work"
    store = LocalArtifactStore(work)
    art = work / "artifacts.old"
    art.mkdir()
    (art / "schema.json").write_text("{}", encoding="utf-8")
    import os

    old = time.time() - 10_000
    os.utime(art, (old, old))
    policy = ArtifactPolicy(ttl_sec=60, max_bytes_per_run=0, max_total_bytes=0, cleanup_min_interval_sec=3600)
    first = maybe_cleanup_artifacts(store, policy=policy, force=True)
    assert first["skipped"] is False
    assert first["deleted_ttl"] >= 1
    # recreate and ensure debounce skips
    art2 = work / "artifacts.old2"
    art2.mkdir()
    (art2 / "schema.json").write_text("{}", encoding="utf-8")
    os.utime(art2, (old, old))
    second = maybe_cleanup_artifacts(store, policy=policy, force=False)
    assert second["skipped"] is True
    assert art2.exists()


def test_call_run_solver_does_not_swallow_typeerror():
    import app.services.simulation as sim_svc

    calls = {"n": 0}

    def boom(_cfg, **_kw):
        calls["n"] += 1
        raise TypeError("real internal type error")

    original = sim_svc.run_solver
    sim_svc.run_solver = boom
    try:
        with pytest.raises(TypeError, match="real internal"):
            sim_svc._call_run_solver({}, run_id="x", cancel_event=None)
        assert calls["n"] == 1
    finally:
        sim_svc.run_solver = original
