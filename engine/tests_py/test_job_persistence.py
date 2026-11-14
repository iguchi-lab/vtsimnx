"""Restart recovery retains ownership/results without retaining simulation inputs."""
import json
from types import SimpleNamespace

import pytest

import app.jobs as jobs
from app.job_store import JobStore


def test_finished_job_survives_restart_without_input(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "execute_simulation", lambda *a, **k: SimpleNamespace(
        output={"status": "ok", "artifact_dir": "run-artifacts"}, warnings=[], warning_details=[]))
    manager = jobs.RunManager(state_dir=tmp_path)
    job, _ = manager.submit({"config": {"secret_input_marker": [1]*1000}, "owner_key_id": "owner"})
    job.future.result(timeout=5)
    manager.shutdown()
    snapshot = (tmp_path / (job.run_id + ".json")).read_text()
    assert "secret_input_marker" not in snapshot and '"request"' not in snapshot
    restored = jobs.RunManager(state_dir=tmp_path)
    try:
        saved = restored.get(job.run_id)
        assert saved.status == "succeeded"
        assert saved.owner_key_id == "owner"
        assert saved.result == job.result and saved.artifact_dir == "run-artifacts"
        assert saved.request == {} and saved.future is None
    finally:
        restored.shutdown()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_interrupted_job_fails_explicitly_without_resubmission(tmp_path, status):
    rid = "a" * 32
    JobStore(tmp_path).save({"run_id": rid, "status": status, "input_hash": "hash",
                             "created_at": jobs._utc_now(), "owner_key_id": "owner"})
    manager = jobs.RunManager(state_dir=tmp_path)
    try:
        job = manager.get(rid)
        assert job.status == "failed" and job.error["code"] == "server_restarted"
        assert job.owner_key_id == "owner" and job.future is None
        assert json.loads((tmp_path / (rid + ".json")).read_text())["status"] == "failed"
    finally:
        manager.shutdown()


def test_corrupt_snapshot_isolated_and_expired_snapshot_pruned(tmp_path):
    (tmp_path / "broken.json").write_text("not json")
    rid = "b" * 32
    JobStore(tmp_path).save({"run_id": rid, "status": "succeeded", "input_hash": "hash",
                             "created_at": "2020-01-01T00:00:00+00:00"})
    manager = jobs.RunManager(state_dir=tmp_path, job_ttl_sec=1)
    try:
        assert manager.job_count == 0
        assert not (tmp_path / (rid + ".json")).exists()
    finally:
        manager.shutdown()


def test_submission_storage_failure_does_not_leave_a_queued_job(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError("disk full")
    manager = jobs.RunManager(state_dir=tmp_path)
    monkeypatch.setattr(manager._store, "save", fail)
    try:
        with pytest.raises(OSError, match="disk full"):
            manager.submit({"config": {}})
        assert manager.job_count == 0 and manager._hash_index == {}
    finally:
        manager.shutdown()


def test_completion_storage_failure_does_not_change_solver_success(monkeypatch, tmp_path):
    manager = jobs.RunManager(state_dir=tmp_path)
    def execute(*a, **k):
        def fail(*args):
            raise OSError("disk full")
        monkeypatch.setattr(manager._store, "save", fail)
        return SimpleNamespace(output={"status": "ok"}, warnings=[], warning_details=[])
    monkeypatch.setattr(jobs, "execute_simulation", execute)
    try:
        job, _ = manager.submit({"config": {}})
        job.future.result(timeout=5)
        assert job.status == "succeeded" and job.result["result"]["status"] == "ok"
    finally:
        manager.shutdown()
