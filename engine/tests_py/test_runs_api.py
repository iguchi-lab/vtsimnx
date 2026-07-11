"""非同期ジョブ API (/runs) のテスト。"""
import time

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


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


def _mock_solver(_unused=None):
    import app.services.simulation as sim_svc

    sim_svc.run_solver = lambda _cfg: {
        "status": "ok",
        "artifact_dir": "artifacts.mock",
        "result_files": {},
    }


def test_runs_submit_poll_result():
    import app.main as main_mod
    import app.services.simulation as sim_svc

    _mock_solver(main_mod)
    with TestClient(app) as client:
        resp = client.post("/runs", json={"config": MIN_CFG})
        assert resp.status_code == 202
        body = resp.json()
        run_id = body["run_id"]
        assert body["status"] == "queued"
        assert isinstance(body["input_hash"], str) and len(body["input_hash"]) == 64

        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            st = client.get(f"/runs/{run_id}")
            assert st.status_code == 200
            status = st.json()["status"]
            if status in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert status == "succeeded"

        result = client.get(f"/runs/{run_id}/result")
        assert result.status_code == 200
        data = result.json()
        assert data["result"]["status"] == "ok"


def test_runs_result_not_ready_returns_409():
    import app.main as main_mod
    import app.services.simulation as sim_svc
    import threading

    started = threading.Event()
    release = threading.Event()

    def slow_solver(_cfg):
        started.set()
        release.wait(timeout=5)
        return {"status": "ok", "artifact_dir": "artifacts.slow", "result_files": {}}

    _mock_solver(main_mod)
    sim_svc.run_solver = slow_solver

    with TestClient(app) as client:
        resp = client.post("/runs", json={"config": MIN_CFG})
        run_id = resp.json()["run_id"]
        assert started.wait(timeout=5)
        not_ready = client.get(f"/runs/{run_id}/result")
        assert not_ready.status_code == 409
        release.set()


def test_runs_cancel_queued(monkeypatch):
    import app.main as main_mod
    import app.services.simulation as sim_svc
    import threading

    monkeypatch.setenv("VTSIMNX_MAX_WORKERS", "1")
    block = threading.Event()
    started = threading.Event()

    def blocking_solver(_cfg):
        started.set()
        block.wait(timeout=10)
        return {"status": "ok", "artifact_dir": "artifacts.block", "result_files": {}}

    sim_svc.run_solver = blocking_solver
    try:
        with TestClient(app) as client:
            r1 = client.post("/runs", json={"config": {**MIN_CFG, "nodes": [{"key": "A"}]}})
            assert r1.status_code == 202
            assert started.wait(timeout=5)
            r2 = client.post("/runs", json={"config": {**MIN_CFG, "nodes": [{"key": "B"}]}})
            assert r2.status_code == 202
            id2 = r2.json()["run_id"]
            st2 = client.get(f"/runs/{id2}").json()
            assert st2["status"] == "queued"
            cancelled = client.delete(f"/runs/{id2}")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
    finally:
        block.set()


def test_runs_duplicate_hash_returns_same_run_id():
    import app.main as main_mod
    import app.services.simulation as sim_svc
    import threading

    gate = threading.Event()

    def slow(_cfg):
        gate.wait(timeout=5)
        return {"status": "ok", "artifact_dir": "artifacts.dup", "result_files": {}}

    sim_svc.run_solver = slow
    with TestClient(app) as client:
        payload = {"config": MIN_CFG}
        a = client.post("/runs", json=payload)
        b = client.post("/runs", json=payload)
        assert a.status_code == 202
        assert b.status_code == 200
        assert a.json()["run_id"] == b.json()["run_id"]
        gate.set()
