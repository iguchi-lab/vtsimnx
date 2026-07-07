import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from app.api_auth import API_KEY_HEADER  # noqa: E402
from app.main import app  # noqa: E402


def test_ping_without_api_key_when_env_unset(monkeypatch):
    monkeypatch.delenv("VTSIMNX_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200


def test_ping_requires_api_key_when_env_set(monkeypatch):
    monkeypatch.setenv("VTSIMNX_API_KEY", "secret-token")
    client = TestClient(app)

    resp = client.get("/ping")
    assert resp.status_code == 401

    resp = client.get("/ping", headers={API_KEY_HEADER: "wrong"})
    assert resp.status_code == 401

    resp = client.get("/ping", headers={API_KEY_HEADER: "secret-token"})
    assert resp.status_code == 200


def test_run_accepts_api_key_when_env_set(monkeypatch):
    import app.main as main_mod

    monkeypatch.setenv("VTSIMNX_API_KEY", "secret-token")
    main_mod.run_solver = lambda _cfg: {"status": "ok"}
    client = TestClient(app)

    payload = {
        "config": {
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
    }

    resp = client.post("/run", json=payload)
    assert resp.status_code == 401

    resp = client.post("/run", json=payload, headers={API_KEY_HEADER: "secret-token"})
    assert resp.status_code == 200
