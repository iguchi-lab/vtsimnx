import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from app.api_auth import API_KEY_HEADER  # noqa: E402
from app.main import app  # noqa: E402


def test_ping_without_api_key_when_env_unset(monkeypatch):
    monkeypatch.delenv("VTSIMNX_API_KEY", raising=False)
    monkeypatch.delenv("VTSIMNX_API_KEYS", raising=False)
    monkeypatch.delenv("VTSIMNX_API_KEYS_JSON", raising=False)
    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200


def test_health_exempt_when_api_key_required(monkeypatch):
    """プローブ系は認証不要。保護対象は /run。"""
    monkeypatch.setenv("VTSIMNX_API_KEY", "secret-token")
    client = TestClient(app)

    assert client.get("/ping").status_code == 200
    assert client.get("/health/live").status_code == 200

    resp = client.post("/run", json={"config": {}})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"

    resp = client.post("/run", json={"config": {}}, headers={API_KEY_HEADER: "wrong"})
    assert resp.status_code == 401


def test_run_accepts_api_key_when_env_set(monkeypatch):
    import app.services.simulation as sim_svc

    monkeypatch.setenv("VTSIMNX_API_KEY", "secret-token")
    sim_svc.run_solver = lambda _cfg: {"status": "ok"}
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


def test_run_accepts_any_key_from_api_keys_env(monkeypatch):
    import app.services.simulation as sim_svc

    monkeypatch.delenv("VTSIMNX_API_KEY", raising=False)
    monkeypatch.setenv("VTSIMNX_API_KEYS", "user-a-key, user-b-key")
    sim_svc.run_solver = lambda _cfg: {"status": "ok"}
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

    assert client.post("/run", json=payload).status_code == 401
    assert client.post("/run", json=payload, headers={API_KEY_HEADER: "user-a-key"}).status_code == 200
    assert client.post("/run", json=payload, headers={API_KEY_HEADER: "user-b-key"}).status_code == 200


def test_run_merges_single_and_multi_key_env(monkeypatch):
    import app.services.simulation as sim_svc

    monkeypatch.setenv("VTSIMNX_API_KEY", "legacy-key")
    monkeypatch.setenv("VTSIMNX_API_KEYS", "shared-key")
    sim_svc.run_solver = lambda _cfg: {"status": "ok"}
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

    assert client.post("/run", json=payload, headers={API_KEY_HEADER: "legacy-key"}).status_code == 200
    assert client.post("/run", json=payload, headers={API_KEY_HEADER: "shared-key"}).status_code == 200
