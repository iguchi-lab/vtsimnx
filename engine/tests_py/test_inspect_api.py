from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

AI_INPUT_DIR = Path(__file__).resolve().parents[2] / "examples" / "ai_input"


MIN_CFG = {
    "simulation": {
        "index": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T01:00:00Z",
            "timestep": 60,
            "length": 60,
        }
    },
    "nodes": [{"key": "N1", "t": 20.0, "calc_t": True, "v": 30.0, "thermal_mass": 36216.0}],
    "ventilation_branches": [],
    "thermal_branches": [],
}


def test_validate_accepts_minimal_and_returns_issues():
    with TestClient(app) as client:
        resp = client.post("/validate", json={"config": MIN_CFG})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["issues"], list)
    for issue in body["issues"]:
        assert "code" in issue
        assert "message" in issue
        assert "path" in issue
        assert issue["severity"] in ("error", "warning")


def test_validate_unknown_heat_source_key_is_error():
    cfg = {
        **MIN_CFG,
        "heat_source": [{"room": "N1", "generation_rate": 80.0, "typo_watt": 1}],
    }
    with TestClient(app) as client:
        resp = client.post("/validate", json={"config": cfg, "unknown_keys": "error"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    unknown = next(i for i in body["issues"] if i["code"] == "unknown_field")
    assert "typo_watt" in unknown["path"]
    assert "heat_source" in unknown["path"]


def test_validate_warns_on_supply_only_fixed_flow():
    cfg = {
        "simulation": MIN_CFG["simulation"],
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "v": 30.0, "thermal_mass": 36216.0},
            {"key": "outside", "t": 5.0},
        ],
        "ventilation_branches": [
            {"key": "outside->room", "type": "fixed_flow", "vol": 0.01},
        ],
        "thermal_branches": [],
    }
    with TestClient(app) as client:
        resp = client.post("/validate", json={"config": cfg})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert any(i["code"] == "fixed_flow_imbalance" for i in body["issues"])


def test_build_returns_expanded_config_and_stats():
    with TestClient(app) as client:
        resp = client.post("/build", json={"config": MIN_CFG})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["config"]["nodes"]
    assert body["stats"]["nodes"] >= 1
    assert "thermal_branches" in body["stats"]
    assert "ventilation_branches" in body["stats"]


@pytest.mark.parametrize("name", [
    "minimal_room.json",
    "heat_source.json",
    "humidity_source.json",
    "surfaces_wall.json",
    "aircon_rac.json",
])
def test_ai_input_examples_validate_and_build(name):
    import json

    cfg = json.loads((AI_INPUT_DIR / name).read_text(encoding="utf-8"))
    with TestClient(app) as client:
        validated = client.post("/validate", json={"config": cfg, "unknown_keys": "error"})
        built = client.post("/build", json={"config": cfg, "unknown_keys": "error"})
    assert validated.status_code == 200
    assert validated.json()["ok"] is True
    assert all(i["severity"] != "error" for i in validated.json()["issues"])
    assert built.status_code == 200
    body = built.json()
    assert body["ok"] is True
    assert body["stats"]["nodes"] >= 2
    assert body["config"] is not None


def test_build_rejects_unknown_nested_key_when_error_mode():
    cfg = {
        **MIN_CFG,
        "heat_source": [{"room": "N1", "generation_rate": 80.0, "typo_watt": 1}],
    }
    with TestClient(app) as client:
        resp = client.post("/build", json={"config": cfg, "unknown_keys": "error"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["config"] is None
    assert any(i["code"] == "unknown_field" for i in body["issues"])
