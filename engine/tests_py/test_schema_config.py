"""型付き config スキーマと unknown_keys モードのテスト。"""
import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import RawSimConfig, prepare_raw_config, UnknownFieldError  # noqa: E402


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


def test_openapi_exposes_simulation_index():
    schema = app.openapi()
    components = schema["components"]["schemas"]
    assert "SimulationRequest" in components
    assert "RawSimConfig" in components
    assert "SimulationIndex" in components
    assert "SimulationSection" in components
    idx = components["SimulationIndex"]["properties"]
    for key in ("start", "end", "timestep", "length"):
        assert key in idx
    raw = components["RawSimConfig"]["properties"]
    for key in ("simulation", "nodes", "ventilation_branches", "thermal_branches", "surfaces", "aircon", "builder"):
        assert key in raw
    assert "unknown_keys" in components["SimulationRequest"]["properties"]


def test_prepare_raw_config_strip_unknown_node_field():
    cfg = {
        **MIN_CFG,
        "nodes": [{"key": "N1", "typo_field": 1}],
    }
    data, warnings, details = prepare_raw_config(cfg, unknown_keys="strip")
    assert "typo_field" not in data["nodes"][0]
    assert any("typo_field" in w for w in warnings)
    assert details[0]["code"] == "unknown_field_stripped"


def test_prepare_raw_config_error_on_unknown():
    cfg = {
        **MIN_CFG,
        "nodes": [{"key": "N1", "typo_field": 1}],
    }
    with pytest.raises(UnknownFieldError):
        prepare_raw_config(cfg, unknown_keys="error")


def test_run_strip_unknown_returns_warning():
    import app.main as main_mod
    import app.services.simulation as sim_svc

    sim_svc.run_solver = lambda _cfg, **_kw: {"status": "ok", "artifact_dir": "artifacts.x", "result_files": {}}
    with TestClient(app) as client:
        payload = {
            "config": {
                **MIN_CFG,
                "nodes": [{"key": "N1", "unknown_field": 99}],
            },
            "unknown_keys": "strip",
        }
        resp = client.post("/run", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert any(d.get("code") == "unknown_field_stripped" for d in body.get("warning_details", []))


def test_run_error_unknown_returns_422():
    import app.main as main_mod
    import app.services.simulation as sim_svc

    sim_svc.run_solver = lambda _cfg, **_kw: {"status": "ok"}
    with TestClient(app) as client:
        payload = {
            "config": {
                **MIN_CFG,
                "nodes": [{"key": "N1", "unknown_field": 99}],
            },
            "unknown_keys": "error",
        }
        resp = client.post("/run", json=payload)
        assert resp.status_code == 422
        err = resp.json()["error"]
        assert err["code"] in ("unknown_field", "validation_error")
        assert "details" in err or err["code"] == "unknown_field"


def test_raw_sim_config_accepts_minimal():
    model = RawSimConfig.model_validate(MIN_CFG)
    assert model.simulation.index.length == 60
    assert model.nodes[0].key == "N1"


def test_prepare_raw_config_keeps_surface_builder_fields():
    """API strip 経路でも surface の builder 受理フィールドが落ちないこと。"""
    cfg = {
        **MIN_CFG,
        "nodes": [{"key": "室内", "t": 20.0}, {"key": "外部", "t": 0.0}],
        "surfaces": [
            {
                "key": "室内->外部",
                "part": "wall",
                "area": 10.0,
                "u_value": 0.5,
                "nocturnal": [1.0, 2.0],
                "night_radiation": [3.0, 4.0],
                "comment": "wall note",
                "a_capacity": 1000.0,
                "response_method": "arx_rc",
                "response_terms": 5,
                "SCC": 0.1,
                "SCR": 0.8,
                "epsilon": 0.9,
            }
        ],
    }
    data, warnings, _details = prepare_raw_config(cfg, unknown_keys="strip")
    surface = data["surfaces"][0]
    assert surface["nocturnal"] == [1.0, 2.0]
    assert surface["night_radiation"] == [3.0, 4.0]
    assert surface["comment"] == "wall note"
    assert surface["a_capacity"] == 1000.0
    assert surface["response_method"] == "arx_rc"
    assert surface["response_terms"] == 5
    assert surface["SCC"] == 0.1
    assert surface["SCR"] == 0.8
    assert not any("nocturnal" in w for w in warnings)


def test_prepare_then_build_keeps_nocturnal_branch():
    """prepare_raw_config → build_config 経由でも nocturnal ブランチが生成される。"""
    from app.builder import build_config

    cfg = {
        **MIN_CFG,
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T00:02:00Z",
                "timestep": 60,
                "length": 2,
            }
        },
        "nodes": [{"key": "室内", "t": 20.0}, {"key": "外部", "t": 0.0}],
        "surfaces": [
            {
                "key": "室内->外部",
                "part": "wall",
                "area": 2.0,
                "u_value": 1.0,
                "epsilon": 0.9,
                "nocturnal": [10.0, 20.0],
            }
        ],
    }
    prepared, _, _ = prepare_raw_config(cfg, unknown_keys="strip")
    assert "nocturnal" in prepared["surfaces"][0]
    out = build_config(prepared, add_aircon=False, add_capacity=False)
    assert any(b.get("subtype") == "nocturnal_loss" for b in out.get("thermal_branches", []))
