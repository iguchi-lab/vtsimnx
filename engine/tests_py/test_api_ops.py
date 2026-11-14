import json

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.api_auth import match_api_key, load_api_key_records  # noqa: E402
from app.main import app  # noqa: E402
from app.services.artifact_policy import (  # noqa: E402
    ArtifactPolicy,
    LocalArtifactStore,
    cleanup_artifacts,
    write_owner_metadata,
)


def test_health_live_and_version_without_auth(monkeypatch):
    monkeypatch.setenv("VTSIMNX_API_KEY", "secret-token")
    client = TestClient(app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/ping").status_code == 200
    ver = client.get("/version")
    assert ver.status_code == 200
    body = ver.json()
    assert "api_version" in body
    assert "schema_format_version" in body
    from app.versioning import get_package_version

    assert body["api_version"] == get_package_version()
    assert body["schema_format_version"] == 5


def test_schema_fields_expose_units():
    from app.schemas.config import NodeModel, ThermalBranchModel, VentilationBranchModel

    node_props = NodeModel.model_json_schema()["properties"]
    assert node_props["t"]["unit"] == "degC"
    assert node_props["p"]["unit"] == "Pa"
    vent_props = VentilationBranchModel.model_json_schema()["properties"]
    assert vent_props["vol"]["unit"] == "m3/s"
    thermal_props = ThermalBranchModel.model_json_schema()["properties"]
    assert thermal_props["heat_generation"]["unit"] == "W"
    assert thermal_props["conductance"]["unit"] == "W/K"


def test_health_ready_shape(monkeypatch):
    monkeypatch.delenv("VTSIMNX_API_KEY", raising=False)
    monkeypatch.delenv("VTSIMNX_API_KEYS", raising=False)
    monkeypatch.delenv("VTSIMNX_API_KEYS_JSON", raising=False)
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert "status" in body
    assert "checks" in body
    assert "solver_binary" in body["checks"]
    assert "work_dir" in body["checks"]


def test_auth_error_is_structured(monkeypatch):
    monkeypatch.setenv("VTSIMNX_API_KEY", "secret-token")
    client = TestClient(app)
    resp = client.post("/run", json={"config": {}})
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("endpoint", ["/validate", "/build", "/run", "/runs"])
@pytest.mark.parametrize("field,value", [("latent_method", "misspelled"), ("bf", 1.0), ("Af", 0.0)])
def test_model_validator_errors_return_serializable_422(monkeypatch, endpoint, field, value):
    for name in ("VTSIMNX_API_KEY", "VTSIMNX_API_KEYS", "VTSIMNX_API_KEYS_JSON"):
        monkeypatch.delenv(name, raising=False)
    config = {
        "simulation": {"index": {"start": "2026-01-01T00:00:00", "end": "2026-01-01T00:00:00",
                                  "timestep": 3600, "length": 1}},
        "nodes": [{"key": "room"}, {"key": "outside"}],
        "aircon": [{"key": "AC", "set": "room", "outside": "outside", "model": "RAC",
                    "mode": "COOLING", "pre_temp": 26.0, "ac_spec": {field: value}}],
    }
    # Pydantic includes the actual ValueError in ctx.error. Exercise the HTTP
    # response, including serialization, rather than only the input validator.
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(endpoint, json={"config": config})
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["path"] == ["body", "config"]
    assert field in error["message"]
    detail = error["details"][0]
    assert detail["type"] == "value_error"
    assert isinstance(detail["ctx"]["error"], str)
    assert field in detail["ctx"]["error"]


def test_constant_time_match_and_revoked(monkeypatch):
    monkeypatch.delenv("VTSIMNX_API_KEY", raising=False)
    monkeypatch.delenv("VTSIMNX_API_KEYS", raising=False)
    monkeypatch.setenv(
        "VTSIMNX_API_KEYS_JSON",
        json.dumps(
            [
                {"id": "ops", "key": "ops-secret", "revoked": False},
                {"id": "old", "key": "old-secret", "revoked": True},
            ]
        ),
    )
    recs = load_api_key_records()
    assert match_api_key("ops-secret", recs).key_id == "ops"
    assert match_api_key("old-secret", recs) is None
    assert match_api_key("nope", recs) is None


def test_artifact_cleanup_ttl(tmp_path, monkeypatch):
    work = tmp_path / "work"
    store = LocalArtifactStore(work)
    art = work / "artifacts.old"
    art.mkdir(parents=True)
    (art / "schema.json").write_text("{}", encoding="utf-8")
    # make it old
    import os
    import time

    old = time.time() - 10_000
    os.utime(art, (old, old))
    policy = ArtifactPolicy(ttl_sec=60, max_bytes_per_run=0, max_total_bytes=0)
    stats = cleanup_artifacts(store, policy=policy)
    assert stats["deleted_ttl"] >= 1
    assert not art.exists()


def test_owner_metadata_roundtrip(tmp_path):
    art = tmp_path / "artifacts.x"
    art.mkdir()
    write_owner_metadata(art, key_id="ops", run_id="abc")
    from app.services.artifact_policy import read_owner_key_id

    assert read_owner_key_id(art) == "ops"
