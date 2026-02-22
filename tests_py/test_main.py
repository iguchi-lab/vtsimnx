import pytest
import gzip
import json

# 既存のAPIテストは starlette.testclient -> httpx に依存する。
# 軽量環境で httpx が入っていない場合は collection error になるため、明示的に skip する。
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_returns_fixed_response():
    # /run は内部で C++ solver を起動するため、そのままだと環境/入力に依存して不安定になる。
    # API層のテストでは solver をモックして「200でJSONが返ること」を検証する。
    import app.main as main_mod
    main_mod.run_solver = lambda _cfg: {"status": "ok"}  # monkeypatch 相当（依存を増やさない）

    # /run は raw JSON を builder で正規化/展開してから solver に渡す。
    # builder が通る最小構成の raw config を渡す。
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
    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert isinstance(body["result"], dict)
    assert body["result"].get("status") == "ok"


def test_run_accepts_gzip_body():
    import app.main as main_mod
    main_mod.run_solver = lambda _cfg: {"status": "ok"}  # mock

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
    raw = json.dumps(payload).encode("utf-8")
    gz = gzip.compress(raw)

    resp = client.post(
        "/run",
        content=gz,
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"].get("status") == "ok"


def test_run_accepts_gzip_body_with_multi_encoding_header():
    import app.main as main_mod
    main_mod.run_solver = lambda _cfg: {"status": "ok"}  # mock

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
    raw = json.dumps(payload).encode("utf-8")
    gz = gzip.compress(raw)

    resp = client.post(
        "/run",
        content=gz,
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip, something"},
    )
    assert resp.status_code == 200


def test_run_returns_warning_details_for_unknown_fields():
    # builder が未知フィールドを削除した場合、warnings（文字列）と warning_details（構造化）を返す
    import app.main as main_mod
    main_mod.run_solver = lambda _cfg: {"status": "ok"}  # mock

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
            "nodes": [{"key": "N1", "unknown_field": 123}],
            "ventilation_branches": [],
            "thermal_branches": [],
        }
    }
    resp = client.post("/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert isinstance(body.get("warnings"), list)
    assert isinstance(body.get("warning_details"), list)
    assert any(d.get("code") == "unknown_field_stripped" and d.get("field") == "unknown_field"
               for d in body["warning_details"])


def test_run_returns_structured_400_on_validation_error():
    # builder 側で入力不正が起きた場合、APIは code/message/hint を含む 400 を返す
    import app.main as main_mod

    original = main_mod.build_config_with_warning_details

    def _raise_validation_error(*_args, **_kwargs):
        raise main_mod.ValidationError("熱ブランチ A->B の'target'のノード 'B' が存在しません")

    main_mod.build_config_with_warning_details = _raise_validation_error
    try:
        resp = client.post("/run", json={"config": {}})
    finally:
        main_mod.build_config_with_warning_details = original

    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "invalid_config"
    assert "存在しません" in body["detail"]["message"]
    assert "nodes" in body["detail"]["hint"]
