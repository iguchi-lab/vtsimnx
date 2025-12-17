import pytest

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

    payload = {"config": {"param1": 123, "mode": "fast"}}
    resp = client.post("/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert isinstance(body["result"], dict)
    assert body["result"].get("status") == "ok"


