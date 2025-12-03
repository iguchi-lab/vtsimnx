from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_returns_fixed_response():
    payload = {"config": {"param1": 123, "mode": "fast"}}
    resp = client.post("/run", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "result" in body
    assert isinstance(body["result"], dict)
    assert body["result"].get("status") == "ok"


