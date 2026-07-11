import json

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app.solver_runner as sr  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # resolve_artifact_path は solver_runner.BASE_DIR/work を見る
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    monkeypatch.delenv("VTSIMNX_API_KEY", raising=False)
    monkeypatch.delenv("VTSIMNX_API_KEYS", raising=False)
    return TestClient(app)


def test_work_static_mount_is_removed(client):
    resp = client.get("/work/")
    assert resp.status_code == 404


def test_artifacts_download_uses_manifest_whitelist(client, tmp_path):
    artifact_dir = "artifacts.demo"
    art = tmp_path / "work" / artifact_dir
    art.mkdir(parents=True)
    (art / "solver.log").write_text("ok\n", encoding="utf-8")
    (art / "secret.txt").write_text("should not be readable\n", encoding="utf-8")
    manifest = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "output": {
            "artifact_dir": artifact_dir,
            "log_file": "solver.log",
            "result_files": {"schema": "schema.json"},
        },
        "result_files": {"schema": "schema.json"},
        "files": {
            "schema": "schema.json",
            "log": "solver.log",
            "manifest": "manifest.json",
        },
    }
    (art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (art / "schema.json").write_text('{"dtype":"f32le"}', encoding="utf-8")

    files_resp = client.get(f"/artifacts/{artifact_dir}/files")
    assert files_resp.status_code == 200
    keys = files_resp.json()["keys"]
    assert "log" in keys
    assert "schema" in keys
    assert "manifest" in keys

    log_resp = client.get(f"/artifacts/{artifact_dir}/download/log")
    assert log_resp.status_code == 200
    assert log_resp.text == "ok\n"

    # ホワイトリスト外の直接パスは download API 経由でも拒否
    bad = client.get(f"/artifacts/{artifact_dir}/download/secret")
    assert bad.status_code == 404

    # 旧 /work 静的公開では取れてしまったパスも 404
    assert client.get(f"/work/{artifact_dir}/secret.txt").status_code == 404
    assert client.get(f"/work/{artifact_dir}/solver.log").status_code == 404
