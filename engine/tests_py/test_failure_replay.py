import gzip
import hashlib
import json
from pathlib import Path

import pytest
import app.solver_runner as sr


@pytest.mark.parametrize("failure", ["error_result", "crash", "bad_json"])
@pytest.mark.parametrize("cached", [False, True])
def test_failure_input_survives_api_cleanup(tmp_path, monkeypatch, failure, cached):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    monkeypatch.delenv("VTSIMNX_KEEP_RUN_FILES", raising=False)
    if cached:
        monkeypatch.setenv("VTSIMNX_INPUT_CACHE", "1")
    else:
        monkeypatch.delenv("VTSIMNX_INPUT_CACHE", raising=False)
    exe = tmp_path / "solver"
    exe.write_bytes(b"solver version under test")
    monkeypatch.setattr(sr, "SOLVER_EXE", exe)
    seen = {}

    def invoke(input_path: Path, output_path: Path, **kwargs):
        seen["input"] = input_path.read_bytes()
        if failure == "crash":
            raise RuntimeError("original crash")
        output_path.write_text(
            "{" if failure == "bad_json" else
            json.dumps({"status": "error", "error_code": "aircon_max_iterations"}),
            encoding="utf-8")

    monkeypatch.setattr(sr, "_invoke_solver", invoke)
    payload = {"simulation": {"index": {"length": 2}}, "label": "湿度診断"}
    if failure == "error_result":
        sr.run_solver(payload, run_id="failure1", write_manifest=False)
    else:
        with pytest.raises((RuntimeError, json.JSONDecodeError)):
            sr.run_solver(payload, run_id="failure1", write_manifest=False)
    sr.cleanup_run_workdir("failure1", keep_artifacts=True)
    replay = sr.run_workdir("failure1") / "replay"
    assert gzip.decompress((replay / "input.json.gz").read_bytes()) == seen["input"]
    metadata = json.loads((replay / "replay.json").read_text())
    assert metadata["input_sha256"] == hashlib.sha256(seen["input"]).hexdigest()
    assert metadata["solver_sha256"] == hashlib.sha256(exe.read_bytes()).hexdigest()
    assert metadata["run_id"] == "failure1"


def test_success_does_not_create_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    def invoke(input_path, output_path, **kwargs):
        output_path.write_text('{"status":"ok"}')
    monkeypatch.setattr(sr, "_invoke_solver", invoke)
    sr.run_solver({}, run_id="ok1", write_manifest=False)
    assert not (sr.run_workdir("ok1") / "replay").exists()


def test_keep_run_files_survives_api_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    monkeypatch.setenv("VTSIMNX_KEEP_RUN_FILES", "1")
    run = sr.run_workdir("keep1")
    inp = run / "input.keep1.json"
    inp.write_text("{}")
    sr.cleanup_run_workdir("keep1", keep_artifacts=True)
    assert inp.is_file()


def test_replay_failure_does_not_mask_solver_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    def invoke(*args, **kwargs):
        raise RuntimeError("original crash")
    def preserve(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(sr, "_invoke_solver", invoke)
    monkeypatch.setattr(sr, "_preserve_failed_run", preserve)
    with pytest.raises(RuntimeError, match="original crash"):
        sr.run_solver({}, write_manifest=False)


def test_replay_refuses_modified_input_and_solver(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "replay_failure", Path(__file__).resolve().parents[1] / "tools/replay_failure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    exe = tmp_path / "solver"
    exe.write_bytes(b"original")
    (bundle / "input.json.gz").write_bytes(gzip.compress(b"{}"))
    (bundle / "replay.json").write_text(json.dumps({
        "run_id": "original1", "input_sha256": hashlib.sha256(b"{}").hexdigest(),
        "solver_sha256": hashlib.sha256(b"original").hexdigest()}))
    def forbidden(*args, **kwargs):
        pytest.fail("solver must not execute with mismatched input or binary")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    exe.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Solver hash differs"):
        module.replay(bundle, exe, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()
    (bundle / "input.json.gz").write_bytes(gzip.compress(b"tampered"))
    with pytest.raises(ValueError, match="Input hash mismatch"):
        module.replay(bundle, exe, tmp_path / "tampered", True)
