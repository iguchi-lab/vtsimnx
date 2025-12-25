import json
from pathlib import Path
import subprocess

import pytest

import app.solver_runner as sr


def test_invoke_solver_raises_on_nonzero_exit(tmp_path, monkeypatch):
    inp = tmp_path / "input.json"
    out = tmp_path / "output.json"
    inp.write_text("{}", encoding="utf-8")

    class DummyResult:
        returncode = 7
        stdout = "OUT"
        stderr = "ERR"

    def fake_run(*_args, **_kwargs):
        return DummyResult()

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as e:
        sr._invoke_solver(inp, out, cwd=tmp_path)
    assert "solver failed: 7" in str(e.value)
    assert "stdout: OUT" in str(e.value)
    assert "stderr: ERR" in str(e.value)


def test_invoke_solver_raises_if_output_missing(tmp_path, monkeypatch):
    inp = tmp_path / "input.json"
    out = tmp_path / "output.json"
    inp.write_text("{}", encoding="utf-8")

    class DummyResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: DummyResult())

    with pytest.raises(RuntimeError) as e:
        sr._invoke_solver(inp, out, cwd=tmp_path)
    assert "did not produce output file" in str(e.value)


def test_run_solver_writes_input_and_overwrites_output(tmp_path, monkeypatch):
    # BASE_DIR を tmp にして work/ をテスト内に閉じ込める
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

    work = tmp_path / "work"
    work.mkdir(exist_ok=True)

    # 旧仕様では output.json を上書きしていたが、並列実行安全性のため
    # いまは output.<run_id>.json を使う（既存ファイルは触らない）。
    old_output = work / "output.json"
    old_output.write_text('{"old": true}', encoding="utf-8")

    def fake_invoke(input_path: Path, output_path: Path, cwd: Path) -> None:
        assert cwd == work
        # 入力が書かれていること
        data = json.loads(input_path.read_text(encoding="utf-8"))
        assert data["simulation"]["index"]["length"] == 1
        assert input_path.name.startswith("input.") and input_path.name.endswith(".json")
        assert output_path.name.startswith("output.") and output_path.name.endswith(".json")
        # ダミーの solver 出力
        output_path.write_text('{"status":"ok","artifact_dir":"x","log_file":"solver.log","result_files":{}}', encoding="utf-8")

    monkeypatch.setattr(sr, "_invoke_solver", fake_invoke)

    out = sr.run_solver(
        {
            "simulation": {
                "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
                "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
                "calc_flag": {"p": False, "t": False, "x": False, "c": False},
                "log": {"verbosity": 0},
            }
        }
    )

    assert out["status"] == "ok"
    # 既存の output.json は触らない
    assert json.loads(old_output.read_text(encoding="utf-8"))["old"] is True


def test_run_solver_uses_unique_io_paths_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    monkeypatch.delenv("VTSIMNX_KEEP_RUN_FILES", raising=False)

    work = tmp_path / "work"
    work.mkdir(exist_ok=True)

    seen = []

    def fake_invoke(input_path: Path, output_path: Path, cwd: Path) -> None:
        seen.append((input_path, output_path))
        output_path.write_text('{"status":"ok","artifact_dir":"x","log_file":"solver.log","result_files":{}}', encoding="utf-8")

    monkeypatch.setattr(sr, "_invoke_solver", fake_invoke)

    _ = sr.run_solver(
        {
            "simulation": {
                "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
                "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
                "calc_flag": {"p": False, "t": False, "x": False, "c": False},
                "log": {"verbosity": 0},
            }
        }
    )
    _ = sr.run_solver(
        {
            "simulation": {
                "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
                "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
                "calc_flag": {"p": False, "t": False, "x": False, "c": False},
                "log": {"verbosity": 0},
            }
        }
    )

    assert len(seen) == 2
    assert seen[0][0] != seen[1][0]
    assert seen[0][1] != seen[1][1]

    # デフォルトでは一時ファイルは消える
    for inp, outp in seen:
        assert not inp.exists()
        assert not outp.exists()


def test_run_solver_keep_run_files_env_keeps_input_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    monkeypatch.setenv("VTSIMNX_KEEP_RUN_FILES", "1")

    work = tmp_path / "work"
    work.mkdir(exist_ok=True)

    seen = {}

    def fake_invoke(input_path: Path, output_path: Path, cwd: Path) -> None:
        seen["inp"] = input_path
        seen["out"] = output_path
        output_path.write_text('{"status":"ok","artifact_dir":"x","log_file":"solver.log","result_files":{}}', encoding="utf-8")

    monkeypatch.setattr(sr, "_invoke_solver", fake_invoke)

    _ = sr.run_solver(
        {
            "simulation": {
                "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
                "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
                "calc_flag": {"p": False, "t": False, "x": False, "c": False},
                "log": {"verbosity": 0},
            }
        }
    )

    assert seen["inp"].exists()
    assert seen["out"].exists()

