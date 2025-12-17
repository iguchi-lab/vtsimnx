from pathlib import Path
import pytest

import app.solver_runner as sr


@pytest.mark.skipif(not Path(sr.SOLVER_EXE).exists(), reason="solver binary not found")
def test_run_solver_returns_json(tmp_path, monkeypatch):
    # 汚染を避けるため、一時ディレクトリを BASE_DIR として使う
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

    # solver は simulation/index/tolerance/calc_flag を必須としているため、最小の正しい入力を渡す
    output = sr.run_solver(
        {
            "simulation": {
                "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
                "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
                "calc_flag": {"p": False, "t": False, "x": False, "c": False},
                "log": {"verbosity": 0},
            }
        }
    )

    assert isinstance(output, dict)
    assert output.get("status") == "ok"
    assert "input_length" in output

    # artifacts が作られていること（最低限: log と schema）
    artifact_dir = output.get("artifact_dir")
    assert isinstance(artifact_dir, str) and artifact_dir
    work_dir = tmp_path / "work"
    art = work_dir / artifact_dir
    assert art.exists() and art.is_dir()
    assert (art / output.get("log_file", "solver.log")).exists()
    assert (art / "schema.json").exists()


