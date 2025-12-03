from pathlib import Path
import pytest

import app.solver_runner as sr


@pytest.mark.skipif(not Path(sr.SOLVER_EXE).exists(), reason="solver binary not found")
def test_run_solver_returns_json(tmp_path, monkeypatch):
    # 汚染を避けるため、一時ディレクトリを BASE_DIR として使う
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

    output = sr.run_solver({"foo": "bar", "n": 10})

    assert isinstance(output, dict)
    assert output.get("status") == "ok"
    assert "input_length" in output


