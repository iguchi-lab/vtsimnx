from __future__ import annotations

from pathlib import Path

import pytest

import app.solver_runner as sr

requires_solver = pytest.mark.skipif(
    not Path(sr.SOLVER_EXE).exists(),
    reason="solver binary not found",
)


@pytest.fixture
def solver_workdir(monkeypatch, tmp_path):
    """Isolate solver work/ under tmp_path for each physics test."""
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    return tmp_path
