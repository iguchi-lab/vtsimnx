"""Performance history tests: record metrics and warn on large regressions (no hard fail)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.solver_runner as sr

from tests_py.perf.measure import BASELINES_PATH, compare_to_baseline, run_all_benches

requires_solver = pytest.mark.skipif(
    not Path(sr.SOLVER_EXE).exists(),
    reason="solver binary not found",
)


@pytest.mark.perf
@requires_solver
def test_perf_representative_history(tmp_path, monkeypatch):
    """Run representative benches, write metrics JSON, warn vs baseline.

    Strict pass/fail on wall-clock is intentionally avoided; the test fails only
    if the solver itself errors. Regression warnings are printed for CI logs.
    """
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    report = run_all_benches(tmp_path)

    out_dir = Path(__file__).resolve().parent / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    # CI では GITHUB_WORKSPACE 配下の artifact 用パスも書く
    local_path = tmp_path / "perf_report.json"
    local_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # リポジトリ内の最新スナップショット（ローカル確認用; CI では artifact を優先）
    snap = out_dir / "last_local_report.json"
    try:
        snap.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass

    print("\n=== perf report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if BASELINES_PATH.is_file():
        baseline = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
        warnings = compare_to_baseline(report, baseline)
        if warnings:
            print("\n=== perf regression WARNINGS (non-fatal) ===")
            for w in warnings:
                print(f"WARNING: {w}")
        else:
            print("\n=== perf: no large regressions vs baseline ===")
    else:
        print(f"\n=== perf: baseline missing at {BASELINES_PATH} (record-only) ===")

    # Always pass if measurement succeeded.
    assert report["cases"], "no perf cases measured"
    for case in report["cases"]:
        assert case["solver_wall_ms"] >= 0
        assert case["artifact_size_bytes"] > 0
