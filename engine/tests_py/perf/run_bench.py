#!/usr/bin/env python3
"""CLI entry for CI perf-history job.

Usage (from engine/):
  python -m tests_py.perf.run_bench --output /tmp/perf_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Ensure engine root is on path when run as module or script.
ENGINE_ROOT = Path(__file__).resolve().parents[2]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from tests_py.perf.measure import BASELINES_PATH, compare_to_baseline, run_all_benches  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VTSimNX performance history benches")
    parser.add_argument("--output", type=Path, required=True, help="Write JSON report path")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit 1 when large regressions are detected (default: warn only)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="vtsimnx_perf_") as td:
        report = run_all_benches(Path(td))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    warnings: list[str] = []
    if BASELINES_PATH.is_file():
        baseline = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
        warnings = compare_to_baseline(report, baseline)
        if warnings:
            print("\n=== perf regression WARNINGS ===", file=sys.stderr)
            for w in warnings:
                print(f"WARNING: {w}", file=sys.stderr)
        else:
            print("\n=== perf: no large regressions vs baseline ===", file=sys.stderr)
    else:
        print(f"baseline missing: {BASELINES_PATH}", file=sys.stderr)

    if args.fail_on_warning and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
