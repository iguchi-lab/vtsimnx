"""Replay the standard-house base example (four room air conditioners).

The distributed base has prescribed ventilation, without the window/door gaps
added in Issue #33. Record model identity separately from numerical variations.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "engine")]
from app.builder import build_config  # noqa: E402
from app import solver_runner as sr  # noqa: E402
from tests_py.physics.helpers import read_series_matrix  # noqa: E402

SOURCE = ROOT / "engine/app/test_json/vs_simheat_8760.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--baseline-solver", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=8760)
    args = parser.parse_args()
    if not 1 <= args.hours <= 8760:
        parser.error("hours must be between 1 and 8760")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(raw["aircon"]) != 4:
        raise ValueError("standard_house_ac4 must contain four air conditioners")
    source_length = raw["simulation"]["index"]["length"]

    def truncate(value):
        if isinstance(value, dict):
            return {k: truncate(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) == source_length and all(not isinstance(x, (dict, list)) for x in value):
                return value[:args.hours]
            return [truncate(v) for v in value]
        return value

    raw = truncate(raw)
    index = raw["simulation"]["index"]
    index["length"] = args.hours
    index["end"] = (datetime.fromisoformat(index["start"]) + timedelta(hours=args.hours-1)).isoformat()
    built = build_config(raw, output_path=None)
    built["simulation"].setdefault("log", {})["verbosity"] = 0
    report = {
        "case_id": "standard_house_ac4",
        "case_name": "標準住宅・基本ケース（AC4台）",
        "source_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(json.dumps(built, sort_keys=True).encode()).hexdigest(),
        "aircon_rooms": [a["set"] for a in raw["aircon"]],
        "hours": args.hours,
        "ventilation": "prescribed flows; no added window/door gaps",
        "issue33_original_replay": False,
        "runs": [],
    }
    os.environ["VTSIMNX_TIMINGS"] = "1"
    sr.BASE_DIR = args.output.resolve()
    reference = None
    solvers = [("baseline", args.baseline_solver)] if args.baseline_solver else []
    solvers.append(("candidate", args.solver))
    for name, executable in solvers:
        sr.SOLVER_EXE = executable.resolve()
        start = time.perf_counter()
        out = sr.run_solver(built, run_id=f"standard_house_ac4_{name}")
        row = {"name": name, "solver_sha256": hashlib.sha256(sr.SOLVER_EXE.read_bytes()).hexdigest(),
               "seconds": time.perf_counter()-start, "status": out.get("status"),
               "metrics": out.get("metrics"), "error": out.get("error")}
        if out.get("status") == "ok":
            art = sr.resolve_artifact_path(out["artifact_dir"])
            values = {}
            for quantity in ("thermal_temperature", "vent_flow_rate", "aircon_sensible_heat"):
                keys, rows = read_series_matrix(art, out, quantity)
                values[quantity] = (keys, np.asarray(rows))
            row["accuracy_ok"] = (out["metrics"]["solve_timesteps"] == args.hours
                                  and all(np.isfinite(v).all() for _, v in values.values()))
            if reference is not None:
                differences = {}
                for quantity, (keys, data) in values.items():
                    old_keys, old = reference[quantity]
                    if keys != old_keys or data.shape != old.shape:
                        raise ValueError(f"Result schema changed: {quantity}")
                    differences[quantity] = float(np.max(abs(data-old), initial=0))
                row["max_abs_difference"] = differences
                row["accuracy_ok"] = row["accuracy_ok"] and all(v <= 1e-5 for v in differences.values())
            reference = values
        report["runs"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    return 0 if all(r["status"] == "ok" and r.get("accuracy_ok") for r in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
