"""Pressure scaling benchmark with an independent, known nonlinear solution.

This synthetic room chain is not the standard house or the Issue #33 input.
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


def pressure_case(count, hours, conductance=0.005):
    phase = 2 * np.pi * np.arange(hours) / 24
    expected = -10 - np.sin(phase[:, None] + np.arange(count)[None, :] * 0.1)
    flow = conductance * np.sqrt(-expected)
    branches = []
    for i in range(count):
        branches.append({"key": f"out->r{i}", "type": "gap", "a": conductance, "n": 2.0})
        if i:
            dp = expected[:, i-1] - expected[:, i]
            q = conductance * np.sign(dp) * np.sqrt(abs(dp))
            flow[:, i-1] -= q
            flow[:, i] += q
            branches.append({"key": f"r{i-1}->r{i}", "type": "gap", "a": conductance, "n": 2.0})
    for i in range(count):
        branches.append({"key": f"r{i}->out", "vol": flow[:, i].tolist()})
    start = datetime(2026, 1, 1)
    raw = {
        "simulation": {
            "index": {"start": start.isoformat(), "end": (start + timedelta(hours=hours-1)).isoformat(),
                      "timestep": 3600, "length": hours},
            "calc_flag": {"p": True, "t": False, "x": False, "c": False},
            "tolerance": {"ventilation": 1e-8},
        },
        "nodes": [{"key": "out", "p": 0.0, "t": 20.0, "calc_p": False}] + [
            {"key": f"r{i}", "p": -1.0, "t": 20.0, "calc_p": True, "v": 40.0}
            for i in range(count)],
        "ventilation_branches": branches,
        "thermal_branches": [],
    }
    return raw, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[17, 64, 128, 256])
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.sizes, args.hours, args.repeats) <= 0:
        parser.error("sizes, hours and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sr.SOLVER_EXE = args.solver.resolve()
    sr.BASE_DIR = args.output.resolve()
    os.environ["VTSIMNX_TIMINGS"] = "1"
    report = {"model": "synthetic_pressure_chain", "solver_sha256": hashlib.sha256(
        sr.SOLVER_EXE.read_bytes()).hexdigest(), "hours": args.hours, "cases": []}
    for count in args.sizes:
        raw, expected = pressure_case(count, args.hours)
        built = build_config(raw, output_path=None, add_surface=False, add_capacity=False,
                             add_moisture_capacity=False)
        built["simulation"].setdefault("log", {})["verbosity"] = 0
        input_hash = hashlib.sha256(json.dumps(built, sort_keys=True).encode()).hexdigest()
        for repeat in range(args.repeats):
            start = time.perf_counter()
            out = sr.run_solver(built, run_id=f"pressure_{count}_{repeat}")
            row = {"nodes": count, "repeat": repeat, "input_sha256": input_hash,
                   "seconds": time.perf_counter()-start, "status": out.get("status"),
                   "metrics": out.get("metrics"), "error": out.get("error")}
            if out.get("status") == "ok":
                art = sr.resolve_artifact_path(out["artifact_dir"])
                keys, rows = read_series_matrix(art, out, "vent_pressure")
                p = np.asarray(rows)[:, [keys.index(f"r{i}") for i in range(count)]]
                row["max_pressure_error_pa"] = float(np.max(abs(p-expected)))
                # Independently recompute balance from the float32 artifact flows.
                keys, rows = read_series_matrix(art, out, "vent_flow_rate")
                balance = np.zeros((args.hours, count))
                for j, key in enumerate(keys):
                    source, target = key.split("->")
                    if source.startswith("r"):
                        balance[:, int(source[1:])] -= np.asarray(rows)[:, j]
                    if target.startswith("r"):
                        balance[:, int(target[1:])] += np.asarray(rows)[:, j]
                row["max_artifact_balance_m3_s"] = float(np.max(abs(balance)))
                row["accuracy_ok"] = (out["metrics"]["solve_timesteps"] == args.hours
                    and row["max_pressure_error_pa"] < 0.0001
                    and row["max_artifact_balance_m3_s"] < 2e-8)
            report["cases"].append(row)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    return 0 if all(r["status"] == "ok" and r.get("accuracy_ok") for r in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
