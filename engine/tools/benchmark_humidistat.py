"""Deterministic two-room cooling-year benchmark (synthetic weather, not a house report)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "engine")]
from app.builder import build_config  # noqa: E402
from app import solver_runner as sr  # noqa: E402
from tests_py.test_room_humidity_control import connected_two_ac_case  # noqa: E402
from tests_py.physics.helpers import read_series  # noqa: E402
from vtsimnx.archenv.archenv import humidity_ratio_from_rh  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=8760)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sr.SOLVER_EXE = args.solver.resolve()
    sr.BASE_DIR = args.output.resolve()
    report = {"solver_sha256": hashlib.sha256(sr.SOLVER_EXE.read_bytes()).hexdigest(),
              "hours": args.hours, "weather": "synthetic cooling year", "cases": []}
    hours = np.arange(args.hours)
    outside_t = 28 + np.sin(2*np.pi*hours/8760) + np.sin(2*np.pi*hours/24)
    outside_rh = 75 + 5*np.sin(2*np.pi*hours/8760 + .4)
    for mode in ("humidity_priority", "temperature_humidity"):
        raw = connected_two_ac_case(mode, count=args.hours)
        raw["simulation"]["index"].update(start="2025-01-01T00:00:00", end="2025-12-31T23:00:00")
        raw["nodes"][0].update(t=outside_t.tolist(), x=humidity_ratio_from_rh(outside_t, outside_rh).tolist())
        for unit in raw["aircon"]:
            unit["max_overcooling"] = 10.0  # also accepted by the baseline parser
        built = build_config(raw, output_path=None, add_surface=False, add_capacity=False,
                             add_moisture_capacity=False)
        built["simulation"].setdefault("log", {})["verbosity"] = 0
        input_hash = hashlib.sha256(json.dumps(built, sort_keys=True).encode()).hexdigest()
        start = time.perf_counter()
        out = sr.run_solver(built, run_id=mode)
        row = {"mode": mode, "input_sha256": input_hash, "seconds": time.perf_counter()-start,
               "status": out.get("status"), "metrics": out.get("metrics"), "error": out.get("error")}
        if out.get("status") == "ok":
            art = sr.resolve_artifact_path(out["artifact_dir"])
            def series(name, key):
                return np.asarray(read_series(art, out, name, key))
            row["units"] = []
            for unit, room, target in (("AC", "room", 60.0), ("AC2", "room2", 58.0)):
                supply_t = series("thermal_temperature", unit)
                room_t = series("thermal_temperature", room)
                load = series("aircon_sensible_heat", unit)+series("aircon_latent_heat", unit)
                free = (supply_t > 5.02) & (room_t > 16.02) & (load < 19999.0)
                if mode == "temperature_humidity":
                    free &= abs(room_t-26.0) < .01
                    free &= series("aircon_reheat", unit) > 1.0
                else:
                    # At the requested temperature, sensible cooling can force
                    # additional condensation; that is an active upper boundary
                    # on supply humidity, not a free RH-equality operating point.
                    free &= room_t < 25.99
                error = series("aircon_room_rh", unit)-target
                row["units"].append({"key": unit, "free_hours": int(free.sum()),
                    "max_free_rh_error_pp": float(max(abs(error[free]), default=0)),
                    "max_rh_unmet_pp": float(max(0.0, max(error))),
                    "min_outlet_temperature": float(min(supply_t))})
        report["cases"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    return 0 if all(r["status"] == "ok" for r in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
