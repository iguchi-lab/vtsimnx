"""Reproduce specification examples; no HTTP or C++ solver execution."""
from __future__ import annotations
import copy
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "engine")]
sys.dont_write_bytecode = True

def main():
    results = []
    def passed(name, method, details):
        results.append({"id": name, "method": method, "status": "passed", "details": details})
    def close(actual, expected):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)
    def rejects(fn, exc):
        try:
            fn()
        except exc:
            return
        raise AssertionError("Expected rejection")

    with tempfile.TemporaryDirectory(prefix="vtsimnx-spec-") as temp, ExitStack() as cleanup:
        cleanup.callback(logging.shutdown)
        os.environ["VTSIMNX_BUILDER_LOG_FILE"] = str(Path(temp) / "builder.log")
        from app.builder import build_config
        from app.builder.build_options import BuildOptions
        from app.builder.utils import ensure_timeseries
        from vtsimnx.utils.jsonable import to_jsonable
        from vtsimnx.run_calc._index import _normalize_simulation_index_inplace
        from vtsimnx.artifacts._decode import decode_f32_series
        from vtsimnx.artifacts.errors import ArtifactDecodeError
        import numpy as np
        import pandas as pd

        raw = {
            "simulation": {
                "index": {"start": "2026-01-01 00:00:00", "end": "2026-01-01 01:00:00",
                          "timestep": 3600, "length": 2},
                "coupling": {"moisture_enabled": False},
                "iteration": {"max_inner": 3},
                "calc_flag": {"p": True, "t": False, "x": False, "c": False},
            },
            "nodes": [
                {"key": "room", "t": 20.0, "x": 0.01, "v": 50,
                 "calc_t": True, "thermal_mass": 120720,
                 "moisture_capacity": 2500000},
                {"key": "outside", "t": 0.0, "x": 0.005},
            ],
            "ventilation_branches": [
                {"key": "outside->room", "vol": 0.1},
                {"key": "room->outside", "vol": 0.1},
            ],
            "thermal_branches": [{"key": "outside->room", "conductance": 10.0}],
        }
        before = copy.deepcopy(raw)
        built = build_config(raw)
        assert raw == before
        sim = built["simulation"]
        assert "coupling" not in sim and "iteration" not in sim
        assert sim["calc_flag"] == {"p": False, "t": True, "x": True, "c": False}
        passed("SPEC-03-002", "actual_builder", "simulation extras not propagated; flags derived; raw unchanged")
        nodes = {n["key"]: n for n in built["nodes"]}
        air = next(b for b in built["thermal_branches"] if b.get("subtype") == "air_capacity")
        furniture = next(b for b in built["thermal_branches"] if b.get("subtype") == "capacity")
        close(air["conductance"], 60360 / 3600)
        close(furniture["conductance"], 60360 / 3600)
        assert "thermal_mass" not in nodes["room"]
        bad = copy.deepcopy(raw)
        bad["nodes"][0]["thermal_mass"] = 1
        rejects(lambda: build_config(bad), ValueError)
        passed("SPEC-04-002", "actual_builder", "air/furniture conductance=16.7666666667 W/K; insufficient capacity rejected")

        close(nodes["room_mx"]["moisture_capacity"], 1)
        moisture = next(b for b in built["thermal_branches"] if b.get("subtype") == "moisture_capacity")
        close(moisture["moisture_conductance"], 1 / 3600)
        disabled = build_config(raw, add_moisture_capacity=False)
        assert not any(n["key"] == "room_mx" for n in disabled["nodes"])
        assert all("moisture_capacity" not in n for n in disabled["nodes"])
        passed("SPEC-04-003", "actual_builder", "capacity conversion, link, and disabling")

        minimal = {"simulation": {"index": raw["simulation"]["index"]},
                   "nodes": [{"key": "room"}]}
        empty = build_config(minimal)
        assert empty["ventilation_branches"] == [] and empty["thermal_branches"] == []
        passed("SPEC-03-001", "actual_builder", "omitted branch arrays become empty")

        assert ensure_timeseries([2.0], 2) == [2.0, 2.0]
        rejects(lambda: ensure_timeseries([1, 2, 3], 2), ValueError)
        passed("SPEC-03-003", "actual_helper", "length-one broadcast and length mismatch")

        option = BuildOptions.resolve({"builder": {"add_surface": False}, "add_surface": True})
        assert option.add_surface is False
        assert BuildOptions.resolve({"builder": {"add_surface": False}}, add_surface=True).add_surface is True
        rejects(lambda: BuildOptions.resolve({"builder": {"response_terms": True}}), ValueError)
        passed("SPEC-04-OPTIONS", "actual_helper", "argument > builder > top-level; bool terms rejected")

        converted = to_jsonable({"series": pd.Series([1, 2]),
                                 "frame": pd.DataFrame({"a": [1, 2]}),
                                 "nan": float("nan"), "inf": float("inf")})
        assert converted == {"series": [1, 2], "frame": {"a": [1, 2]}, "nan": None, "inf": None}
        passed("SPEC-02-001", "actual_helper", "Series, DataFrame and non-finite floats")

        cfg = {"simulation": {"index": pd.date_range("2026-01-01", periods=2, freq="h", tz="Asia/Tokyo").as_unit("ns")}}
        _normalize_simulation_index_inplace(cfg)
        assert cfg["simulation"]["index"]["start"] == "2025-12-31 15:00:00"
        assert cfg["simulation"]["index"]["timestep"] == 3600
        irregular = {"simulation": {"index": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 03:00"])}}
        rejects(lambda: _normalize_simulation_index_inplace(irregular), ValueError)
        passed("SPEC-02-002", "actual_helper", "timezone conversion and irregular spacing rejection")

        precision_observations = {}
        for unit in ["ns", "us"]:
            idx = pd.date_range("2026-01-01", periods=2, freq="h").as_unit(unit)
            sample = {"simulation": {"index": idx}}
            _normalize_simulation_index_inplace(sample)
            actual = sample["simulation"]["index"]["timestep"]
            assert actual == {"ns": 3600, "us": 4}[unit]
            precision_observations[unit] = actual
        passed("SPEC-02-PRECISION", "actual_helper", precision_observations)

        schema = {"dtype": "f32le", "layout": "timestep-major", "length": 2,
                  "series": {"aircon_power": {"keys": ["a", "b"]}}}
        binary = np.array([1, 2, 3, 4], dtype="<f4").tobytes()
        df = decode_f32_series(binary, schema, "aircon_power")
        assert df.values.tolist() == [[1, 2], [3, 4]]
        assert df.attrs["unit"] == "W"
        rejects(lambda: decode_f32_series(binary[:-4], schema, "aircon_power"), ArtifactDecodeError)
        schema["series"]["aircon_power"]["keys"] = []
        assert decode_f32_series(b"", schema, "aircon_power").shape == (2, 0)
        passed("SPEC-13-001-002", "actual_decoder", "timestep-major, byte count, units, zero columns")

        x = (0.01 + 0.06 * 0.005) / 1.06
        close(x, 0.00971698113207547)
        passed("SPEC-09-EXAMPLE", "arithmetic_only", {"x": x})
        c = 100 * math.exp(-0.001 * 60)
        close(c, 94.17645335842487)
        passed("SPEC-10-EXAMPLE", "arithmetic_only", {"c": c})
        hrv = 1.2 * 1006 * 0.1 * 14
        close(hrv, 1690.08)
        passed("SPEC-11-EXAMPLE", "arithmetic_only", {"hrv_W_when_exhaust_zero": hrv})
        energy = sum(p * 1800 / 3600000 for p in [1000, 1000])
        close(energy, 1)
        passed("SPEC-12-EXAMPLE", "arithmetic_only", {"energy_kWh": energy})

        # Publish all outputs only after every check passes.
        for filename, data in [("raw_example.json", raw), ("solver_example.json", built)]:
            (HERE / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "target_commit": "07a2181cec92029f841c4be63fe5c0932ba95ece",
            "cpp_solver_executed": False,
            "http_executed": False,
            "checks": results,
        }
        (HERE / "verification_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Close the temporary builder FileHandler before removing its directory on Windows.
        for handler in list(logging.getLogger("vtsim_config").handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logging.getLogger("vtsim_config").removeHandler(handler)
        print(json.dumps({"checks_passed": len(results), "actual_implementation_checks": sum(r["method"] != "arithmetic_only" for r in results), "arithmetic_checks": sum(r["method"] == "arithmetic_only" for r in results)}))

if __name__ == "__main__":
    main()
