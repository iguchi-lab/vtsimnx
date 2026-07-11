"""Measure builder/solver performance metrics for representative cases."""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import app.solver_runner as sr
from app.builder import build_config

from tests_py.physics.log_metrics import parse_solver_log
from tests_py.physics.tolerances import (
    PERF_WARN_AIRCON_RECOMPUTE_RATIO,
    PERF_WARN_ARTIFACT_RATIO,
    PERF_WARN_LU_RATIO,
    PERF_WARN_MEMORY_RATIO,
    PERF_WARN_TIME_RATIO,
)
from tests_py.perf.cases import BENCH_CASES

BASELINES_PATH = Path(__file__).resolve().parent / "baselines" / "representative.json"


def _timing_sum(timings: list[dict[str, Any]], name: str) -> float:
    return float(sum(float(t.get("duration_ms") or 0.0) for t in timings if t.get("name") == name))


def _artifact_size_bytes(artifact_dir: Path) -> int:
    total = 0
    for p in artifact_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def measure_case(
    *,
    name: str,
    raw_factory,
    build_kwargs: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    sr.BASE_DIR = work_dir
    raw = raw_factory()

    t0 = time.perf_counter()
    cfg = build_config(raw, output_path=None, **build_kwargs)
    builder_ms = (time.perf_counter() - t0) * 1000.0
    cfg.setdefault("simulation", {}).setdefault("log", {})["verbosity"] = 1

    prev = os.environ.get("VTSIMNX_TIMINGS")
    os.environ["VTSIMNX_TIMINGS"] = "1"
    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t1 = time.perf_counter()
    try:
        output = sr.run_solver(cfg, run_id=f"perf_{name}", write_manifest=False)
    finally:
        if prev is None:
            os.environ.pop("VTSIMNX_TIMINGS", None)
        else:
            os.environ["VTSIMNX_TIMINGS"] = prev
    wall_solver_ms = (time.perf_counter() - t1) * 1000.0
    rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Linux: ru_maxrss is KiB (peak for waited children). Prefer absolute peak when
    # this process only ran one solver child (see run_all_benches isolation).
    max_memory_kib = int(rss_after) if int(rss_after) > 0 else max(0, int(rss_after) - int(rss_before))


    assert output.get("status") == "ok", output.get("error") or output
    art = sr.resolve_artifact_path(str(output.get("artifact_dir") or ""))
    assert art is not None and art.is_dir()

    timings = output.get("timings") or []
    log_text = (art / str(output.get("log_file") or "solver.log")).read_text(
        encoding="utf-8", errors="replace"
    )
    log_metrics = parse_solver_log(log_text).to_dict()

    return {
        "case": name,
        "builder_ms": round(builder_ms, 3),
        "solver_wall_ms": round(wall_solver_ms, 3),
        "solver_simulation_total_ms": round(_timing_sum(timings, "simulation_total"), 3),
        "solver_load_input_ms": round(_timing_sum(timings, "load_input"), 3),
        "max_memory_kib": max_memory_kib,
        "lu_factorize": log_metrics.get("lu_factorize"),
        "topo_rebuild": log_metrics.get("topo_rebuild"),
        "pattern_rebuild": log_metrics.get("pattern_rebuild"),
        "aircon_recompute_count": log_metrics.get("aircon_recompute_count"),
        "coupled_iterations_sum": log_metrics.get("coupled_iterations_sum"),
        "artifact_size_bytes": _artifact_size_bytes(art),
        "timings_count": len(timings),
    }


def run_all_benches(work_dir: Path) -> dict[str, Any]:
    """Run each case in a child process so ru_maxrss is not dominated by prior cases."""
    import multiprocessing as mp

    results: list[dict[str, Any]] = []

    def _worker(q: mp.Queue, name: str, factory, kwargs: dict[str, Any], wd: str) -> None:
        try:
            q.put(("ok", measure_case(name=name, raw_factory=factory, build_kwargs=kwargs, work_dir=Path(wd))))
        except Exception as exc:  # noqa: BLE001 - surface to parent
            q.put(("err", f"{type(exc).__name__}: {exc}"))

    for name, factory, kwargs in BENCH_CASES:
        case_dir = work_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        q: mp.Queue = mp.Queue()
        proc = mp.Process(target=_worker, args=(q, name, factory, kwargs, str(case_dir)))
        proc.start()
        proc.join(timeout=300)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            raise TimeoutError(f"perf case timed out: {name}")
        status, payload = q.get()
        if status != "ok":
            raise RuntimeError(f"perf case failed: {name}: {payload}")
        # In child, max_memory_kib is delta from 0 children → absolute child peak.
        results.append(payload)

    return {
        "schema_version": 1,
        "cases": results,
    }


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    """Return human-readable warning strings for large regressions (never fails)."""
    warnings: list[str] = []
    base_by_name = {c["case"]: c for c in baseline.get("cases", [])}
    for cur in current.get("cases", []):
        name = cur["case"]
        base = base_by_name.get(name)
        if not base:
            warnings.append(f"[{name}] no baseline entry (new case)")
            continue

        def _ratio(key: str) -> float | None:
            b = base.get(key)
            c = cur.get(key)
            if b is None or c is None:
                return None
            try:
                b_f = float(b)
                c_f = float(c)
            except (TypeError, ValueError):
                return None
            if b_f <= 0:
                return None
            return c_f / b_f

        checks = [
            ("solver_wall_ms", PERF_WARN_TIME_RATIO, "solver wall time"),
            ("builder_ms", PERF_WARN_TIME_RATIO, "builder time"),
            ("max_memory_kib", PERF_WARN_MEMORY_RATIO, "max memory"),
            ("artifact_size_bytes", PERF_WARN_ARTIFACT_RATIO, "artifact size"),
            ("lu_factorize", PERF_WARN_LU_RATIO, "LU factorize count"),
            ("aircon_recompute_count", PERF_WARN_AIRCON_RECOMPUTE_RATIO, "aircon recompute count"),
        ]
        for key, limit, label in checks:
            r = _ratio(key)
            if r is not None and r >= limit:
                warnings.append(
                    f"[{name}] {label} regression: {base.get(key)} -> {cur.get(key)} "
                    f"(x{r:.2f} >= {limit})"
                )
    return warnings
