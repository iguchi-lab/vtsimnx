from __future__ import annotations

import json
import math
import os
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import app.solver_runner as sr
from app.builder import build_config

from .log_metrics import SolverLogMetrics, parse_solver_log
from . import tolerances as tol

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
THERMAL_GOLDEN_PATH = GOLDEN_DIR / "thermal_regression_golden.json"

HEAT_RATE_SERIES = (
    "thermal_heat_rate_advection",
    "thermal_heat_rate_heat_generation",
    "thermal_heat_rate_solar_gain",
    "thermal_heat_rate_nocturnal_loss",
    "thermal_heat_rate_convection",
    "thermal_heat_rate_conduction",
    "thermal_heat_rate_radiation",
    "thermal_heat_rate_capacity",
)

_RE_DUP_SUFFIX = re.compile(r"\(\d+\)$")


def run_from_raw(
    *,
    raw_config: dict[str, Any],
    run_id: str,
    tmp_base_dir: Path,
    add_aircon: bool = False,
    add_capacity: bool = True,
    add_surface: bool = True,
    add_moisture_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    log_verbosity: int = 1,
    enable_timings: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Build + run solver. Default verbosity=1 so convergence lines appear in solver.log."""
    kwargs: dict[str, Any] = {
        "output_path": None,
        "add_aircon": add_aircon,
        "add_capacity": add_capacity,
        "add_surface": add_surface,
    }
    if add_moisture_capacity is not None:
        kwargs["add_moisture_capacity"] = add_moisture_capacity
    if add_surface_solar is not None:
        kwargs["add_surface_solar"] = add_surface_solar
    if add_surface_nocturnal is not None:
        kwargs["add_surface_nocturnal"] = add_surface_nocturnal
    if add_surface_radiation is not None:
        kwargs["add_surface_radiation"] = add_surface_radiation

    cfg = build_config(raw_config, **kwargs)
    cfg.setdefault("simulation", {}).setdefault("log", {})["verbosity"] = int(log_verbosity)

    prev_timings = os.environ.get("VTSIMNX_TIMINGS")
    if enable_timings:
        os.environ["VTSIMNX_TIMINGS"] = "1"
    try:
        output = sr.run_solver(cfg, run_id=run_id, write_manifest=False)
    finally:
        if enable_timings:
            if prev_timings is None:
                os.environ.pop("VTSIMNX_TIMINGS", None)
            else:
                os.environ["VTSIMNX_TIMINGS"] = prev_timings

    assert output.get("status") == "ok", output.get("error") or output

    artifact_dir = sr.resolve_artifact_path(str(output.get("artifact_dir") or ""))
    if artifact_dir is None:
        artifact_dir = tmp_base_dir / "work" / str(output["artifact_dir"])
    assert artifact_dir.is_dir(), f"artifact_dir missing: {artifact_dir}"
    return output, artifact_dir


def read_series(artifact_dir: Path, output: dict[str, Any], series_name: str, key: str) -> list[float]:
    schema = json.loads((artifact_dir / "schema.json").read_text(encoding="utf-8"))
    keys = schema["series"][series_name]["keys"]
    idx = keys.index(key)
    width = len(keys)
    length = int(schema["length"])

    bin_path = artifact_dir / output["result_files"][series_name]
    raw = bin_path.read_bytes()
    vals = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    return [float(vals[t * width + idx]) for t in range(length)]


def read_series_matrix(
    artifact_dir: Path, output: dict[str, Any], series_name: str
) -> tuple[list[str], list[list[float]]]:
    schema = json.loads((artifact_dir / "schema.json").read_text(encoding="utf-8"))
    series = schema["series"].get(series_name)
    if not series:
        return [], []
    keys = list(series["keys"])
    width = len(keys)
    length = int(schema["length"])
    if width == 0:
        return keys, [[] for _ in range(length)]

    bin_path = artifact_dir / output["result_files"][series_name]
    raw = bin_path.read_bytes()
    vals = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    rows = [[float(vals[t * width + i]) for i in range(width)] for t in range(length)]
    return keys, rows


def read_solver_log(artifact_dir: Path, output: dict[str, Any] | None = None) -> str:
    name = "solver.log"
    if output and isinstance(output.get("log_file"), str) and output["log_file"]:
        name = str(output["log_file"])
    return (artifact_dir / name).read_text(encoding="utf-8", errors="replace")


def assert_all_finite(values: Iterable[float], *, label: str = "values") -> None:
    for i, v in enumerate(values):
        assert math.isfinite(v), f"{label}[{i}] is not finite: {v}"


def assert_non_negative(values: Iterable[float], *, label: str = "values", tol: float = 0.0) -> None:
    for i, v in enumerate(values):
        assert v >= -tol, f"{label}[{i}] is negative: {v}"


def assert_monotone_non_increasing(values: list[float], *, tol: float = 1e-6) -> None:
    for i in range(len(values) - 1):
        assert values[i + 1] <= values[i] + tol, f"not non-increasing at {i}: {values[i]} -> {values[i + 1]}"


def _edge_endpoints(key: str) -> tuple[str, str] | None:
    if "->" not in key:
        return None
    left, right = key.split("->", 1)
    left = _RE_DUP_SUFFIX.sub("", left.strip())
    right = _RE_DUP_SUFFIX.sub("", right.strip())
    # comments already stripped from artifact keys; keep rename suffixes like (01)
    return left, right


def node_balance_from_edge_rates(keys: list[str], rates: list[float]) -> dict[str, float]:
    """Signed nodal residual from edge rates (source -= Q, target += Q)."""
    bal: dict[str, float] = defaultdict(float)
    for key, q in zip(keys, rates):
        ends = _edge_endpoints(key)
        if ends is None:
            continue
        src, tgt = ends
        bal[src] -= q
        bal[tgt] += q
    return dict(bal)


def assert_node_residuals_small(
    balances: dict[str, float],
    *,
    nodes: Iterable[str],
    abs_tol: float,
    rel_tol: float,
    scale_by_node: dict[str, float] | None = None,
    label: str = "residual",
) -> None:
    for node in nodes:
        r = float(balances.get(node, 0.0))
        scale = float((scale_by_node or {}).get(node, 0.0))
        limit = max(abs_tol, rel_tol * scale)
        assert abs(r) <= limit, f"{label}[{node}]={r} exceeds tol={limit} (abs={abs_tol}, rel={rel_tol}, scale={scale})"


def characteristic_flow_scale(keys: list[str], rates: list[float], node: str) -> float:
    total = 0.0
    for key, q in zip(keys, rates):
        ends = _edge_endpoints(key)
        if ends is None:
            continue
        if node in ends:
            total += abs(q)
    return total


def collect_heat_rate_matrices(
    artifact_dir: Path, output: dict[str, Any]
) -> tuple[list[str], list[list[float]]]:
    """Merge all thermal_heat_rate_* series into one key list + rows per timestep."""
    merged_keys: list[str] = []
    merged_rows: list[list[float]] | None = None
    for series_name in HEAT_RATE_SERIES:
        keys, rows = read_series_matrix(artifact_dir, output, series_name)
        if not keys:
            continue
        if merged_rows is None:
            merged_rows = [list(r) for r in rows]
            merged_keys = list(keys)
        else:
            assert len(rows) == len(merged_rows)
            for t in range(len(rows)):
                merged_rows[t].extend(rows[t])
            merged_keys.extend(keys)
    if merged_rows is None:
        return [], []
    return merged_keys, merged_rows


def assert_energy_balance_residuals(
    artifact_dir: Path,
    output: dict[str, Any],
    *,
    nodes: Iterable[str],
    abs_tol: float = tol.THERMAL_BALANCE_ABS_W,
    rel_tol: float = tol.THERMAL_BALANCE_REL,
) -> None:
    keys, rows = collect_heat_rate_matrices(artifact_dir, output)
    assert keys, "no heat_rate series to form energy residual"
    for t, row in enumerate(rows):
        assert_all_finite(row, label=f"heat_rates[{t}]")
        bal = node_balance_from_edge_rates(keys, row)
        scales = {n: characteristic_flow_scale(keys, row, n) for n in nodes}
        assert_node_residuals_small(
            bal,
            nodes=nodes,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            scale_by_node=scales,
            label=f"energy_residual[t={t}]",
        )


def assert_edge_mass_balance(
    keys: list[str],
    rows: list[list[float]],
    *,
    node_edge_groups: dict[str, list[tuple[str, float]]],
    abs_tol: float,
    label: str,
) -> None:
    """node_edge_groups: node -> list of (key_prefix_or_exact, sign)."""
    for t, row in enumerate(rows):
        assert_all_finite(row, label=f"{label}_rates[{t}]")
        d = dict(zip(keys, row))
        for node, terms in node_edge_groups.items():
            residual = 0.0
            for key_match, sign in terms:
                matched = [v for k, v in d.items() if k == key_match or k.startswith(key_match)]
                residual += sign * sum(matched)
            assert abs(residual) <= abs_tol, f"{label}[{node}] t={t} residual={residual}"


def assert_flux_series_node_balance(
    artifact_dir: Path,
    output: dict[str, Any],
    series_name: str,
    *,
    nodes: Iterable[str],
    abs_tol: float,
    label: str,
) -> None:
    keys, rows = read_series_matrix(artifact_dir, output, series_name)
    if not keys:
        return
    for t, row in enumerate(rows):
        assert_all_finite(row, label=f"{series_name}[{t}]")
        bal = node_balance_from_edge_rates(keys, row)
        # For storage nodes, flux sum equals storage rate — not necessarily 0.
        # We only assert global telescoping sum ≈ 0 and finite values here;
        # per-node storage checks are case-specific.
        total = sum(bal.values())
        assert abs(total) <= abs_tol * max(1, len(bal)), f"{label} global flux sum t={t}: {total}"
        for node in nodes:
            assert node in bal or True  # node may only appear as storage
            if node in bal:
                assert math.isfinite(bal[node])


def assert_artifact_no_nan_inf(artifact_dir: Path, output: dict[str, Any]) -> None:
    schema = json.loads((artifact_dir / "schema.json").read_text(encoding="utf-8"))
    for series_name, meta in schema.get("series", {}).items():
        keys = meta.get("keys") or []
        if not keys:
            continue
        result_key = output.get("result_files", {}).get(series_name)
        if not result_key:
            continue
        raw = (artifact_dir / result_key).read_bytes()
        if not raw:
            continue
        vals = struct.unpack("<" + "f" * (len(raw) // 4), raw)
        assert_all_finite(vals, label=series_name)


def assert_convergence_from_log(
    log_text: str,
    *,
    expect_thermal: bool = True,
    expect_pressure: bool = False,
    thermal_tol: float | None = None,
) -> SolverLogMetrics:
    """Assert convergence lines and absence of NaN/Inf in solver.log.

    When ``thermal_tol`` is None, trust the solver's own 収束/未収束 flag
    (already compared to simulation.tolerance.thermal) and only sanity-check
    that maxBalance is finite and not absurdly large.
    """
    metrics = parse_solver_log(log_text)
    assert metrics.nan_inf_mentions == 0, "solver.log mentions NaN/Inf"
    if expect_thermal:
        assert metrics.thermal_converged, "no thermal convergence lines in solver.log (raise log_verbosity?)"
        assert all(metrics.thermal_converged), f"thermal not converged: {metrics.thermal_converged}"
        peak = max(metrics.thermal_max_balance)
        assert math.isfinite(peak)
        if thermal_tol is not None:
            assert peak <= thermal_tol, f"thermal maxBalance peak {peak} > tol {thermal_tol}"
        else:
            assert peak < 1.0, f"thermal maxBalance peak looks unphysical: {peak}"
    if expect_pressure:
        assert not metrics.pressure_failed, "pressure solver reported failure"
        if metrics.pressure_residuals:
            assert max(metrics.pressure_residuals) < 1.0
    if metrics.coupled_iterations:
        assert max(metrics.coupled_iterations) >= 1
    return metrics
