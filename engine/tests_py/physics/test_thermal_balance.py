from __future__ import annotations

import json

import pytest

from .conftest import requires_solver
from .helpers import (
    THERMAL_GOLDEN_PATH,
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    assert_energy_balance_residuals,
    assert_monotone_non_increasing,
    read_series,
    read_series_matrix,
    read_solver_log,
    run_from_raw,
)
from . import tolerances as tol


def _raw_two_layer_wall_rc_case() -> dict:
    return {
        "builder": {"surface_layer_method": "rc"},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": 6,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True},
            {"key": "outside", "t": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "calc_t": False},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "room->outside",
                "part": "wall",
                "area": 10.0,
                "alpha_i": 4.4,
                "alpha_o": 23.0,
                "layers": [
                    {"lambda": 0.16, "t": 0.12, "v_capa": 700000.0},
                    {"lambda": 0.04, "t": 0.05, "v_capa": 30000.0},
                ],
            }
        ],
    }


def _raw_equivalent_uvalue_case(*, method: str) -> dict:
    raw = {
        "builder": {"surface_layer_method": method},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": 6,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "thermal_mass": 2.0e6},
            {"key": "outside", "t": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "calc_t": False},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "room->outside",
                "part": "wall",
                "area": 10.0,
                "u_value": 0.5,
                "alpha_i": 4.4,
                "alpha_o": 23.0,
            }
        ],
    }
    if method == "response":
        raw["surfaces"][0]["layer_method"] = "response"
        raw["surfaces"][0]["response"] = {
            "resp_a_src": [0.5],
            "resp_b_src": [-0.5],
            "resp_a_tgt": [0.5],
            "resp_b_tgt": [-0.5],
            "resp_c_src": [],
            "resp_c_tgt": [],
        }
    return raw


@pytest.mark.physics
@requires_solver
def test_physical_golden_room_cooling_rc(solver_workdir):
    golden = json.loads(THERMAL_GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden["two_layer_wall_rc"]["room_temperature"]

    out, art = run_from_raw(
        raw_config=_raw_two_layer_wall_rc_case(),
        run_id="physics_rc_golden",
        tmp_base_dir=solver_workdir,
        log_verbosity=1,
    )
    actual = read_series(art, out, "thermal_temperature", "room")

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) <= 1e-4

    assert_monotone_non_increasing(actual)
    assert actual[-1] < actual[0]
    assert_artifact_no_nan_inf(art, out)
    assert_energy_balance_residuals(art, out, nodes=["room"])
    assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)

    for series_name in (
        "thermal_heat_rate_convection",
        "thermal_heat_rate_conduction",
        "thermal_heat_rate_capacity",
    ):
        _keys, rows = read_series_matrix(art, out, series_name)
        for t, row in enumerate(rows):
            assert_all_finite(row, label=f"{series_name}[{t}]")


@pytest.mark.physics
@requires_solver
def test_rc_vs_response_numeric_regression(solver_workdir):
    golden = json.loads(THERMAL_GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden["equivalent_uvalue_rc_vs_response"]["room_temperature"]

    out_rc, art_rc = run_from_raw(
        raw_config=_raw_equivalent_uvalue_case(method="rc"),
        run_id="equiv_rc",
        tmp_base_dir=solver_workdir,
    )
    out_resp, art_resp = run_from_raw(
        raw_config=_raw_equivalent_uvalue_case(method="response"),
        run_id="equiv_response",
        tmp_base_dir=solver_workdir,
    )

    rc = read_series(art_rc, out_rc, "thermal_temperature", "room")
    resp = read_series(art_resp, out_resp, "thermal_temperature", "room")

    assert len(rc) == len(resp) == len(expected)
    for r, e in zip(rc, expected):
        assert abs(r - e) <= 1e-4
    for s, e in zip(resp, expected):
        assert abs(s - e) <= 1e-4

    max_abs_diff = max(abs(r - s) for r, s in zip(rc, resp))
    assert max_abs_diff <= 1e-6

    assert_artifact_no_nan_inf(art_rc, out_rc)
    assert_artifact_no_nan_inf(art_resp, out_resp)
    assert_energy_balance_residuals(art_rc, out_rc, nodes=["room"])
    assert_convergence_from_log(read_solver_log(art_rc, out_rc), expect_thermal=True)
    # response 法は heat_rate 符号規約が異なる場合があるため、収束ログと有限性を主検証にする
    assert_convergence_from_log(read_solver_log(art_resp, out_resp), expect_thermal=True)
