from copy import deepcopy

import pytest

from app.builder import build_config
from app.builder.warmup import apply_typical_year_warmup


def _year_raw(*, warmup=None, length=365, timestep=86400):
    t_out = list(range(length))
    raw = {
        "simulation": {
            "index": {
                "start": "2026-01-01 00:00:00",
                "end": "2026-12-31 00:00:00",
                "timestep": timestep,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "外部", "t": t_out},
            {"key": "室", "calc_t": True, "t": 20.0, "v": 10.0, "thermal_mass": 2.0e4},
        ],
        "ventilation_branches": [],
        "thermal_branches": [{"key": "室->外部", "conductance": 10.0}],
    }
    if warmup is not None:
        raw["simulation"]["warmup"] = warmup
    return raw


def test_warmup_prepends_december_tail():
    raw = _year_raw(warmup={"start": "12-15"})
    raw["aircon"] = [{"key": "AC", "set": "室", "mode": ["heating"] * 365}]
    apply_typical_year_warmup(raw)
    sim = raw["simulation"]
    assert sim["warmup"]["start"] == "12-15"
    assert sim["warmup"]["steps"] == 17
    assert sim["warmup"]["eval_length"] == 365
    assert sim["index"]["length"] == 365 + 17
    assert str(sim["index"]["start"]).startswith("2025-12-15")

    outdoor = next(node for node in raw["nodes"] if node["key"] == "外部")
    # 12/15 は 0-based 348（1/1 から 348 日後）
    assert outdoor["t"][0] == 348
    assert outdoor["t"][17] == 0
    assert outdoor["t"][-1] == 364
    assert raw["aircon"][0]["mode"][0] == "heating"
    assert len(raw["aircon"][0]["mode"]) == 382


def test_warmup_survives_build_config():
    out = build_config(_year_raw(warmup={"start": "12-15"}), output_path=None, add_surface=False)
    sim = out["simulation"]
    assert sim["warmup"]["start"] == "12-15"
    assert sim["warmup"]["steps"] == 17
    assert sim["warmup"]["eval_length"] == 365
    assert sim["index"]["length"] == 365 + 17
    outdoor = next(node for node in out["nodes"] if node["key"] == "外部")
    assert outdoor["t"][0] == 348
    assert len(outdoor["t"]) == 382


def test_warmup_string_start():
    raw = _year_raw(warmup="12-15")
    apply_typical_year_warmup(raw)
    assert raw["simulation"]["warmup"]["steps"] == 17
    assert raw["nodes"][0]["t"][0] == 348


def test_warmup_missing_date_errors():
    raw = _year_raw(warmup={"start": "12-15"}, length=31)
    raw["simulation"]["index"]["end"] = "2026-01-31 00:00:00"
    with pytest.raises(ValueError, match="12-15"):
        apply_typical_year_warmup(raw)


def test_warmup_is_idempotent_when_steps_set():
    raw = _year_raw(warmup={"start": "12-15"})
    apply_typical_year_warmup(raw)
    once = deepcopy(raw)
    apply_typical_year_warmup(raw)
    assert raw["simulation"]["index"]["length"] == once["simulation"]["index"]["length"]
    assert raw["nodes"][0]["t"] == once["nodes"][0]["t"]


def test_hourly_year_warmup_steps():
    raw = _year_raw(warmup={"start": "12-15"}, length=8760, timestep=3600)
    raw["simulation"]["index"]["end"] = "2026-12-31 23:00:00"
    apply_typical_year_warmup(raw)
    assert raw["simulation"]["warmup"]["steps"] == 17 * 24
    assert raw["simulation"]["index"]["length"] == 8760 + 17 * 24
    assert str(raw["simulation"]["index"]["start"]).startswith("2025-12-15")


def test_no_warmup_keeps_length():
    out = build_config(_year_raw(), output_path=None, add_surface=False)
    assert out["simulation"]["index"]["length"] == 365
    assert "warmup" not in out["simulation"]


def test_time_axis_is_extended_without_changing_response_coefficients():
    raw = _year_raw(warmup="12-15")
    matrix = [[i, -i] for i in range(365)]
    raw["nodes"][0]["c"] = matrix
    raw["thermal_branches"][0]["resp_a_src"] = list(range(365))
    apply_typical_year_warmup(raw)
    assert raw["nodes"][0]["c"] == matrix[-17:] + matrix
    assert raw["thermal_branches"][0]["resp_a_src"] == list(range(365))
