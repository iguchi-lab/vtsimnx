import json

import pandas as pd
import pytest

from vtsimnx.archenv.nocturnal import nocturnal_gain_by_angles
from .conftest import requires_solver
from .helpers import read_series, run_from_raw

pytestmark = [pytest.mark.physics, requires_solver]


def wall_config():
    return {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00",
                "end": "2025-01-01T01:00:00",
                "length": 2,
                "timestep": 3600,
            }
        },
        "nodes": [{"key": "R", "t": 20}, {"key": "OA", "t": 20}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [{"key": "R->OA", "part": "wall", "area": 1, "u_value": 1}],
    }


def test_estimated_nocturnal_radiation_is_surface_heat_loss(solver_workdir):
    raw = wall_config()
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    loss = nocturnal_gain_by_angles(
        tilt_deg=0, t_out=pd.Series([5, 5], idx), rh_out=pd.Series([70, 70], idx)
    )
    raw["surfaces"][0].update({"nocturnal": loss.tolist(), "epsilon": 0.9})
    out, art = run_from_raw(
        raw_config=raw,
        run_id="night_sign",
        tmp_base_dir=solver_workdir,
        add_capacity=False,
        add_surface_radiation=False,
    )
    schema = json.loads((art / "schema.json").read_text())
    keys = schema["series"]["thermal_heat_rate_nocturnal_loss"]["keys"]
    assert len(keys) == 1
    assert read_series(
        art, out, "thermal_heat_rate_nocturnal_loss", keys[0]
    ) == pytest.approx([-60.28002487] * 2)


def test_reduced_response_wall_equal_boundaries_has_no_heat_source(solver_workdir):
    raw = wall_config()
    surface = raw["surfaces"][0]
    surface.pop("u_value")
    surface.update(
        {
            "layer_method": "response",
            "response_method": "modal_expsum",
            "response_terms": 1,
            "layers": [
                {"lambda": 1.4, "t": 0.1, "v_capa": 2e6},
                {"lambda": 0.04, "t": 0.05, "v_capa": 3e4},
            ],
        }
    )
    out, art = run_from_raw(
        raw_config=raw,
        run_id="modal_isothermal",
        tmp_base_dir=solver_workdir,
        add_capacity=False,
        add_surface_radiation=False,
    )
    schema = json.loads((art / "schema.json").read_text())
    keys = schema["series"]["thermal_temperature_layer"]["keys"]
    assert len(keys) == 2
    for key in keys:
        assert read_series(art, out, "thermal_temperature_layer", key) == pytest.approx(
            [20, 20], abs=1e-5
        )
