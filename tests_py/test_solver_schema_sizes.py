import json
from pathlib import Path

import pytest

import app.solver_runner as sr


def _minimal_valid_input(length: int):
    return {
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": length},
            "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
            "log": {"verbosity": 0},
        },
        "nodes": [],
        "ventilation_branches": [],
        "thermal_branches": [],
    }


@pytest.mark.skipif(not Path(sr.SOLVER_EXE).exists(), reason="solver binary not found")
def test_schema_and_bin_sizes_match(tmp_path):
    length = 3
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_path.write_text(json.dumps(_minimal_valid_input(length), ensure_ascii=False, indent=2), encoding="utf-8")

    if output_path.exists():
        output_path.unlink()
    sr._invoke_solver(input_path, output_path, cwd=output_path.parent)
    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert out["status"] == "ok"

    art_dir = tmp_path / out["artifact_dir"]
    schema_path = art_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["length"] == length
    assert schema["dtype"] == "f32le"
    assert schema["layout"] == "timestep-major"

    # series の keys 数と bin サイズが一致すること（f32 = 4 bytes）
    series_to_result_key = {
        "vent_pressure": "vent_pressure",
        "vent_flow_rate": "vent_flow_rate",
        "thermal_temperature": "thermal_temperature",
        "thermal_temperature_capacity": "thermal_temperature_capacity",
        "thermal_temperature_layer": "thermal_temperature_layer",
        "thermal_heat_rate_advection": "thermal_heat_rate_advection",
        "thermal_heat_rate_heat_generation": "thermal_heat_rate_heat_generation",
        "thermal_heat_rate_solar_gain": "thermal_heat_rate_solar_gain",
        "thermal_heat_rate_nocturnal_loss": "thermal_heat_rate_nocturnal_loss",
        "thermal_heat_rate_convection": "thermal_heat_rate_convection",
        "thermal_heat_rate_conduction": "thermal_heat_rate_conduction",
        "thermal_heat_rate_radiation": "thermal_heat_rate_radiation",
        "thermal_heat_rate_capacity": "thermal_heat_rate_capacity",
        "aircon_sensible_heat": "aircon_sensible_heat",
        "aircon_latent_heat": "aircon_latent_heat",
        "aircon_power": "aircon_power",
        "aircon_cop": "aircon_cop",
    }
    for series_name, result_key in series_to_result_key.items():
        keys = schema["series"][series_name]["keys"]
        expected_bytes = length * len(keys) * 4
        bin_name = out["result_files"][result_key]
        bin_path = art_dir / bin_name
        assert bin_path.exists()
        assert bin_path.stat().st_size == expected_bytes


