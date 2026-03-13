from pathlib import Path
import json
import pytest

import app.solver_runner as sr


def _minimal_valid_input():
    # solver/parser/sim_constants_parser.cpp が要求する最低限の形
    return {
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
            "log": {"verbosity": 0},
        },
        # 省略しても warn で済むが、明示しておく（将来の必須化に備える）
        "nodes": [],
        "ventilation_branches": [],
        "thermal_branches": [],
    }


@pytest.mark.skipif(not Path(sr.SOLVER_EXE).exists(), reason="solver binary not found")
def test_solver_outputs_artifacts(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    input_path.write_text(json.dumps(_minimal_valid_input(), ensure_ascii=False, indent=2), encoding="utf-8")

    if output_path.exists():
        output_path.unlink()
    sr._invoke_solver(input_path, output_path, cwd=output_path.parent)
    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert out["status"] == "ok"

    artifact_dir = out["artifact_dir"]
    art = tmp_path / artifact_dir
    assert art.exists() and art.is_dir()

    # 最低限の成果物
    assert (art / out["log_file"]).exists()
    assert (art / "schema.json").exists()

    # bin は calc_flag=false でも空ファイルとして作られる（存在だけ確認）
    for key in (
        "vent_pressure",
        "vent_flow_rate",
        "thermal_temperature",
        "thermal_temperature_capacity",
        "thermal_temperature_layer",
        "thermal_heat_rate_advection",
        "thermal_heat_rate_heat_generation",
        "thermal_heat_rate_solar_gain",
        "thermal_heat_rate_nocturnal_loss",
        "thermal_heat_rate_convection",
        "thermal_heat_rate_conduction",
        "thermal_heat_rate_radiation",
        "thermal_heat_rate_capacity",
        "aircon_sensible_heat",
        "aircon_latent_heat",
        "aircon_power",
        "aircon_cop",
    ):
        assert key in out["result_files"]
        assert (art / out["result_files"][key]).exists()


