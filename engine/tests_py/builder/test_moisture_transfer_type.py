from app.builder import build_config
from app.schemas.config import ThermalBranchModel


def test_moisture_transfer_type_survives_builder_and_schema():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": True, "x": True, "c": False},
        },
        "nodes": [
            {"key": "ROOM", "v": 50.0, "calc_x": True, "t": 20.0},
            {"key": "MAT", "moisture_capacity": 10.0, "calc_x": True, "t": 20.0},
        ],
        "ventilation_branches": [],
        "thermal_branches": [
            {
                "key": "MAT->ROOM",
                "conductance": 1.0,
                "moisture_conductance": 0.002,
                "moisture_transfer_type": "vapor_diffusion",
            }
        ],
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=False)
    tb = next(b for b in out["thermal_branches"] if b["key"] == "MAT->ROOM")
    assert tb["moisture_conductance"] == 0.002
    assert tb["moisture_transfer_type"] == "vapor_diffusion"

    model = ThermalBranchModel.model_validate(tb)
    assert model.moisture_transfer_type == "vapor_diffusion"
