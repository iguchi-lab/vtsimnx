import app.builder.validate as validate


def test_validate_dict_with_warning_details_collects_unknown_field_and_duplicate_key():
    cfg = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "A", "unknown_field": 123}, {"key": "B"}],
        "ventilation_branches": [
            {"key": "A->B", "vol": 0.01},
            {"key": "A->B", "vol": 0.02},
        ],
        "thermal_branches": [],
    }

    _out, _warnings, details = validate.validate_dict_with_warning_details(cfg)
    assert any(d.get("code") == "unknown_field_stripped" and d.get("field") == "unknown_field" for d in details)
    assert any(d.get("code") == "duplicate_key_renamed" and d.get("original_key") == "A->B" for d in details)


