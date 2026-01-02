import pandas as pd

from vtsimnx.run_calc.run_calc import _time_index_from_config, _time_index_from_output


def test_time_index_from_config_builds_datetimeindex():
    cfg = {
        "simulation": {
            "index": {
                "start": "2025-01-01 01:00:00",
                "end": "2025-01-01 03:00:00",
                "timestep": 3600,
                "length": 3,
            }
        }
    }
    idx = _time_index_from_config(cfg, expected_length=3)
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx[0] == pd.Timestamp("2025-01-01 01:00:00")
    assert idx[-1] == pd.Timestamp("2025-01-01 03:00:00")


def test_time_index_from_output_builds_datetimeindex():
    resp = {
        "output": {
            "index": {
                "start": "2025-01-01 01:00:00",
                "end": "2025-01-01 03:00:00",
                "timestep": 3600,
                "length": 3,
            }
        }
    }
    idx = _time_index_from_output(resp, expected_length=3)
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx[0] == pd.Timestamp("2025-01-01 01:00:00")
    assert idx[-1] == pd.Timestamp("2025-01-01 03:00:00")


