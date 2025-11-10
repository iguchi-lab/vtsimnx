import pytest


@pytest.fixture
def minimal_simulation():
    return {
        "index": {
            "start": "2025-01-01 00:00:00",
            "end":   "2025-01-01 23:00:00",
            "timestep": 3600,
            "length":   24,
        },
        "tolerance": {
            "ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6
        },
        # calc_flag は builder 側で nodes を見て自動設定されるが、最低限は与えておく
        "calc_flag": {"p": False, "t": False, "x": False, "c": False},
    }


@pytest.fixture
def minimal_input_config(minimal_simulation):
    """
    builder / validate 用の最小構成。
    surfaces / aircon は省略しても動作する。
    """
    return {
        "simulation": minimal_simulation,
        "nodes": [{"key": "室1"}],
        "ventilation_branches": [],
        "thermal_branches": [],
    }


