"""Representative cases for performance history (not strict pass/fail)."""

from __future__ import annotations

from typing import Any, Callable


def case_thermal_rc_wall() -> dict[str, Any]:
    return {
        "builder": {"surface_layer_method": "rc"},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T11:00:00",
                "timestep": 3600,
                "length": 12,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "thermal_mass": 2.0e6},
            {"key": "outside", "t": [0.0] * 12, "calc_t": False},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "room->outside",
                "part": "wall",
                "area": 20.0,
                "alpha_i": 4.4,
                "alpha_o": 23.0,
                "layers": [
                    {"lambda": 0.16, "t": 0.12, "v_capa": 700000.0},
                    {"lambda": 0.04, "t": 0.05, "v_capa": 30000.0},
                ],
            }
        ],
    }


def case_stiff_pressure() -> dict[str, Any]:
    length = 6
    openings = []
    for i in range(12):
        openings.append(
            {
                "key": f"out_h->room_a||oa{i}",
                "type": "simple_opening",
                "alpha": 0.01 if i % 2 == 0 else 0.99,
                "area": 1e-4 if i % 2 == 0 else 2.0,
            }
        )
        openings.append(
            {
                "key": f"room_a->room_b||ab{i}",
                "type": "simple_opening",
                "alpha": 0.05 if i % 3 == 0 else 0.8,
                "area": 1e-3 if i % 3 == 0 else 1.5,
            }
        )
        openings.append(
            {
                "key": f"room_b->out_l||bo{i}",
                "type": "simple_opening",
                "alpha": 0.02 if i % 2 else 0.95,
                "area": 5e-4 if i % 2 else 1.8,
            }
        )
    return {
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-5, "thermal": 1e-5, "convergence": 1e-5},
            # 圧力網の性能計測が主目的。熱連成は別ケースで測る。
            "calc_flag": {"p": True, "t": False, "x": False, "c": False},
        },
        "nodes": [
            {"key": "out_h", "p": [30.0] * length, "t": 10.0, "calc_p": False},
            {"key": "out_l", "p": [0.0] * length, "t": 10.0, "calc_p": False},
            {"key": "room_a", "p": 0.0, "t": 22.0, "calc_p": True, "v": 40.0},
            {"key": "room_b", "p": 0.0, "t": 18.0, "calc_p": True, "v": 40.0},
        ],
        "ventilation_branches": openings,
        "thermal_branches": [],
    }


def case_hvac_heating() -> dict[str, Any]:
    # physics の ON/OFF 境界ケースと同型（性能計測用に少し長め）
    length = 6
    return {
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-3},
            "calc_flag": {"p": False, "t": True, "x": False, "c": False},
        },
        "nodes": [
            {"key": "outside", "t": 5.0},
            {"key": "room", "calc_t": True, "t": 15.0, "v": 40.0, "thermal_mass": 5.0e5},
        ],
        "ventilation_branches": [],
        "thermal_branches": [
            {"key": "outside->room", "type": "conductance", "conductance": 30.0},
        ],
        "aircon": [
            {
                "key": "AC1",
                "set": "room",
                "outside": "outside",
                "pre_temp": 22.0,
                "model": "CRIEPI",
                "mode": ["OFF", "OFF", "OFF", "HEATING", "HEATING", "HEATING"],
                "vol": 0.2,
                "ac_spec": {
                    "Q": {
                        "cooling": {"min": 0.7, "rtd": 2.2, "max": 3.3},
                        "heating": {"min": 0.7, "rtd": 2.5, "max": 5.4},
                    },
                    "P": {
                        "cooling": {"min": 0.095, "rtd": 0.395, "max": 0.78},
                        "heating": {"min": 0.095, "rtd": 0.39, "max": 1.36},
                    },
                    "V_inner": {
                        "cooling": {"rtd": 0.2016666666667},
                        "heating": {"rtd": 0.2016666666667},
                    },
                    "V_outer": {
                        "cooling": {"rtd": 0.47},
                        "heating": {"rtd": 0.47},
                    },
                },
            }
        ],
    }


BENCH_CASES: list[tuple[str, Callable[[], dict[str, Any]], dict[str, Any]]] = [
    (
        "thermal_rc_wall",
        case_thermal_rc_wall,
        {"add_aircon": False, "add_surface": True, "add_capacity": True},
    ),
    (
        "stiff_pressure",
        case_stiff_pressure,
        {"add_aircon": False, "add_surface": False, "add_capacity": False},
    ),
    (
        "hvac_heating",
        case_hvac_heating,
        {"add_aircon": True, "add_surface": False, "add_capacity": True},
    ),
]
