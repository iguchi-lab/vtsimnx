import json
import struct
from pathlib import Path

import pytest

import app.solver_runner as sr
from app.builder import build_config


GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "thermal_regression_golden.json"


def _run_from_raw(
    *,
    raw_config: dict,
    run_id: str,
    tmp_base_dir: Path,
) -> tuple[dict, Path]:
    cfg = build_config(
        raw_config,
        output_path=None,
        add_aircon=False,
        add_capacity=True,
        add_surface=True,
    )
    cfg.setdefault("simulation", {}).setdefault("log", {})["verbosity"] = 0
    output = sr.run_solver(cfg, run_id=run_id, write_manifest=False)
    artifact_dir = tmp_base_dir / "work" / output["artifact_dir"]
    return output, artifact_dir


def _read_series(artifact_dir: Path, output: dict, series_name: str, key: str) -> list[float]:
    schema = json.loads((artifact_dir / "schema.json").read_text(encoding="utf-8"))
    keys = schema["series"][series_name]["keys"]
    idx = keys.index(key)
    width = len(keys)
    length = int(schema["length"])

    bin_path = artifact_dir / output["result_files"][series_name]
    raw = bin_path.read_bytes()
    vals = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    return [float(vals[t * width + idx]) for t in range(length)]


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
        # U値（定常）と等価な1項CTFを明示し、RCと同一挙動の回帰点を作る。
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


@pytest.mark.skipif(not Path(sr.SOLVER_EXE).exists(), reason="solver binary not found")
def test_physical_golden_room_cooling_rc(monkeypatch, tmp_path):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden["two_layer_wall_rc"]["room_temperature"]

    out, art = _run_from_raw(
        raw_config=_raw_two_layer_wall_rc_case(),
        run_id="physics_rc_golden",
        tmp_base_dir=tmp_path,
    )
    actual = _read_series(art, out, "thermal_temperature", "room")

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) <= 1e-4

    # 物理的な健全性チェック: 外気0C・室20C・発熱なしなら単調減少する。
    assert all(actual[i + 1] <= actual[i] + 1e-6 for i in range(len(actual) - 1))
    assert actual[-1] < actual[0]


@pytest.mark.skipif(not Path(sr.SOLVER_EXE).exists(), reason="solver binary not found")
def test_rc_vs_response_numeric_regression(monkeypatch, tmp_path):
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden["equivalent_uvalue_rc_vs_response"]["room_temperature"]

    out_rc, art_rc = _run_from_raw(
        raw_config=_raw_equivalent_uvalue_case(method="rc"),
        run_id="equiv_rc",
        tmp_base_dir=tmp_path,
    )
    out_resp, art_resp = _run_from_raw(
        raw_config=_raw_equivalent_uvalue_case(method="response"),
        run_id="equiv_response",
        tmp_base_dir=tmp_path,
    )

    rc = _read_series(art_rc, out_rc, "thermal_temperature", "room")
    resp = _read_series(art_resp, out_resp, "thermal_temperature", "room")

    assert len(rc) == len(resp) == len(expected)
    for r, e in zip(rc, expected):
        assert abs(r - e) <= 1e-4
    for s, e in zip(resp, expected):
        assert abs(s - e) <= 1e-4

    max_abs_diff = max(abs(r - s) for r, s in zip(rc, resp))
    assert max_abs_diff <= 1e-6
