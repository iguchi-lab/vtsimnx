from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_tools_artifacts_to_csv_smoke(tmp_path: Path) -> None:
    # work/output.artifacts.XXXX/artifacts/...
    artifact_dir = tmp_path / "work" / "output.artifacts.000"
    artifacts_dir = artifact_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    # schema.json
    schema = {
        "dtype": "f32le",
        "layout": "timestep-major",
        "length": 3,
        "series": {
            "vent_flow_rate": {"keys": ["a", "b"]},  # N=2
            "vent_pressure": {"keys": []},  # scalar N=1
        },
    }
    (artifacts_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")

    # manifest.json (calc_result.json 互換の形)
    manifest = {
        "result": {
            "result_files": {
                "schema": "schema.json",
                "vent_flow_rate": "vent.flow_rate.f32.bin",
                "vent_pressure": "vent.pressure.f32.bin",
            }
        }
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # binaries
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="<f4").tofile(artifacts_dir / "vent.flow_rate.f32.bin")  # (3,2)
    np.array([10.0, 11.0, 12.0], dtype="<f4").tofile(artifacts_dir / "vent.pressure.f32.bin")  # (3,1)

    # run main()
    from vtsimnx.tools.artifacts_to_csv import main

    import sys

    argv_bak = sys.argv[:]
    try:
        sys.argv = ["vtsimnx.tools.artifacts_to_csv", "--artifact-dir", str(artifact_dir)]
        rc = main()
    finally:
        sys.argv = argv_bak

    assert rc == 0
    assert (artifacts_dir / "vent.flow_rate.csv").exists()
    assert (artifacts_dir / "vent.pressure.csv").exists()

    # quick content checks
    text = (artifacts_dir / "vent.flow_rate.csv").read_text(encoding="utf-8").splitlines()
    assert text[0] == "a,b"
    assert text[1].startswith("1.0,2.0")

    text2 = (artifacts_dir / "vent.pressure.csv").read_text(encoding="utf-8").splitlines()
    assert text2[0] == "vent_pressure"
    assert text2[1].startswith("10.0")


