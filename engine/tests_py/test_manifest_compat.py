from __future__ import annotations

import json
import uuid

import app.solver_runner as sr


def test_write_artifact_manifest_includes_top_level_result_files_aliases():
    artifact_dir = f"output.test-manifest-compat.artifacts.{uuid.uuid4().hex}"
    output_data = {
        "artifact_dir": artifact_dir,
        "format_version": 5,
        "status": "ok",
        "log_file": "solver.log",
        "result_files": {
            "schema": "schema.json",
            "thermal_temperature": "thermal.temperature.f32.bin",
        },
    }

    manifest_path = sr.write_artifact_manifest(output_data)
    assert manifest_path is not None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output"]["result_files"] == output_data["result_files"]
    assert manifest["result_files"] == output_data["result_files"]
    assert manifest["files"]["schema"] == "schema.json"
    assert manifest["files"]["thermal_temperature"] == "thermal.temperature.f32.bin"
    assert manifest["files"]["log"] == "solver.log"
    assert manifest["files"]["manifest"] == "manifest.json"


def test_write_artifact_manifest_keeps_files_non_empty_on_error():
    artifact_dir = f"output.test-manifest-error.artifacts.{uuid.uuid4().hex}"
    output_data = {
        "artifact_dir": artifact_dir,
        "format_version": 5,
        "status": "error",
        "error": "example failure",
        "log_file": "solver.log",
        "builder_log_file": "builder.log",
    }

    manifest_path = sr.write_artifact_manifest(output_data)
    assert manifest_path is not None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["result_files"] == {}
    assert manifest["files"]["log"] == "solver.log"
    assert manifest["files"]["builder_log"] == "builder.log"
    assert manifest["files"]["manifest"] == "manifest.json"
