import pytest
import vtsimnx.builder.validate as validate


def test_validate_dict_minimal(minimal_input_config):
    out = validate.validate_dict(minimal_input_config)
    assert set(out.keys()) >= {"simulation", "nodes", "ventilation_branches", "thermal_branches"}


def test_validate_dict_collects_errors(minimal_input_config):
    # 故意に不正な thermal_branches を混ぜる（存在しないノード参照）
    cfg = dict(minimal_input_config)
    cfg["thermal_branches"] = [{"key": "X->Y", "conductance": 1.0}]
    try:
        validate.validate_dict(cfg)
        assert False, "ValidationError が発生すべき"
    except validate.ValidationError as e:
        # エラーメッセージが複数含まれる可能性があるが、最低限文字列であることを確認
        assert isinstance(str(e), str) and len(str(e)) > 0


def test_duplicate_key_rename_preserves_source_target_ventilation(minimal_simulation):
    cfg = {
        "simulation": minimal_simulation,
        "nodes": [{"key": "A"}, {"key": "B"}],
        "ventilation_branches": [
            {"key": "A->B", "vol": 0.01},
            {"key": "A->B", "vol": 0.02},
        ],
        "thermal_branches": [],
    }
    out = validate.validate_dict(cfg)
    keys = [b["key"] for b in out["ventilation_branches"]]
    assert all(k.startswith("A->B(") and k.endswith(")") for k in keys)
    assert all(b["source"] == "A" and b["target"] == "B" for b in out["ventilation_branches"])


def test_duplicate_key_rename_preserves_source_target_thermal(minimal_simulation):
    cfg = {
        "simulation": minimal_simulation,
        "nodes": [{"key": "B"}],
        "ventilation_branches": [],
        "thermal_branches": [
            {"key": "->B", "conductance": 1.0},
            {"key": "->B", "conductance": 2.0},
        ],
    }
    out = validate.validate_dict(cfg)
    keys = [b["key"] for b in out["thermal_branches"]]
    assert all(k.startswith("->B(") and k.endswith(")") for k in keys)
    assert all(b["source"] == "void" and b["target"] == "B" for b in out["thermal_branches"])


def test_invalid_chain_in_ventilation_raises(minimal_simulation):
    cfg = {
        "simulation": minimal_simulation,
        "nodes": [{"key": "A"}, {"key": "B"}],
        "ventilation_branches": [{"key": "A-B", "vol": 0.01}],  # "->" でない不正形式
        "thermal_branches": [],
    }
    with pytest.raises(validate.ValidationError):
        validate.validate_dict(cfg)


def test_validate_config_file_requires_all_sections(tmp_path):
    bad = {
        "simulation": {
            "index": {"start": "2025-01-01 00:00:00", "end": "2025-01-01 01:00:00", "timestep": 3600, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [],
        "ventilation_branches": [],
        # "thermal_branches" が欠落
    }
    p = tmp_path / "bad.json"
    import json as _json
    p.write_text(_json.dumps(bad), encoding="utf-8")
    with pytest.raises(validate.ConfigFileError):
        validate.validate(str(p))


def test_unknown_fields_are_stripped(minimal_simulation):
    cfg = {
        "simulation": minimal_simulation,
        "nodes": [{"key": "A", "unknown_field": 123}],  # NodeType に無いフィールド
        "ventilation_branches": [
            {"key": "A->A2", "vol": 0.01, "unknown_x": True}
        ],
        "thermal_branches": [],
    }
    cfg["nodes"].append({"key": "A2"})
    out = validate.validate_dict(cfg)
    assert "unknown_field" not in out["nodes"][0]
    assert "unknown_x" not in out["ventilation_branches"][0]
