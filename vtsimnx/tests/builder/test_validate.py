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


