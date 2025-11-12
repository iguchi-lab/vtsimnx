import json
import vtsim_config as vt


def test_build_config_minimal(tmp_path, minimal_input_config):
    out_path = tmp_path / "parsed.json"
    out = vt.build_config(minimal_input_config, output_path=str(out_path))
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    # 基本キーが揃っている
    assert set(saved.keys()) >= {"simulation", "nodes", "ventilation_branches", "thermal_branches"}
    # 返り値も辞書で一致（キーの一部だけ検証）
    assert "simulation" in out and "nodes" in out


