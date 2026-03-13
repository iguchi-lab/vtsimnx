import pytest

from app.builder.surfaces import process_surfaces
from app.builder.surfaces import _auto_response_coefficients_from_layers


def test_surfaces_response_conduction_generates_two_surface_nodes_and_response_branch():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "layers": [{"lambda": 1.0, "t": 0.1, "v_capa": 1000.0}],
            "layer_method": "response",
            "response": {
                "resp_a_src": [5.0],
                "resp_b_src": [-5.0],
                "resp_a_tgt": [5.0],
                "resp_b_tgt": [-5.0],
                "resp_c_src": [],
                "resp_c_tgt": [],
            },
        }
    ]
    nodes, tbs = process_surfaces(
        surfaces,
        sim_length=2,
        node_config=[{"key": "A", "t": 20.0}, {"key": "B", "t": 0.0}],
        add_solar=False,
        add_radiation=False,
        time_step=60.0,
    )

    # 両端表面ノードのみ（内部層ノードなし）
    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 2

    # response_conduction ブランチが1本
    resp = [b for b in tbs if b.get("type") == "response_conduction"]
    assert len(resp) == 1
    assert resp[0]["subtype"] == "conduction"
    assert resp[0]["area"] == 10.0
    for k in ("resp_a_src", "resp_b_src", "resp_a_tgt", "resp_b_tgt"):
        assert k in resp[0]


def test_auto_response_coefficients_from_layers_generates_per_m2_coeffs_and_shapes():
    # 通常ケース: フォールバックせず、len(c)=len(a)-1 になること
    resp = _auto_response_coefficients_from_layers(
        layers=[{"lambda": 0.8, "t": 0.12, "v_capa": 900000.0}],
        time_step=3600.0,
    )
    for k in ("resp_a_src", "resp_b_src", "resp_a_tgt", "resp_b_tgt", "resp_c_src", "resp_c_tgt"):
        assert k in resp
        assert isinstance(resp[k], list)

    # a/b は同じ長さ（少なくとも現在係数あり）
    assert len(resp["resp_a_src"]) >= 1
    assert len(resp["resp_a_src"]) == len(resp["resp_b_src"])
    assert len(resp["resp_a_tgt"]) == len(resp["resp_b_tgt"])

    # 自動生成系は len(c)=len(a)-1
    assert len(resp["resp_c_src"]) == max(len(resp["resp_a_src"]) - 1, 0)
    assert len(resp["resp_c_tgt"]) == max(len(resp["resp_a_tgt"]) - 1, 0)

    # 相互項は対称化される（同一配列）
    assert resp["resp_b_src"] == resp["resp_b_tgt"]


def test_auto_response_coefficients_falls_back_to_steady_state_when_sum_c_near_1():
    # 極端に遅い系（巨大熱容量）を作って sum(c)≈1 を誘発し、
    # builder が定常U値（メモリなし）にフォールバックすることを確認する。
    resp = _auto_response_coefficients_from_layers(
        layers=[{"lambda": 1.0, "t": 1.0, "v_capa": 1e12}],
        time_step=3600.0,
    )
    assert resp["resp_c_src"] == []
    assert resp["resp_c_tgt"] == []
    # R_total = t/lambda = 1, U=1
    assert resp["resp_a_src"] == [1.0]
    assert resp["resp_b_src"] == [-1.0]
    assert resp["resp_a_tgt"] == [1.0]
    assert resp["resp_b_tgt"] == [-1.0]


def test_auto_response_coefficients_modal_expsum_respects_terms():
    # terms を 1 にして次数を落とせること（a/b が長さ2、c が長さ1）
    resp = _auto_response_coefficients_from_layers(
        layers=[{"lambda": 0.8, "t": 0.12, "v_capa": 900000.0}, {"lambda": 0.04, "t": 0.08, "v_capa": 30000.0}],
        time_step=3600.0,
        response_method="modal_expsum",
        response_terms=1,
    )
    assert len(resp["resp_a_src"]) == 2
    assert len(resp["resp_b_src"]) == 2
    assert len(resp["resp_c_src"]) == 1
    assert len(resp["resp_a_tgt"]) == 2
    assert len(resp["resp_b_tgt"]) == 2
    assert len(resp["resp_c_tgt"]) == 1


