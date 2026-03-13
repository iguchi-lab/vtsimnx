import app.builder.parsers as parsers


def test_parse_nodes_compound_and_comment():
    raw = {"nodes": [{"key": "A&&B||comment"}]}
    out = parsers._parse_nodes(raw)
    keys = [n["key"] for n in out]
    # 先頭は予約の void
    assert keys[0] == "void"
    assert "A" in keys and "B" in keys
    for n in out:
        if n.get("key") in ("A", "B"):
            assert n.get("comment") == "comment"


def test_parse_chain_branches_expands_chain_and_preserves_comment():
    raw = {
        "ventilation_branches": [
            {"key": "A->B->C||x", "vol": 0.01},
        ]
    }
    out = parsers._parse_chain_branches(raw, "ventilation_branches")
    keys = [b["key"] for b in out]
    assert keys == ["A->B", "B->C"]
    assert all(b.get("comment") == "x" for b in out)
    assert all(b.get("vol") == 0.01 for b in out)


def test_parse_all_accepts_json_string_bytes_pathlike_and_filelike(tmp_path):
    cfg = {
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [{"key": "A"}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [],
        "aircon": [],
    }

    # JSON string
    sim, nodes, vent, therm, surf, ac = parsers.parse_all(__import__("json").dumps(cfg))
    assert len(nodes) >= 2 and nodes[0]["key"] == "void"

    # bytes
    sim2, nodes2, *_ = parsers.parse_all(__import__("json").dumps(cfg).encode("utf-8"))
    assert nodes2[0]["key"] == "void"

    # PathLike
    p = tmp_path / "x.json"
    p.write_text(__import__("json").dumps(cfg), encoding="utf-8")
    sim3, nodes3, *_ = parsers.parse_all(p)
    assert nodes3[0]["key"] == "void"

    # file-like
    import io, json
    f = io.StringIO(json.dumps(cfg))
    sim4, nodes4, *_ = parsers.parse_all(f)
    assert nodes4[0]["key"] == "void"


