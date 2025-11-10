import vtsimnx.parsers as parsers


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


