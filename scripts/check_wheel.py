"""Compare an isolated installed wheel with the source material catalog."""
import argparse
import json
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-materials", type=Path)
    parser.add_argument("--expected-materials", type=Path)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.write_materials:
        sys.path.insert(0, str(root))
    import vtsimnx as vt
    from vtsimnx.materials import copy_materials

    if args.write_materials:
        payload = {"materials": copy_materials(),
                   "csv": sorted(p.name for p in (root / "vtsimnx/materials").glob("*.csv"))}
        args.write_materials.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return
    assert sys.flags.isolated, "Run the installed-wheel check with python -I"
    assert not Path(vt.__file__).resolve().is_relative_to(root / "vtsimnx"), vt.__file__
    assert callable(vt.run_calc) and callable(vt.wait_for_run)
    expected = json.loads(args.expected_materials.read_text(encoding="utf-8"))
    assert copy_materials() == expected["materials"], "Installed material properties differ from source"
    with zipfile.ZipFile(args.wheel) as archive:
        names = archive.namelist()
    assert not any(n.startswith("vtsimnx/tests/") for n in names), "Tests leaked into wheel"
    assert sorted(Path(n).name for n in names if n.startswith("vtsimnx/materials/") and n.endswith(".csv")) == expected["csv"]
    print(f"Installed wheel verified: {len(expected['materials'])} materials, {len(expected['csv'])} CSVs, no test package")


if __name__ == "__main__":
    main()
