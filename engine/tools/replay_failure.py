"""Replay a captured solver input without overwriting the original run."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def replay(bundle, solver, destination, allow_different_solver=False):
    bundle, solver, destination = (Path(p).resolve() for p in (bundle, solver, destination))
    metadata = json.loads((bundle / "replay.json").read_text(encoding="utf-8"))
    solver_hash = digest(solver)
    if solver_hash != metadata.get("solver_sha256") and not allow_different_solver:
        raise ValueError("Solver hash differs; select the original binary or explicitly allow comparison")
    destination.mkdir(parents=True, exist_ok=False)
    inp = destination / "input.json"
    with gzip.open(bundle / "input.json.gz", "rb") as source, inp.open("wb") as target:
        shutil.copyfileobj(source, target)
    if digest(inp) != metadata["input_sha256"]:
        raise ValueError("Input hash mismatch; solver was not executed")
    (destination / "replay_source.json").write_text(json.dumps({
        "source_run_id": metadata["run_id"], "input_sha256": metadata["input_sha256"],
        "original_solver_sha256": metadata.get("solver_sha256"),
        "executed_solver_sha256": solver_hash,
    }, indent=2), encoding="utf-8")
    with (destination / "process.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(solver), str(inp), str(destination / "output.json")],
                                cwd=destination, stdout=log, stderr=subprocess.STDOUT, check=False)
    output = destination / "output.json"
    if output.is_file():
        data = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps({"status": data.get("status"), "error_code": data.get("error_code"),
                          "output": str(output)}, ensure_ascii=False))
        if data.get("status") != "ok" or data.get("error"):
            return result.returncode or 1
    return result.returncode if output.is_file() else (result.returncode or 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--solver", type=Path, default=Path(__file__).resolve().parents[1] / "build/vtsimnx_solver")
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--allow-different-solver", action="store_true")
    args = parser.parse_args()
    raise SystemExit(replay(args.bundle, args.solver, args.out, args.allow_different_solver))
