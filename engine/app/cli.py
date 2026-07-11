"""CLI 単発実行（API /run と同経路）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.builder.validate import ConfigFileError, ValidationError
from app.services.simulation import run_simulation_core


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run VTSimNX once (same path as API /run)")
    parser.add_argument("input_path", type=str, help="Input JSON file path (raw config)")
    parser.add_argument("--output", type=str, default=None, help="Write SimulationResponse JSON to this path")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--debug", action="store_true", help="デバッグ: verbosity を引き上げる（最低 debug_verbosity）")
    g.add_argument("--quiet", action="store_true", help="静かに: verbosity=0（silent）にする")
    parser.add_argument("--debug-verbosity", type=int, default=2, help="--debug時のverbosity下限（既定: 2）")
    parser.add_argument("--verbosity", type=int, default=None, help="verbosityを明示指定（指定時は--debug/--quietより優先）")
    args = parser.parse_args(argv)

    try:
        raw = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError("input.json root must be object")
        if args.verbosity is not None:
            sim = raw.get("simulation")
            if not isinstance(sim, dict):
                sim = {}
                raw["simulation"] = sim
            log = sim.get("log")
            if not isinstance(log, dict):
                log = {}
                sim["log"] = log
            log["verbosity"] = int(args.verbosity)
            debug = True
            debug_verbosity = int(args.verbosity)
        elif args.quiet:
            sim = raw.get("simulation")
            if not isinstance(sim, dict):
                sim = {}
                raw["simulation"] = sim
            log = sim.get("log")
            if not isinstance(log, dict):
                log = {}
                sim["log"] = log
            log["verbosity"] = 0
            debug = True
            debug_verbosity = 0
        else:
            debug = bool(args.debug)
            debug_verbosity = int(args.debug_verbosity)

        resp = run_simulation_core(raw_config=raw, debug=debug, debug_verbosity=debug_verbosity)
        payload = resp.model_dump()
        text = json.dumps(payload, ensure_ascii=False, indent=2)

        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0
    except (ValidationError, ConfigFileError) as e:
        sys.stderr.write(str(e) + "\n")
        return 2
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        return 1
