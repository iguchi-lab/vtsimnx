from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import vtsimnx as vt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run_calc の簡易疎通確認を行います")
    parser.add_argument("--base-url", default=os.getenv("VTSIMNX_API_URL", "http://localhost:8000"))
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--timestep", type=int, default=3600)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    config_json: Dict[str, Any] = {
        "simulation": {
            "length": args.length,
            "timestep": args.timestep,
        }
    }
    try:
        result = vt.run_calc(args.base_url, config_json, with_dataframes=True)
    except Exception as e:
        print(f"run_calc 実行に失敗しました: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if isinstance(result, vt.CalcRunResult):
        print(result.output)
        print("df_vent_flow:", None if result.df_vent_flow is None else result.df_vent_flow.shape)
        print("df_vent_pressure:", None if result.df_vent_pressure is None else result.df_vent_pressure.shape)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
