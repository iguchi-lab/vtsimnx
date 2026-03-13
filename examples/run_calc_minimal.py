from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import vtsimnx as vt


def build_input_data() -> Dict[str, Any]:
    return {
        "builder": {},
        "simulation": {
            "index": {
                "start": "2026-01-01 01:00:00",
                "end": "2026-01-02 00:00:00",
                "timestep": 3600,
                "length": 24,
            }
        },
        "nodes": [
            {"key": "外部", "t": 5.0},
            {"key": "室1", "calc_t": True, "v": 30.0},
        ],
        "ventilation_branches": [
            {
                "key": "外部->室1",
                "source": "外部",
                "target": "室1",
                "type": "fixed_flow",
                "vol": 30.0,
            }
        ],
        "thermal_branches": [
            {
                "key": "外部->室1",
                "source": "外部",
                "target": "室1",
                "type": "conductance",
                "conductance": 50.0,
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vt.run_calc の最小実行サンプル")
    parser.add_argument(
        "--base-url",
        default=os.getenv("VTSIMNX_API_URL", "http://127.0.0.1:8000"),
        help="VTSimNX API base URL",
    )
    parser.add_argument(
        "--request-json",
        default="result_request_minimal.json",
        help="送信した input_data を保存するパス",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_data = build_input_data()

    result = vt.run_calc(
        args.base_url,
        input_data,
        request_output_path=args.request_json,
    )
    print(result.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
