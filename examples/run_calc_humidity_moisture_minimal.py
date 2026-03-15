from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import vtsimnx as vt


def build_outdoor_x_series(length: int) -> List[float]:
    half = max(1, length // 2)
    return [0.010] * half + [0.004] * (length - half)


def build_input_data() -> Dict[str, Any]:
    length = 24
    outdoor_x = build_outdoor_x_series(length)

    return {
        "builder": {
            # nodes[].moisture_capacity を持つノードから湿気容量ノードを自動生成
            "add_moisture_capacity": True,
        },
        "simulation": {
            "index": {
                "start": "2026-01-01 01:00:00",
                "end": "2026-01-02 00:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {
                "ventilation": 1e-6,
                "thermal": 1e-6,
                "convergence": 1e-6,
                "coupling_humidity": 1e-8,
            },
            "calc_flag": {
                "p": False,
                "t": True,
                "x": True,
                "c": False,
            },
            "coupling": {
                "moisture_enabled": True,
                "humidity_relaxation": 1.0,
                "humidity_solver_tolerance": 1e-9,
            },
        },
        "nodes": [
            {"key": "外部", "t": 5.0, "x": outdoor_x},
            {
                "key": "室1",
                "calc_t": True,
                "calc_x": True,
                "v": 30.0,
                "t": 20.0,
                "x": 0.006,
            },
            {
                # 壁材ノード（湿気容量を持つ単純モデル）
                "key": "壁材",
                "calc_x": True,
                "calc_t": False,
                "x": 0.006,
                "moisture_capacity": 5.0e5,
                "moisture_capacity_unit": "J/(kg/kg')",
            },
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
                "key": "外部->室1_熱",
                "source": "外部",
                "target": "室1",
                "type": "conductance",
                "conductance": 50.0,
            },
            {
                # 記述規約は source->target に統一（物理モデル上は moisture_conductance として双方向に扱われる）
                "key": "室1->壁材_湿気",
                "source": "室1",
                "target": "壁材",
                "type": "conductance",
                "conductance": 0.0,
                "moisture_conductance": 0.002,
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="湿気計算（calc_x + moisture_capacity）の最小実行サンプル"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VTSIMNX_API_URL", "http://127.0.0.1:8000"),
        help="VTSimNX API base URL",
    )
    parser.add_argument(
        "--request-json",
        default="result_request_humidity_minimal.json",
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

    df_x = result.get_series_df("humidity_x")
    df_flux = result.get_series_df("humidity_flux")

    print("\n[humidity_x columns]")
    print(list(df_x.columns))
    print(df_x.head())

    print("\n[humidity_flux columns]")
    print(list(df_flux.columns))
    print(df_flux.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
