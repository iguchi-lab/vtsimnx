from __future__ import annotations

from typing import Any, Dict

import vtsimnx as vt


def main() -> int:
    config_json: Dict[str, Any] = {
        "simulation": {
            "length": 24,
            "timestep": 3600,
        }
    }
    base_url = "http://localhost:8000"
    result = vt.run_calc(base_url, config_json, with_dataframes=True)

    if isinstance(result, vt.CalcRunResult):
        print(result.output)
        print("df_vent_flow:", None if result.df_vent_flow is None else result.df_vent_flow.shape)
        print("df_vent_pressure:", None if result.df_vent_pressure is None else result.df_vent_pressure.shape)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
