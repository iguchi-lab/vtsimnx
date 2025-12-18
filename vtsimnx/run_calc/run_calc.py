import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests


@dataclass(frozen=True)
class CalcRunResult:
    """
    run_calc(with_dataframes=True) の戻り値。

    - output: /run のレスポンス（manifest相当のJSON）
    - dataframes: series_name -> DataFrame
    - df_vent_flow / df_vent_pressure: よく使う系列へのショートカット
    """

    output: Dict[str, Any]
    artifact_dir: str
    dataframes: Dict[str, pd.DataFrame]
    errors: Dict[str, str]
    log_text: Optional[str]

    @property
    def df_vent_flow(self) -> Optional[pd.DataFrame]:
        return self.dataframes.get("vent_flow_rate")

    @property
    def df_vent_pressure(self) -> Optional[pd.DataFrame]:
        return self.dataframes.get("vent_pressure")

    @property
    def log(self) -> Optional[str]:
        return self.log_text


def _extract_output_block(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    # APIの形の揺れを吸収: {"output": {...}} / {"result": {...}} / 直下
    if isinstance(resp_json.get("output"), dict):
        return resp_json["output"]  # type: ignore[return-value]
    if isinstance(resp_json.get("result"), dict):
        return resp_json["result"]  # type: ignore[return-value]
    return resp_json


def run_calc(
    base_url: str,
    config_json: Dict[str, Any],
    output_path: Optional[str] = "calc_result.json",
    *,
    with_dataframes: bool = False,
    timeout: float = 60.0,
) -> Union[Dict[str, Any], CalcRunResult]:
    url = base_url.rstrip("/") + "/run"
    response = requests.post(url, json={"config": config_json}, timeout=timeout)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=4, ensure_ascii=False)
    resp_json: Dict[str, Any] = response.json()

    if not with_dataframes:
        return resp_json

    output = _extract_output_block(resp_json)
    artifact_dir = output.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError(f"run_calcレスポンスから artifact_dir を取得できませんでした: {artifact_dir!r}")

    result_files = output.get("result_files")
    if not isinstance(result_files, dict):
        raise ValueError("run_calcレスポンスから result_files を取得できませんでした")

    # 遅延import（import順の循環を避ける）
    from vtsimnx.artifacts.get_artifact_file import get_artifact_file

    dfs: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    log_text: Optional[str] = None

    # solver.log（あれば）を簡単に参照できるように取得しておく
    log_file = output.get("log_file")
    if isinstance(log_file, str) and log_file:
        try:
            raw = get_artifact_file(base_url, artifact_dir, log_file)
            if isinstance(raw, (bytes, bytearray)):
                log_text = bytes(raw).decode("utf-8", errors="replace")
        except Exception as e:
            errors["__log__"] = f"{type(e).__name__}: {e}"

    for series_name, fname in result_files.items():
        if not isinstance(series_name, str) or not isinstance(fname, str):
            continue
        if not fname.endswith(".f32.bin"):
            continue
        try:
            df = get_artifact_file(base_url, artifact_dir, fname)
            if isinstance(df, pd.DataFrame):
                dfs[series_name] = df
        except Exception as e:
            # 一部の系列が未生成/空/不整合でも、他の系列は利用できるようにスキップする
            errors[series_name] = f"{type(e).__name__}: {e}"

    return CalcRunResult(output=resp_json, artifact_dir=artifact_dir, dataframes=dfs, errors=errors, log_text=log_text)

if __name__ == "__main__":
    config_json = {
        "simulation": {
            "length": 8760,
            "timestep": 3600,
        }
    }
    base_url = "http://localhost:8000"
    calced = run_calc(base_url, config_json, with_dataframes=True)
    if isinstance(calced, CalcRunResult):
        print(calced.output)
        print("df_vent_flow:", None if calced.df_vent_flow is None else calced.df_vent_flow.shape)
        print("df_vent_pressure:", None if calced.df_vent_pressure is None else calced.df_vent_pressure.shape)
    else:
        print(calced)