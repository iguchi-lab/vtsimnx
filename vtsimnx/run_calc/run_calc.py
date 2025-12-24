import json
import gzip
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import requests

from vtsimnx.utils.jsonable import to_jsonable


@dataclass
class CalcRunResult:
    """
    run_calc(with_dataframes=True) の戻り値。

    以前は run_calc() 内で全系列を DataFrame 化していましたが、
    大量のHTTP GET / 復元コストがかかるため、現在は「必要になったときだけ」
    DataFrame を取得・復元する（遅延ロード）方式にしています。

    - output: /run のレスポンス（manifest相当のJSON）
    - artifact_dir: 成果物ディレクトリ名
    - base_url: APIベースURL
    - result_files: series_name -> filename（*.f32.bin 等）
    """

    output: Dict[str, Any]
    artifact_dir: str
    base_url: str
    result_files: Dict[str, str]
    errors: Dict[str, str] = field(default_factory=dict)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _log_text: Optional[str] = field(default=None, repr=False)
    _schema: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        # 後方互換: すでにロード済みのDataFrameだけ返す（ロードはしない）
        return self._dataframes

    @property
    def df_vent_flow(self) -> Optional[pd.DataFrame]:
        return self.get_series_df("vent_flow_rate")

    @property
    def df_vent_pressure(self) -> Optional[pd.DataFrame]:
        return self.get_series_df("vent_pressure")

    @property
    def log(self) -> Optional[str]:
        return self.get_log_text()

    def get_log_text(self) -> Optional[str]:
        """
        solver.log などのログ本文を返す（必要なら取得）。

        - /runレスポンスに log.text が埋まっている場合はそれを使う（HTTP GET不要）
        - 無い場合は log_file を見て get_artifact_file() で取得する
        """
        if self._log_text is not None:
            return self._log_text

        output = _extract_output_block(self.output)

        # まずはレスポンス内の log.text を優先
        log_block = output.get("log")
        if isinstance(log_block, dict):
            text = log_block.get("text")
            if isinstance(text, str):
                self._log_text = text
                return self._log_text

        log_file = output.get("log_file")
        if not isinstance(log_file, str) or not log_file:
            return None

        try:
            # 遅延import（import順の循環を避ける）
            from vtsimnx.artifacts.get_artifact_file import get_artifact_file

            raw = get_artifact_file(self.base_url, self.artifact_dir, log_file)
            if isinstance(raw, (bytes, bytearray)):
                self._log_text = bytes(raw).decode("utf-8", errors="replace")
                return self._log_text
        except Exception as e:
            self.errors["__log__"] = f"{type(e).__name__}: {e}"

        return None

    def get_series_df(self, series_name: str) -> Optional[pd.DataFrame]:
        """
        指定系列の DataFrame を取得する（必要なら取得して復元）。
        """
        if series_name in self._dataframes:
            return self._dataframes[series_name]

        fname = self.result_files.get(series_name)
        if not isinstance(fname, str) or not fname:
            return None

        # ここでは *.f32.bin のみ対象（他は bytes で返る想定）
        if not fname.endswith(".f32.bin"):
            return None

        try:
            # 遅延import（import順の循環を避ける）
            from vtsimnx.artifacts.get_artifact_file import get_artifact_file, get_artifact_bytes
            from vtsimnx.artifacts._schema import series_columns

            # schema.json は複数系列で共通なのでキャッシュする（GET回数削減）
            if self._schema is None:
                raw_schema = get_artifact_file(self.base_url, self.artifact_dir, "schema.json")
                if not isinstance(raw_schema, (bytes, bytearray)):
                    raise TypeError(f"schema.json: expected bytes, got {type(raw_schema).__name__}")
                self._schema = json.loads(bytes(raw_schema).decode("utf-8"))

            schema = self._schema
            if not isinstance(schema, dict):
                raise TypeError(f"schema.json: expected dict, got {type(schema).__name__}")

            T = schema.get("length")
            if not isinstance(T, int) or T < 0:
                raise ValueError(f"schema.json length が不正です: {T!r}")

            cols = series_columns(schema, series_name)
            N = len(cols)

            # bin本体は bytes で取得して自前で復元（manifest.json は不要）
            data = get_artifact_bytes(self.base_url, self.artifact_dir, fname)
            arr = np.frombuffer(data, dtype=np.dtype("<f4"))
            expected = T * N
            if arr.size != expected:
                raise ValueError(
                    f"{fname}: 要素数が不一致です (actual={arr.size}, expected={expected}, T={T}, N={N})"
                )
            arr = arr.reshape((T, N))
            df = pd.DataFrame(arr, columns=cols)

            self._dataframes[series_name] = df
            return df
        except Exception as e:
            self.errors[series_name] = f"{type(e).__name__}: {e}"

        return None

    def load_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        全系列をロードする（旧挙動に近い動き）。
        """
        for series_name in list(self.result_files.keys()):
            _ = self.get_series_df(series_name)
        return self._dataframes


def _extract_output_block(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    # APIの形の揺れを吸収: {"output": {...}} / {"result": {...}} / 直下
    if isinstance(resp_json.get("output"), dict):
        return resp_json["output"]  # type: ignore[return-value]
    if isinstance(resp_json.get("result"), dict):
        return resp_json["result"]  # type: ignore[return-value]
    return resp_json


def _extract_result_files(output: Dict[str, Any]) -> Dict[str, str]:
    result_files = output.get("result_files")
    if not isinstance(result_files, dict):
        # 互換: files という名前の場合
        result_files = output.get("files")
    if not isinstance(result_files, dict):
        raise ValueError("run_calcレスポンスから result_files/files を取得できませんでした")

    out: Dict[str, str] = {}
    for k, v in result_files.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def run_calc(
    base_url: str,
    config_json: Union[Dict[str, Any], str, Path],
    output_path: Optional[str] = "calc_result.json",
    *,
    with_dataframes: bool = True,
    compress_request: bool = True,
    timeout: float = 60.0,
    request_output_path: Optional[Union[str, Path]] = None,
) -> Union[Dict[str, Any], CalcRunResult]:
    # 互換: 設定をファイル（.json / .json.gz）で渡せるようにする
    if not isinstance(config_json, dict):
        # 遅延import（循環回避）
        from vtsimnx.utils.utils import read_json

        config_json = read_json(config_json)  # type: ignore[assignment]
        if not isinstance(config_json, dict):
            raise TypeError(f"config_json must be dict (or json file path), got {type(config_json).__name__}")

    # pandas.Series などを含む場合でも送れるよう、JSON互換へ正規化
    config_json = to_jsonable(config_json)  # type: ignore[assignment]
    if not isinstance(config_json, dict):
        raise TypeError(f"config_json must be dict after normalization, got {type(config_json).__name__}")

    url = base_url.rstrip("/") + "/run"
    payload = {"config": config_json}

    # デバッグ/監査用途: 送信するリクエストJSONを保存（必要な場合のみ）
    if request_output_path is not None:
        p = Path(request_output_path)
        if p.suffix.lower() == ".gz":
            with gzip.open(p, "wt", encoding="utf-8") as f:
                json.dump(config_json, f, ensure_ascii=False, indent=2)
        else:
            with p.open("w", encoding="utf-8") as f:
                json.dump(config_json, f, ensure_ascii=False, indent=2)

    if compress_request:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        gz = gzip.compress(raw)
        response = requests.post(
            url,
            data=gz,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
    else:
        response = requests.post(url, json=payload, timeout=timeout)
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

    result_files = _extract_result_files(output)

    # ここでは DataFrame を作らない（遅延ロード）
    return CalcRunResult(
        output=resp_json,
        artifact_dir=artifact_dir,
        base_url=base_url,
        result_files=result_files,
    )

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