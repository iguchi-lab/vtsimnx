import json
import gzip
from dataclasses import dataclass, field
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests


def _is_nan_or_inf(x: float) -> bool:
    # JSONとしてInfinity/NaNを送ると相手側で失敗することがあるため None に落とす
    return math.isnan(x) or math.isinf(x)


def _to_jsonable(obj: Any) -> Any:
    """
    JSON化できない型（pandas/numpy 等）を、JSON互換の型へ再帰変換する。
    - pd.Series -> list
    - pd.DataFrame -> dict[str, list]
    - numpy scalar/ndarray -> python scalar / list
    - Timestamp/datetime/date -> ISO文字列
    - NaN/Inf/NaT -> None
    """
    # None / bool / int / str はそのまま
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # float: NaN/Inf を None へ
    if isinstance(obj, float):
        return None if _is_nan_or_inf(obj) else obj

    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # pandas
    if isinstance(obj, pd.Timestamp):
        # NaT もここに来る可能性があるため isna を見る
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj.isoformat()

    if isinstance(obj, pd.Series):
        return [_to_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, pd.Index):
        return [_to_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        # orient="list" で列->配列の形にしてから再帰変換
        as_dict = obj.to_dict(orient="list")
        return {str(k): [_to_jsonable(v) for v in vals] for k, vals in as_dict.items()}

    # numpy（依存は環境によっては optional の可能性があるので遅延import）
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return _to_jsonable(obj.item())

        if isinstance(obj, np.ndarray):
            return _to_jsonable(obj.tolist())
    except Exception:
        pass

    # dict / list / tuple
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            # JSONのキーは文字列のみ
            out[str(k)] = _to_jsonable(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # pandas の NaT / numpy.nan などスカラ判定の最後の砦
    try:
        if pd.isna(obj):  # type: ignore[arg-type]
            return None
    except Exception:
        pass

    raise TypeError(f"run_calc: JSONに変換できない型です: {type(obj).__name__}")


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
            from vtsimnx.artifacts.get_artifact_file import get_artifact_file

            df = get_artifact_file(self.base_url, self.artifact_dir, fname)
            if isinstance(df, pd.DataFrame):
                self._dataframes[series_name] = df
                return df
            self.errors[series_name] = f"TypeError: expected DataFrame, got {type(df).__name__}"
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
) -> Union[Dict[str, Any], CalcRunResult]:
    # 互換: 設定をファイル（.json / .json.gz）で渡せるようにする
    if not isinstance(config_json, dict):
        # 遅延import（循環回避）
        from vtsimnx.utils.utils import read_json

        config_json = read_json(config_json)  # type: ignore[assignment]
        if not isinstance(config_json, dict):
            raise TypeError(f"config_json must be dict (or json file path), got {type(config_json).__name__}")

    # pandas.Series などを含む場合でも送れるよう、JSON互換へ正規化
    config_json = _to_jsonable(config_json)  # type: ignore[assignment]
    if not isinstance(config_json, dict):
        raise TypeError(f"config_json must be dict after normalization, got {type(config_json).__name__}")

    url = base_url.rstrip("/") + "/run"
    payload = {"config": config_json}
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