import json
import copy
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from vtsimnx.artifacts._schema import extract_manifest_error, extract_result_files
from vtsimnx.run_calc._http import _post_run
from vtsimnx.run_calc._index import (
    _normalize_simulation_index_inplace,
    _time_index_from_config,
    _time_index_from_output,
)
from vtsimnx.run_calc._io import _write_json
from vtsimnx.run_calc._response import _output_block
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
    # 送信した設定（クライアント側）。indexの復元などに使う
    config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    errors: Dict[str, str] = field(default_factory=dict)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _log_text: Optional[str] = field(default=None, repr=False)
    _schema: Optional[Dict[str, Any]] = field(default=None, repr=False)
    client_profile: Dict[str, Any] = field(default_factory=dict)
    _series_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    _log_profile: Dict[str, Any] = field(default_factory=dict, repr=False)

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

    @property
    def series_profiles(self) -> Dict[str, Dict[str, Any]]:
        return self._series_profiles

    @property
    def log_profile(self) -> Dict[str, Any]:
        return self._log_profile

    def get_server_timings(self) -> List[Dict[str, Any]]:
        """
        APIレスポンス内 output.timings（C++ソルバ計測）を返す。
        """
        output = _output_block(self.output)
        timings = output.get("timings")
        if not isinstance(timings, list):
            return []
        rows: List[Dict[str, Any]] = []
        for entry in timings:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            duration = entry.get("duration_ms")
            if not isinstance(name, str) or not isinstance(duration, (int, float)):
                continue
            row: Dict[str, Any] = {"name": name, "duration_ms": float(duration)}
            meta = entry.get("meta")
            if isinstance(meta, str) and meta:
                row["meta"] = meta
            rows.append(row)
        return rows

    def get_timing_report(self) -> Dict[str, Any]:
        """
        クライアント側 + サーバー側の時間情報をまとめて返す。
        """
        server_timings = self.get_server_timings()

        load_input_ms = 0.0
        simulation_total_ms = 0.0
        for row in server_timings:
            name = row["name"]
            duration = row["duration_ms"]
            if name == "load_input":
                load_input_ms += duration
            elif name == "simulation_total":
                simulation_total_ms += duration

        solver_core_ms = load_input_ms + simulation_total_ms
        post_run_ms = float(self.client_profile.get("run_post_ms", 0.0) or 0.0)
        api_network_overhead_ms = max(post_run_ms - solver_core_ms, 0.0)

        return {
            "client": self.client_profile,
            "server": {
                "load_input_ms": load_input_ms,
                "simulation_total_ms": simulation_total_ms,
                "timings": server_timings,
            },
            "estimated": {
                "solver_core_ms": solver_core_ms,
                "api_network_overhead_ms": api_network_overhead_ms,
            },
            "artifacts": {
                "log_fetch": self._log_profile,
                "series_fetch": self._series_profiles,
            },
        }

    def get_log_text(self) -> Optional[str]:
        """
        solver.log などのログ本文を返す（必要なら取得）。

        - /runレスポンスに log.text が埋まっている場合はそれを使う（HTTP GET不要）
        - 無い場合は log_file を見て get_artifact_file() で取得する
        """
        if self._log_text is not None:
            return self._log_text

        output = _output_block(self.output)

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
        except ImportError as e:
            self.errors["__log__"] = f"ImportError: {e}"
            return None

        try:
            t0 = time.perf_counter()
            raw = get_artifact_file(self.base_url, self.artifact_dir, log_file)
            t1 = time.perf_counter()
            if isinstance(raw, (bytes, bytearray)):
                self._log_text = bytes(raw).decode("utf-8", errors="replace")
                self._log_profile = {
                    "download_and_decode_ms": (t1 - t0) * 1000.0,
                    "bytes": len(raw),
                }
                return self._log_text
            self.errors["__log__"] = f"TypeError: expected bytes, got {type(raw).__name__}"
        except (TypeError, ValueError, OSError) as e:
            self.errors["__log__"] = f"{type(e).__name__}: {e}"
        except Exception as e:
            # 通信系など実行時依存の失敗は「取得失敗」として扱う
            self.errors["__log__"] = f"RuntimeError: {type(e).__name__}: {e}"

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
        except ImportError as e:
            self.errors[series_name] = f"ImportError: {e}"
            return None

        try:
            # schema.json は複数系列で共通なのでキャッシュする（GET回数削減）
            t0 = time.perf_counter()
            if self._schema is None:
                t_schema0 = time.perf_counter()
                raw_schema = get_artifact_file(self.base_url, self.artifact_dir, "schema.json")
                if not isinstance(raw_schema, (bytes, bytearray)):
                    raise TypeError(f"schema.json: expected bytes, got {type(raw_schema).__name__}")
                self._schema = json.loads(bytes(raw_schema).decode("utf-8"))
                t_schema1 = time.perf_counter()
                schema_fetch_ms = (t_schema1 - t_schema0) * 1000.0
            else:
                schema_fetch_ms = 0.0

            schema = self._schema
            if not isinstance(schema, dict):
                raise TypeError(f"schema.json: expected dict, got {type(schema).__name__}")

            T = schema.get("length")
            if not isinstance(T, int) or T < 0:
                raise ValueError(f"schema.json length が不正です: {T!r}")

            cols = series_columns(schema, series_name)
            N = len(cols)

            # bin本体は bytes で取得して自前で復元（manifest.json は不要）
            t_bin0 = time.perf_counter()
            data = get_artifact_bytes(self.base_url, self.artifact_dir, fname)
            t_bin1 = time.perf_counter()
            arr = np.frombuffer(data, dtype=np.dtype("<f4"))
            expected = T * N
            if arr.size != expected:
                raise ValueError(
                    f"{fname}: 要素数が不一致です (actual={arr.size}, expected={expected}, T={T}, N={N})"
                )
            arr = arr.reshape((T, N))
            df = pd.DataFrame(arr, columns=cols)
            t_df = time.perf_counter()

            # 可能なら時間軸インデックスを付与
            # - まずAPIレスポンス(output.index) を優先
            # - 無ければクライアントが送った simulation.index から復元
            try:
                idx = _time_index_from_output(self.output, expected_length=T)
                if idx is None:
                    idx = _time_index_from_config(self.config, expected_length=T)
                if idx is not None:
                    df.index = idx
                    df.index.name = "time"
            except (TypeError, ValueError) as e:
                # 取得自体は成功させたいので、index付与の失敗は errors に記録して続行
                self.errors["__index__"] = f"{type(e).__name__}: {e}"

            self._dataframes[series_name] = df
            t_end = time.perf_counter()
            self._series_profiles[series_name] = {
                "total_ms": (t_end - t0) * 1000.0,
                "schema_fetch_ms": schema_fetch_ms,
                "bin_download_ms": (t_bin1 - t_bin0) * 1000.0,
                "dataframe_build_ms": (t_df - t_bin1) * 1000.0,
                "bytes": len(data),
                "rows": T,
                "cols": N,
            }
            return df
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            self.errors[series_name] = f"{type(e).__name__}: {e}"
        except Exception as e:
            self.errors[series_name] = f"RuntimeError: {type(e).__name__}: {e}"

        return None

    def load_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        全系列をロードする（旧挙動に近い動き）。
        """
        for series_name in list(self.result_files.keys()):
            _ = self.get_series_df(series_name)
        return self._dataframes


def run_calc(
    base_url: str,
    config_json: Union[Dict[str, Any], str, Path],
    output_path: Optional[str] = None,
    *,
    with_dataframes: bool = True,
    compress_request: bool = True,
    timeout: float = 600.0,
    request_output_path: Optional[Union[str, Path]] = None,
) -> Union[Dict[str, Any], CalcRunResult]:
    client_profile: Dict[str, Any] = {}
    t_total0 = time.perf_counter()

    # 互換: 設定をファイル（.json / .json.gz）で渡せるようにする
    t_prepare0 = time.perf_counter()
    if not isinstance(config_json, dict):
        # 遅延import（循環回避）
        from vtsimnx.utils.utils import read_json

        config_json = read_json(config_json)  # type: ignore[assignment]
        if not isinstance(config_json, dict):
            raise TypeError(f"config_json must be dict (or json file path), got {type(config_json).__name__}")

    # 呼び出し側の辞書を破壊しないようにコピーしてから正規化する
    config_json = copy.deepcopy(config_json)  # type: ignore[assignment]

    # pandas.Series などを含む場合でも送れるよう、JSON互換へ正規化
    _normalize_simulation_index_inplace(config_json)
    config_json = to_jsonable(config_json)  # type: ignore[assignment]
    if not isinstance(config_json, dict):
        raise TypeError(f"config_json must be dict after normalization, got {type(config_json).__name__}")

    payload = {"config": config_json}

    # デバッグ/監査用途: 送信するリクエストJSONを保存（必要な場合のみ）
    if request_output_path is not None:
        _write_json(request_output_path, config_json)
    t_prepare1 = time.perf_counter()
    client_profile["prepare_input_ms"] = (t_prepare1 - t_prepare0) * 1000.0

    http_profile: Dict[str, Any] = {}
    t_post0 = time.perf_counter()
    resp_json = _post_run(
        base_url,
        payload=payload,
        compress_request=compress_request,
        timeout=timeout,
        profile_out=http_profile,
    )
    t_post1 = time.perf_counter()
    client_profile.update(http_profile)
    client_profile["run_post_ms"] = (t_post1 - t_post0) * 1000.0

    if output_path is not None:
        _write_json(output_path, resp_json)
    t_total1 = time.perf_counter()
    client_profile["run_calc_total_ms"] = (t_total1 - t_total0) * 1000.0

    if not with_dataframes:
        return resp_json

    output = _output_block(resp_json)
    error_message = extract_manifest_error(output)
    if error_message:
        raise ValueError(error_message)

    artifact_dir = output.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError(f"run_calcレスポンスから artifact_dir を取得できませんでした: {artifact_dir!r}")

    result_files = extract_result_files(output)

    # ここでは DataFrame を作らない（遅延ロード）
    return CalcRunResult(
        output=resp_json,
        artifact_dir=artifact_dir,
        base_url=base_url,
        result_files=result_files,
        config=config_json,
        client_profile=client_profile,
    )