import json
import copy
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from vtsimnx.artifacts import ArtifactClient, decode_f32_series
from vtsimnx.artifacts._schema import extract_manifest_error, extract_result_files
from vtsimnx.artifacts.errors import ArtifactError, ArtifactNotFound
from vtsimnx.run_calc._http import RunCalcAPIError, _post_run, _submit_and_wait
from vtsimnx.run_calc._index import (
    _normalize_simulation_index_inplace,
    _pick_index_spec,
)
from vtsimnx.run_calc._io import _write_json
from vtsimnx.run_calc._response import _output_block
from vtsimnx.utils.jsonable import to_jsonable

__all__ = ["CalcRunResult", "RunCalcAPIError", "run_calc"]


def _resolve_as_result(
    *,
    as_result: Optional[bool],
    with_dataframes: Optional[bool],
) -> bool:
    """
    戻り値モードを解決する。

    - 推奨: ``as_result``（True なら CalcRunResult、False なら生 dict）
    - ``with_dataframes`` は旧別名（DeprecationWarning）
    """
    if as_result is not None and with_dataframes is not None:
        if bool(as_result) != bool(with_dataframes):
            raise ValueError(
                "as_result と with_dataframes の値が一致しません "
                f"(as_result={as_result!r}, with_dataframes={with_dataframes!r})"
            )
        warnings.warn(
            "with_dataframes は非推奨です。as_result を使ってください "
            "(True=CalcRunResult, False=生のレスポンス dict)。",
            DeprecationWarning,
            stacklevel=3,
        )
        return bool(as_result)
    if as_result is not None:
        return bool(as_result)
    if with_dataframes is not None:
        warnings.warn(
            "with_dataframes は非推奨です。as_result を使ってください "
            "(True=CalcRunResult, False=生のレスポンス dict)。"
            " なお with_dataframes=True でも DataFrame は遅延ロードです。",
            DeprecationWarning,
            stacklevel=3,
        )
        return bool(with_dataframes)
    return True


@dataclass
class CalcRunResult:
    """
    run_calc(as_result=True) の戻り値。

    系列 DataFrame は必要になったときだけ取得・復元する（遅延ロード）。

    - output: /run または /runs/.../result のレスポンス（manifest相当のJSON）
    - artifact_dir: 成果物ディレクトリ名
    - base_url: APIベースURL
    - result_files: series_name -> filename（*.f32.bin 等）
    - raise_on_error: True なら系列/ログ取得失敗時に例外を再送出（既定は errors に記録して None）
    """

    output: Dict[str, Any]
    artifact_dir: str
    base_url: str
    result_files: Dict[str, str]
    # 送信した設定（クライアント側）。indexの復元などに使う
    config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    api_key: Optional[str] = field(default=None, repr=False)
    raise_on_error: bool = False
    errors: Dict[str, str] = field(default_factory=dict)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _log_text: Optional[str] = field(default=None, repr=False)
    _artifact_client: Optional[ArtifactClient] = field(default=None, repr=False)
    client_profile: Dict[str, Any] = field(default_factory=dict)
    _series_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    _log_profile: Dict[str, Any] = field(default_factory=dict, repr=False)

    def _get_artifact_client(self) -> ArtifactClient:
        if self._artifact_client is None:
            client = ArtifactClient(
                self.base_url, self.artifact_dir, api_key=self.api_key
            )
            # /runs レスポンスを seed し、追加の manifest GET を避ける
            client.seed_manifest(self.output)
            self._artifact_client = client
        return self._artifact_client

    def _want_raise(self, raise_on_error: Optional[bool]) -> bool:
        return self.raise_on_error if raise_on_error is None else bool(raise_on_error)

    def _handle_fetch_error(
        self,
        key: str,
        exc: BaseException,
        *,
        raise_on_error: Optional[bool],
    ) -> None:
        self.errors[key] = f"{type(exc).__name__}: {exc}"
        if self._want_raise(raise_on_error):
            raise exc

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

    def get_log_text(self, *, raise_on_error: Optional[bool] = None) -> Optional[str]:
        """
        solver.log などのログ本文を返す（必要なら取得）。

        - /runレスポンスに log.text が埋まっている場合はそれを使う（HTTP GET不要）
        - 無い場合は log_file を見て ArtifactClient で取得する
        - 失敗時は既定で errors['__log__'] に記録して None。raise_on_error=True なら例外
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
            t0 = time.perf_counter()
            client = self._get_artifact_client()
            raw = client.get_bytes(log_file)
            t1 = time.perf_counter()
            self._log_text = bytes(raw).decode("utf-8", errors="replace")
            self._log_profile = {
                "download_and_decode_ms": (t1 - t0) * 1000.0,
                "bytes": len(raw),
            }
            return self._log_text
        except ArtifactError as e:
            self._handle_fetch_error("__log__", e, raise_on_error=raise_on_error)
        except (TypeError, ValueError, OSError) as e:
            self._handle_fetch_error("__log__", e, raise_on_error=raise_on_error)
        except Exception as e:
            # 通信系など実行時依存の失敗は「取得失敗」として扱う
            wrapped = RuntimeError(f"{type(e).__name__}: {e}")
            wrapped.__cause__ = e
            self._handle_fetch_error("__log__", wrapped, raise_on_error=raise_on_error)

        return None

    def get_series_df(
        self,
        series_name: str,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        """
        指定系列の DataFrame を取得する（必要なら取得して復元）。

        - 系列が無い / 非 f32.bin の場合は既定で None。raise_on_error=True なら ArtifactNotFound
        - 取得・復元失敗時は既定で errors[series] に記録して None。raise_on_error=True なら例外
        """
        if series_name in self._dataframes:
            return self._dataframes[series_name]

        fname = self.result_files.get(series_name)
        if not isinstance(fname, str) or not fname:
            exc = ArtifactNotFound(f"series not in result_files: {series_name!r}")
            if self._want_raise(raise_on_error):
                raise exc
            return None

        # ここでは *.f32.bin のみ対象（他は bytes で返る想定）
        if not fname.endswith(".f32.bin"):
            exc = ArtifactNotFound(f"series is not f32.bin: {series_name!r} -> {fname!r}")
            if self._want_raise(raise_on_error):
                raise exc
            return None

        try:
            client = self._get_artifact_client()
            t0 = time.perf_counter()

            t_schema0 = time.perf_counter()
            schema_was_cached = client._schema is not None
            schema = client.get_schema()
            t_schema1 = time.perf_counter()
            schema_fetch_ms = 0.0 if schema_was_cached else (t_schema1 - t_schema0) * 1000.0

            T = schema.get("length")
            if not isinstance(T, int):
                raise TypeError(f"schema.json length が不正です: {T!r}")

            index_spec = _pick_index_spec(self.output, self.config, expected_length=T)

            t_bin0 = time.perf_counter()
            data = client.get_bytes(fname)
            t_bin1 = time.perf_counter()

            df = decode_f32_series(
                data,
                schema,
                series_name,
                index_spec=index_spec,
                source_name=fname,
            )
            t_df = time.perf_counter()

            self._dataframes[series_name] = df
            t_end = time.perf_counter()
            self._series_profiles[series_name] = {
                "total_ms": (t_end - t0) * 1000.0,
                "schema_fetch_ms": schema_fetch_ms,
                "bin_download_ms": (t_bin1 - t_bin0) * 1000.0,
                "dataframe_build_ms": (t_df - t_bin1) * 1000.0,
                "bytes": len(data),
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            }
            return df
        except ArtifactError as e:
            self._handle_fetch_error(series_name, e, raise_on_error=raise_on_error)
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            self._handle_fetch_error(series_name, e, raise_on_error=raise_on_error)
        except Exception as e:
            wrapped = RuntimeError(f"{type(e).__name__}: {e}")
            wrapped.__cause__ = e
            self._handle_fetch_error(series_name, wrapped, raise_on_error=raise_on_error)

        return None

    def load_all_dataframes(self, *, raise_on_error: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """
        全系列をロードする（旧挙動に近い動き）。
        """
        for series_name in list(self.result_files.keys()):
            _ = self.get_series_df(series_name, raise_on_error=raise_on_error)
        return self._dataframes


def run_calc(
    base_url: str,
    config_json: Union[Dict[str, Any], str, Path],
    output_path: Optional[str] = None,
    *,
    as_result: Optional[bool] = None,
    with_dataframes: Optional[bool] = None,
    raise_on_error: bool = False,
    compress_request: bool = True,
    timeout: float = 1800.0,
    api_key: Optional[str] = None,
    request_output_path: Optional[Union[str, Path]] = None,
    use_legacy_run: bool = False,
    poll_interval: float = 1.0,
) -> Union[Dict[str, Any], CalcRunResult]:
    """
    シミュレーションを実行する。

    Parameters
    ----------
    as_result:
        True（既定）なら ``CalcRunResult`` を返す。False なら API レスポンス dict を返す。
        DataFrame 自体は作らない（遅延ロード）。
    with_dataframes:
        ``as_result`` の旧別名（非推奨）。意味は as_result と同じ。
    raise_on_error:
        ``CalcRunResult`` 生成時に渡し、系列/ログ取得失敗を例外にする（既定は soft-fail）。
    timeout:
        ``/runs`` ポーリング打ち切り時間（秒）。既定は 1800（30分）。
    """
    return_result = _resolve_as_result(as_result=as_result, with_dataframes=with_dataframes)

    client_profile: Dict[str, Any] = {}
    t_total0 = time.perf_counter()

    if api_key is None:
        env_key = os.getenv("VTSIMNX_API_KEY", "").strip()
        api_key = env_key or None

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
    if use_legacy_run:
        resp_json = _post_run(
            base_url,
            payload=payload,
            compress_request=compress_request,
            timeout=timeout,
            api_key=api_key,
            profile_out=http_profile,
        )
    else:
        resp_json = _submit_and_wait(
            base_url,
            payload=payload,
            compress_request=compress_request,
            timeout=timeout,
            api_key=api_key,
            poll_interval=poll_interval,
            profile_out=http_profile,
        )
    t_post1 = time.perf_counter()
    client_profile.update(http_profile)
    client_profile["run_post_ms"] = (t_post1 - t_post0) * 1000.0
    client_profile["use_legacy_run"] = bool(use_legacy_run)

    if output_path is not None:
        _write_json(output_path, resp_json)
    t_total1 = time.perf_counter()
    client_profile["run_calc_total_ms"] = (t_total1 - t_total0) * 1000.0

    if not return_result:
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
        api_key=api_key,
        raise_on_error=bool(raise_on_error),
        client_profile=client_profile,
    )
