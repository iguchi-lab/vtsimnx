"""manifest / schema の正規化とヘルパー。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import ArtifactDecodeError, ArtifactNotFound


def extract_manifest_error(manifest: Dict[str, Any]) -> Optional[str]:
    """
    manifest/レスポンス相当のJSONから、ユーザーに見せるべき失敗理由を抽出する。
    """
    output: Dict[str, Any]
    if isinstance(manifest.get("output"), dict):
        output = manifest["output"]
    elif isinstance(manifest.get("result"), dict):
        output = manifest["result"]
    else:
        output = manifest

    error = output.get("error")
    if isinstance(error, str) and error.strip():
        extras: List[str] = []
        artifact_dir = output.get("artifact_dir")
        if isinstance(artifact_dir, str) and artifact_dir:
            extras.append(f"artifact_dir={artifact_dir}")
        log_file = output.get("log_file")
        if isinstance(log_file, str) and log_file:
            extras.append(f"log={log_file}")
        builder_log_file = output.get("builder_log_file")
        if isinstance(builder_log_file, str) and builder_log_file:
            extras.append(f"builder_log={builder_log_file}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        message = f"シミュレーションに失敗しました: {error.strip()}{suffix}"

        log_obj = output.get("log")
        if isinstance(log_obj, dict):
            log_text = log_obj.get("text")
            if isinstance(log_text, str) and log_text.strip():
                message = f"{message}\n--- solver.log (tail) ---\n{log_text.strip()}"
        return message

    status = output.get("status")
    if isinstance(status, str) and status.lower() == "error":
        status_extras: List[str] = []
        artifact_dir = output.get("artifact_dir")
        if isinstance(artifact_dir, str) and artifact_dir:
            status_extras.append(f"artifact_dir={artifact_dir}")
        suffix = f" ({', '.join(status_extras)})" if status_extras else ""
        return f"シミュレーションに失敗しました{suffix}"

    return None


def _pick_result_files(manifest: Dict[str, Any]) -> Dict[str, str]:
    candidates = []
    if isinstance(manifest.get("output"), dict) and isinstance(manifest["output"].get("result_files"), dict):
        candidates.append(manifest["output"]["result_files"])
    if isinstance(manifest.get("result"), dict) and isinstance(manifest["result"].get("result_files"), dict):
        candidates.append(manifest["result"]["result_files"])
    if isinstance(manifest.get("result_files"), dict):
        candidates.append(manifest["result_files"])
    if isinstance(manifest.get("files"), dict):
        candidates.append(manifest["files"])

    for result_files in candidates:
        out: Dict[str, str] = {}
        for k, v in result_files.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        if out:
            return out
    return {}


def extract_result_files(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    manifest/manifest相当のJSONから「系列名 -> ファイル名」の辞書を取り出す。

    想定される形（いずれか）:
      - {"output": {"result_files": {...}}}
      - {"result": {"result_files": {...}}}
      - {"result_files": {...}}
      - {"files": {...}}
    """
    out = _pick_result_files(manifest)
    if out:
        return out

    error_message = extract_manifest_error(manifest)
    if error_message:
        raise ValueError(error_message)
    raise ValueError("manifest.json から result_files/files が見つかりませんでした")


def series_columns(schema: Dict[str, Any], series_name: str) -> List[str]:
    """
    schema.json から指定 series の列名配列を取り出す。

    - series.<name>.keys が [] の場合はスカラー扱いで [series_name] を返す
    """
    series = schema.get("series")
    if not isinstance(series, dict) or series_name not in series:
        raise ArtifactNotFound(f"schema.json に series.{series_name} がありません")

    spec = series[series_name]
    if not isinstance(spec, dict):
        raise ArtifactDecodeError(f"schema.json の series.{series_name} が不正です")

    keys = spec.get("keys", [])
    if keys is None:
        keys = []

    if not isinstance(keys, list):
        raise ArtifactDecodeError(f"schema.json の series.{series_name}.keys が配列ではありません")

    if len(keys) == 0:
        return [series_name]

    cols: List[str] = []
    for k in keys:
        if not isinstance(k, str):
            raise ArtifactDecodeError(
                f"schema.json の series.{series_name}.keys に文字列以外が含まれています"
            )
        cols.append(k)
    return cols


def _output_block(manifest: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(manifest.get("output"), dict):
        return manifest["output"]
    if isinstance(manifest.get("result"), dict):
        return manifest["result"]
    return manifest


@dataclass(frozen=True)
class NormalizedManifest:
    """manifest のゆれを吸収した読み取り専用ビュー。"""

    raw: Dict[str, Any]
    output: Dict[str, Any]
    result_files: Dict[str, str]
    index: Optional[Dict[str, Any]] = None
    log_file: Optional[str] = None
    builder_log_file: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, manifest: Dict[str, Any], *, require_result_files: bool = False) -> "NormalizedManifest":
        if not isinstance(manifest, dict):
            raise TypeError(f"manifest must be dict, got {type(manifest).__name__}")
        output = _output_block(manifest)
        result_files = _pick_result_files(manifest)
        if require_result_files and not result_files:
            error_message = extract_manifest_error(manifest)
            if error_message:
                raise ValueError(error_message)
            raise ValueError("manifest.json から result_files/files が見つかりませんでした")

        index = None
        if isinstance(output.get("index"), dict):
            index = output["index"]
        elif isinstance(manifest.get("index"), dict):
            index = manifest["index"]

        log_file = output.get("log_file") if isinstance(output.get("log_file"), str) else None
        builder_log = (
            output.get("builder_log_file") if isinstance(output.get("builder_log_file"), str) else None
        )

        files: Dict[str, str] = {}
        if isinstance(manifest.get("files"), dict):
            for k, v in manifest["files"].items():
                if isinstance(k, str) and isinstance(v, str):
                    files[k] = v

        return cls(
            raw=manifest,
            output=output,
            result_files=result_files,
            index=index,
            log_file=log_file,
            builder_log_file=builder_log,
            files=files,
        )

    def filename_to_key_map(self) -> Dict[str, str]:
        """ファイル名 -> download key。"""
        mapping: Dict[str, str] = {"manifest.json": "manifest"}
        if self.log_file:
            mapping[self.log_file] = "log"
        else:
            mapping["solver.log"] = "log"
        if self.builder_log_file:
            mapping[self.builder_log_file] = "builder_log"
        else:
            mapping.setdefault("builder.log", "builder_log")

        for key, filename in self.result_files.items():
            if filename:
                mapping[filename] = key
        for key, filename in self.files.items():
            if filename:
                mapping[filename] = key
        return mapping

    def resolve_download_key(self, name_or_filename: str) -> str:
        raw = name_or_filename.strip().lstrip("/")
        basename = raw.split("/")[-1]
        mapping = self.filename_to_key_map()
        known_keys = set(mapping.values())

        if basename in mapping:
            return mapping[basename]
        if raw in known_keys:
            return raw
        if basename in known_keys:
            return basename
        raise ArtifactNotFound(
            f"artifact key/filename not found in manifest whitelist: {name_or_filename!r}"
        )


__all__ = [
    "NormalizedManifest",
    "extract_result_files",
    "extract_manifest_error",
    "series_columns",
]
