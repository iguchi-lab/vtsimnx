"""Atomic single-process job snapshots; simulation inputs are never serialized here."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)
_ID = re.compile(r"[0-9a-f]{32}")


class JobStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not _ID.fullmatch(run_id):
            raise ValueError("invalid persisted run_id")
        return self.directory / (run_id + ".json")

    def save(self, record: dict) -> None:
        path = self._path(record["run_id"])
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    def delete(self, run_id: str) -> None:
        self._path(run_id).unlink(missing_ok=True)

    def load(self):
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict) or self._path(record["run_id"]) != path:
                    raise ValueError("snapshot filename does not match run_id")
                yield record
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Cannot restore job snapshot %s", path.name)
