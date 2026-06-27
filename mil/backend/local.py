"""Local backend."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

from mil.refs import make_local_ref, parse_ref


class LocalBackend:
    """Fully offline backend with file-backed artifact storage and JSONL logs."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._artifact_dir = self._cache_dir / "cache"
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._cache_dir / "run_log.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_artifact(self, obj: Any, name: str, content_hash: str) -> str:
        artifact_path = self._artifact_dir / f"{content_hash}.pkl"
        if not artifact_path.exists():
            with artifact_path.open("wb") as f:
                pickle.dump(obj, f)

        ref = make_local_ref(content_hash, name)
        self.log(
            {
                "kind": "tracked",
                "name": name,
                "ref": ref,
                "content_hash": content_hash,
                "type": type(obj).__qualname__,
                "when": time.time(),
            }
        )
        return ref

    def get_artifact(self, ref: str) -> Any:
        backend, locator = parse_ref(ref)
        if backend != "local":
            raise KeyError(f"LocalBackend cannot load ref: {ref!r}")
        parts = locator.rsplit(":", 1)
        if len(parts) != 2:
            raise KeyError(f"Cannot parse locator: {locator!r}")
        prefix = parts[1]
        matches = sorted(self._artifact_dir.glob(f"{prefix}*.pkl"))
        if not matches:
            raise KeyError(f"Artifact not found for ref: {ref!r}")
        with matches[0].open("rb") as f:
            return pickle.load(f)

    def recent_runs(self, n: int = 5) -> list[dict]:
        if not self._log_path.exists():
            return []
        records = []
        with self._log_path.open("rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(file_size, 64 * 1024)
            f.seek(-read_size, 2)
            chunk = f.read().decode("utf-8", errors="replace")
        for line in chunk.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        interesting = [r for r in records if r.get("kind") in ("tracked", "tool_call")]
        return interesting[-n:]
