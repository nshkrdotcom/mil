"""Mech Interp Lab public API."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from mil.config import get_settings, update_settings

_active_backend: Any = None


def _get_backend() -> Any:
    global _active_backend
    if _active_backend is None:
        settings = get_settings()
        if settings.backend == "local":
            from mil.backend.local import LocalBackend

            _active_backend = LocalBackend(settings.cache_dir)
        elif settings.backend == "wandb":
            from mil.backend.wandb_backend import WandbBackend

            _active_backend = WandbBackend()
        else:
            raise ValueError(f"Unknown backend: {settings.backend!r}")
    return _active_backend


def _get_active_ipython() -> Any:
    try:
        from IPython import get_ipython

        return get_ipython()
    except Exception:
        return None


def configure(backend: str = "local", device: str = "cpu", **kwargs: Any) -> None:
    update_settings(backend=backend, device=device, **kwargs)
    global _active_backend
    _active_backend = None


def track(obj: Any, name: str, note: str | None = None) -> str:
    try:
        raw = pickle.dumps(obj)
    except Exception:
        raw = repr(obj).encode()
    content_hash = hashlib.sha256(raw).hexdigest()

    backend = _get_backend()
    ref = backend.log_artifact(obj, name, content_hash)
    if note:
        backend.log({"kind": "note", "ref": ref, "note": note, "when": time.time()})
    return ref


def recent(n: int = 5, session_only: bool = False) -> list[dict]:
    if session_only:
        return _recent_session_records(n)
    return _get_backend().recent_runs(n)


def _recent_session_records(n: int) -> list[dict]:
    ip = _get_active_ipython()
    session_id = ip.user_ns.get("_mil_session") if ip is not None else None
    if not session_id:
        return []
    log_path = Path(get_settings().cache_dir) / "sessions" / f"{session_id}.jsonl"
    if not log_path.exists():
        return []
    records = []
    with log_path.open("rb") as f:
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
    interesting = [r for r in records if r.get("kind") in ("cell", "tool_call", "tracked", "note")]
    return interesting[-n:]
