"""IPython capture extension."""

from __future__ import annotations

import datetime as dt
import json
import secrets
import time
import traceback
from pathlib import Path
from typing import Any

from mil.config import get_settings

_IPYTHON_INTERNALS = frozenset(
    {
        "In",
        "Out",
        "_",
        "__",
        "___",
        "_ih",
        "_oh",
        "_dh",
        "_i",
        "_ii",
        "_iii",
        "quit",
        "exit",
        "get_ipython",
    }
)


def _is_internal(name: str) -> bool:
    return name in _IPYTHON_INTERNALS or name.startswith("_i") or name.startswith("_oh")


def _cheap_meta(obj: Any) -> dict:
    meta: dict[str, Any] = {"type": type(obj).__qualname__, "obj_id": id(obj)}
    if hasattr(obj, "shape"):
        try:
            meta["shape"] = [int(x) for x in obj.shape]
        except Exception:
            pass
    if hasattr(obj, "dtype"):
        try:
            meta["dtype"] = str(obj.dtype)
        except Exception:
            pass
    if hasattr(obj, "device"):
        try:
            meta["device"] = str(obj.device)
        except Exception:
            pass
    if hasattr(obj, "__len__") and not isinstance(obj, str):
        try:
            meta["len"] = len(obj)
        except Exception:
            pass
    try:
        text = repr(obj)
        meta["repr"] = text[:200] + ("..." if len(text) > 200 else "")
    except Exception as e:
        meta["repr"] = f"<repr error: {type(e).__name__}>"
    return meta


def _snapshot_ns(user_ns: dict) -> dict[str, tuple[int, str]]:
    return {
        name: (id(obj), type(obj).__qualname__)
        for name, obj in user_ns.items()
        if not _is_internal(name)
    }


def _diff_ns(before: dict[str, tuple[int, str]], after: dict[str, tuple[int, str]], ns: dict) -> list[dict]:
    diffs = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name)
        new = after.get(name)
        if old is None and new is not None:
            change = "new"
        elif old is not None and new is None:
            change = "deleted"
        elif old is not None and new is not None and old[0] != new[0]:
            change = "changed"
        else:
            continue
        record: dict[str, Any] = {"name": name, "change": change}
        if change != "deleted" and name in ns:
            record.update(_cheap_meta(ns[name]))
        diffs.append(record)
    return diffs


class _CaptureState:
    def __init__(self, session_id: str, log_path: Path) -> None:
        self.session_id = session_id
        self.log_path = log_path
        self.cell_index = 0
        self.pre_snapshot: dict[str, tuple[int, str]] = {}
        self.cell_start = 0.0
        self.wall_start = ""
        self.cell_source = ""

    def write(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_ipython_extension(ip) -> None:
    settings = get_settings()
    session_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(4)
    log_dir = Path(settings.cache_dir) / "sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    state = _CaptureState(session_id, log_dir / f"{session_id}.jsonl")
    state.write({"kind": "session_start", "session_id": session_id, "started_at": dt.datetime.now(dt.UTC).isoformat()})

    def pre_run_cell(info) -> None:
        state.pre_snapshot = _snapshot_ns(ip.user_ns)
        state.cell_start = time.monotonic()
        state.wall_start = dt.datetime.now(dt.UTC).isoformat()
        state.cell_source = getattr(info, "raw_cell", "") or ""

    def post_run_cell(result) -> None:
        error_info = None
        if getattr(result, "error_in_exec", None) is not None:
            exc = result.error_in_exec
            error_info = {
                "type": type(exc).__qualname__,
                "message": str(exc)[:300],
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1000:],
            }
        record = {
            "kind": "cell",
            "session_id": state.session_id,
            "cell_index": state.cell_index,
            "source": state.cell_source[:2000],
            "started_at": state.wall_start,
            "duration_ms": round((time.monotonic() - state.cell_start) * 1000, 1),
            "status": "error" if error_info else "ok",
            "error": error_info,
            "bindings": _diff_ns(state.pre_snapshot, _snapshot_ns(ip.user_ns), ip.user_ns),
        }
        state.write(record)
        state.cell_index += 1

    ip.events.register("pre_run_cell", pre_run_cell)
    ip.events.register("post_run_cell", post_run_cell)
    ip.user_ns["_mil_session"] = session_id


def unload_ipython_extension(ip) -> None:
    return None
