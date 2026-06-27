from __future__ import annotations

from mil.backend.local import LocalBackend


def test_local_backend_round_trip(tmp_path):
    backend = LocalBackend(tmp_path / ".mil")
    ref = backend.log_artifact({"x": 1}, "obj", "abcdef1234567890")

    assert ref == "local:obj:abcdef123456"
    assert backend.get_artifact(ref) == {"x": 1}


def test_recent_runs_contains_tracked_and_tool_calls(tmp_path):
    backend = LocalBackend(tmp_path / ".mil")
    backend.log_artifact("hello", "greeting", "hash001")
    backend.log({"kind": "tool_call", "tool": "patch", "when": 1})
    backend.log({"kind": "cell", "when": 2})

    kinds = [r["kind"] for r in backend.recent_runs(5)]
    assert kinds == ["tracked", "tool_call"]

