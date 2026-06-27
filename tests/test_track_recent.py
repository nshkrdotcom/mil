from __future__ import annotations

import json

import mil


def test_track_and_recent_round_trip():
    ref = mil.track({"a": 1}, "obj")

    assert ref.startswith("local:obj:")
    recent = mil.recent(n=5)
    assert any(r.get("ref") == ref for r in recent)


def test_recent_session_only_reads_active_capture_log(tmp_path):
    import mil.config as cfg

    session_id = "session123"
    session_dir = cfg.get_settings().cache_dir / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / f"{session_id}.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "session_start", "session_id": session_id}),
                json.dumps({"kind": "cell", "cell_index": 0}),
                json.dumps({"kind": "tool_call", "tool": "patch"}),
            ]
        )
        + "\n"
    )

    class IP:
        user_ns = {"_mil_session": session_id}

    mil._get_active_ipython = lambda: IP()

    assert [r["kind"] for r in mil.recent(n=5, session_only=True)] == ["cell", "tool_call"]

