from __future__ import annotations

import json


def test_capture_records_plain_namespace_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("MIL_CACHE_DIR", str(tmp_path / ".mil"))
    import mil.config as cfg

    cfg._settings = cfg.Settings.from_env()

    from mil.capture import load_ipython_extension

    class Events:
        def __init__(self):
            self.callbacks = {}

        def register(self, name, callback):
            self.callbacks[name] = callback

    class Shell:
        def __init__(self):
            self.user_ns = {}
            self.events = Events()

    class Info:
        raw_cell = "x = [1, 2, 3]"

    class Result:
        error_in_exec = None

    shell = Shell()
    load_ipython_extension(shell)
    shell.events.callbacks["pre_run_cell"](Info())
    shell.user_ns["x"] = [1, 2, 3]
    shell.events.callbacks["post_run_cell"](Result())

    files = list((tmp_path / ".mil" / "sessions").glob("*.jsonl"))
    assert files
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    cell = [r for r in records if r["kind"] == "cell"][-1]
    assert cell["status"] == "ok"
    assert any(b["name"] == "x" and b["change"] == "new" for b in cell["bindings"])
