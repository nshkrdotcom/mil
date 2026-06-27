from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_mil(tmp_path, monkeypatch):
    monkeypatch.setenv("MIL_CACHE_DIR", str(tmp_path / ".mil"))
    import mil.config as cfg

    if hasattr(cfg, "Settings"):
        cfg._settings = cfg.Settings.from_env()
    import mil

    original_get_backend = getattr(mil, "_get_backend", None)
    original_get_ipython = getattr(mil, "_get_active_ipython", None)
    if hasattr(mil, "_active_backend"):
        mil._active_backend = None
    yield
    if original_get_backend is not None:
        mil._get_backend = original_get_backend
    if original_get_ipython is not None:
        mil._get_active_ipython = original_get_ipython
    if hasattr(mil, "_active_backend"):
        mil._active_backend = None
