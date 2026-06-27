"""Runtime configuration for mil."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    backend: str = "local"
    device: str = "cpu"
    cache_dir: Path = field(default_factory=lambda: Path(".mil"))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend=os.getenv("MIL_BACKEND", "local"),
            device=os.getenv("MIL_DEVICE", "cpu"),
            cache_dir=Path(os.getenv("MIL_CACHE_DIR", ".mil")),
        )


_settings = Settings.from_env()


def get_settings() -> Settings:
    return _settings


def update_settings(**kwargs: object) -> None:
    global _settings
    for key, value in kwargs.items():
        if not hasattr(_settings, key):
            raise ValueError(f"Unknown setting: {key!r}")
        object.__setattr__(_settings, key, value)
