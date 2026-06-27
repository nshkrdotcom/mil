"""Artifact reference helpers."""

from __future__ import annotations


def make_local_ref(content_hash: str, name: str) -> str:
    return f"local:{name}:{content_hash[:12]}"


def parse_ref(ref: str) -> tuple[str, str]:
    parts = ref.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid ref: {ref!r}. Expected 'backend:locator'.")
    return parts[0], parts[1]
