"""Compact result comparison helpers."""

from __future__ import annotations

from typing import Any

from mil.tools.features import FeatureHits, FeatureRanking
from mil.tools.interventions import PatchResult


def compare(a: Any, b: Any) -> dict:
    if type(a) is not type(b):
        return {"error": f"Cannot compare {type(a).__name__} with {type(b).__name__}"}
    if isinstance(a, PatchResult) and isinstance(b, PatchResult):
        return {
            "kind": "PatchResult",
            "hook": {"a": a.hook, "b": b.hook, "same": a.hook == b.hook},
            "target_mean": {
                "a": a.target_mean,
                "b": b.target_mean,
                "delta": b.target_mean - a.target_mean,
            },
            "control_mean": {"a": a.control_mean, "b": b.control_mean},
        }
    if isinstance(a, FeatureRanking) and isinstance(b, FeatureRanking):
        ids_a = {r["feature_id"] for r in a.top_features}
        ids_b = {r["feature_id"] for r in b.top_features}
        return {
            "kind": "FeatureRanking",
            "overlap_count": len(ids_a & ids_b),
            "only_in_a": sorted(ids_a - ids_b)[:5],
            "only_in_b": sorted(ids_b - ids_a)[:5],
        }
    if isinstance(a, FeatureHits) and isinstance(b, FeatureHits):
        ids_a = {r["feature_id"] for r in a.top_features}
        ids_b = {r["feature_id"] for r in b.top_features}
        return {"kind": "FeatureHits", "overlap_count": len(ids_a & ids_b)}
    return {"error": "Unsupported type for compare"}
