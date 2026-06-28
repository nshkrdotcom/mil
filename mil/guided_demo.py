"""Data contracts and small helpers for the guided negation walkthrough."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


GUIDED_DEMO_DIR = Path("artifacts/guided_demo")


@dataclass(frozen=True)
class GuidedConfig:
    model_name: str = "pythia-70m-deduped"
    hook: str = "blocks.2.hook_resid_post"
    sae_release: str = "pythia-70m-deduped-res-sm"
    sae_id: str = "blocks.2.hook_resid_post"
    attention_layer: int = 0
    attention_head: int = 0
    device: str = "cuda"


def default_guided_config() -> GuidedConfig:
    return GuidedConfig()


def build_builtin_prompt_family() -> dict:
    """Return a compact SELF-GROUND-style sentiment negation prompt family."""
    base = "The movie was"
    return {
        "family_id": "guided-sentiment-negation-movie",
        "target_concept": "sentiment_negation",
        "description": "A small negation/control family adapted from the SELF-GROUND task bank.",
        "variants": [
            {
                "label": "negated",
                "role": "target",
                "prompt": f"{base} not positive. The movie was",
                "target_token": " negative",
                "foil_token": " positive",
                "expected_behavior": "negative > positive",
            },
            {
                "label": "paraphrased_negation",
                "role": "target_paraphrase",
                "prompt": f"{base}n't positive. The movie was",
                "target_token": " negative",
                "foil_token": " positive",
                "expected_behavior": "negative > positive",
            },
            {
                "label": "non_negated_control",
                "role": "control",
                "prompt": f"{base} positive. The movie was",
                "target_token": " positive",
                "foil_token": " negative",
                "expected_behavior": "positive > negative",
            },
            {
                "label": "neutral_decoy",
                "role": "control_decoy",
                "prompt": f"{base} often positive. The movie was",
                "target_token": " positive",
                "foil_token": " negative",
                "expected_behavior": "positive > negative",
            },
        ],
    }


def config_dict(config: GuidedConfig | None = None) -> dict:
    return asdict(config or default_guided_config())


def load_guided_manifest(artifact_dir: str | Path = GUIDED_DEMO_DIR) -> dict:
    root = Path(artifact_dir)
    manifest_path = root / "demo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = {}
    for name, rel_path in manifest.get("artifacts", {}).items():
        resolved[name] = root / rel_path
    manifest["resolved_artifacts"] = resolved
    return manifest


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def causal_difference(target: list[list[float]], control: list[list[float]]) -> list[list[float | None]]:
    rows = max(len(target), len(control))
    cols = max(max((len(row) for row in target), default=0), max((len(row) for row in control), default=0))
    diff: list[list[float | None]] = []
    for row_index in range(rows):
        out_row: list[float | None] = []
        for col_index in range(cols):
            t = _get_cell(target, row_index, col_index)
            c = _get_cell(control, row_index, col_index)
            out_row.append(None if t is None or c is None else t - c)
        diff.append(out_row)
    return diff


def residual_trajectory_projection(trajectories: dict[str, np.ndarray]) -> dict:
    labels = list(trajectories)
    if not labels:
        raise ValueError("At least one trajectory is required.")
    stacked = np.concatenate([np.asarray(trajectories[label], dtype=float) for label in labels], axis=0)
    if stacked.ndim != 2:
        raise ValueError("Trajectories must be 2D arrays shaped [step, dimension].")
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    projected = {}
    for label in labels:
        points = (np.asarray(trajectories[label], dtype=float) - mean) @ basis
        projected[label] = [
            {"step": int(i), "x": float(point[0]), "y": float(point[1])}
            for i, point in enumerate(points)
        ]
    return {
        "basis_shape": [int(x) for x in basis.shape],
        "mean": mean.squeeze(0).tolist(),
        "variants": projected,
    }


def _get_cell(matrix: list[list[float]], row_index: int, col_index: int) -> float | None:
    if row_index >= len(matrix) or col_index >= len(matrix[row_index]):
        return None
    return matrix[row_index][col_index]
