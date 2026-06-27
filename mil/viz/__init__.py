"""Visualization wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mil.tools.activations import ActivationsHandle
    from mil.tools.features import FeatureRanking
    from mil.tools.interventions import PatchResult


def _require(package: str) -> None:
    try:
        __import__(package)
    except ImportError as e:
        raise ImportError(f"{package} is required. Install with: pip install mil[viz]") from e


def attention(activations: "ActivationsHandle", layer: int, head: int):
    _require("circuitsvis")
    import circuitsvis as cv

    hook = f"blocks.{layer}.attn.hook_pattern"
    if hook not in activations._cache:
        raise ValueError(f"Hook {hook!r} not in activations.")
    pattern = activations._cache[hook]
    return cv.attention.attention_heads(
        tokens=activations.prompts[0].split(),
        attention=pattern[0, head].cpu().numpy(),
    )


def feature_table(ranking: "FeatureRanking", top_k: int = 20):
    _require("plotly")
    import importlib
    from urllib.parse import quote_plus

    go = importlib.import_module("plotly.graph_objects")
    rows = ranking.top_features[:top_k]
    feature_ids = [str(r["feature_id"]) for r in rows]
    scores = [f"{r.get('score', r.get('activation', 0.0)):.4f}" for r in rows]
    mean_a = [f"{r.get('mean_a', 0.0):.4f}" for r in rows]
    mean_b = [f"{r.get('mean_b', 0.0):.4f}" for r in rows]
    links = [
        f"https://www.neuronpedia.org/search?q={quote_plus(f'{ranking.sae_id} feature {fid}')}"
        for fid in feature_ids
    ]
    return go.Figure(
        data=[
            go.Table(
                header={"values": ["Feature", "Score", "Mean A", "Mean B", "Neuronpedia"]},
                cells={"values": [feature_ids, scores, mean_a, mean_b, links]},
            )
        ]
    )


def patch_bar(result: "PatchResult"):
    _require("plotly")
    import importlib

    go = importlib.import_module("plotly.graph_objects")
    fig = go.Figure()
    fig.add_bar(x=list(range(len(result.target_deltas))), y=result.target_deltas, name="target")
    if result.control_deltas:
        fig.add_bar(x=list(range(len(result.control_deltas))), y=result.control_deltas, name="control")
    fig.update_layout(yaxis_title=f"{result.metric} delta", barmode="group")
    return fig
