"""Visualization wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mil.tools.activations import ActivationsHandle
    from mil.tools.features import FeatureRanking
    from mil.tools.interventions import PatchResult


def _require(package: str) -> None:
    try:
        __import__(package)
    except ImportError as e:
        raise ImportError(f"{package} is required. Install with: pip install mil[viz]") from e


def attention(
    activations: "ActivationsHandle",
    layer: int,
    head: int,
    tokens: list[str] | None = None,
    prompt_index: int = 0,
):
    _require("circuitsvis")
    import circuitsvis as cv

    hook = f"blocks.{layer}.attn.hook_pattern"
    if hook not in activations._cache:
        raise ValueError(f"Hook {hook!r} not in activations.")
    pattern = activations._cache[hook]
    seq_len = int(pattern.shape[-1])
    if tokens is None or len(tokens) != seq_len:
        tokens = [str(i) for i in range(seq_len)]
    return cv.attention.attention_heads(
        tokens=tokens,
        attention=pattern[prompt_index, head : head + 1].detach().cpu().numpy(),
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
    fig.update_layout(
        title=f"{result.hook} - {result.mode}",
        xaxis_title="prompt index",
        yaxis_title=f"{result.metric} delta",
        barmode="group",
    )
    return fig


def token_deltas(result: "PatchResult", top_k: int = 20):
    _require("plotly")
    import importlib

    go = importlib.import_module("plotly.graph_objects")
    rows = sorted(result.token_deltas, key=lambda row: row["delta"])[:top_k]
    rows += sorted(result.token_deltas, key=lambda row: row["delta"], reverse=True)[:top_k]
    deduped = {}
    for row in rows:
        deduped[row["token_id"]] = row
    ordered = sorted(deduped.values(), key=lambda row: row["delta"])
    colors = ["#b33a3a" if row["delta"] < 0 else "#2f7d4f" for row in ordered]
    labels = [f"{row['token']!r} ({row['token_id']})" for row in ordered]
    fig = go.Figure(data=[go.Bar(x=[row["delta"] for row in ordered], y=labels, orientation="h")])
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        title="Top last-token logit deltas",
        xaxis_title="patched - clean logit delta",
        yaxis_title="token",
        height=max(420, 22 * len(ordered)),
    )
    return fig


def activation_heatmap(
    activations: "ActivationsHandle",
    hook: str,
    prompt_index: int = 0,
    *,
    tokens: list[str] | None = None,
    feature_ids: list[int] | None = None,
    sae: Any | None = None,
    max_features: int = 20,
    title: str | None = None,
):
    _require("plotly")
    import importlib

    go = importlib.import_module("plotly.graph_objects")
    if hook not in activations._cache:
        raise ValueError(f"Hook {hook!r} not in activations.")

    acts = activations._cache[hook]
    if acts.ndim != 3:
        raise ValueError("activation_heatmap expects a [prompt, position, feature] activation tensor.")
    selected = acts[prompt_index]
    if tokens is None or len(tokens) != selected.shape[0]:
        tokens = [str(i) for i in range(int(selected.shape[0]))]

    if sae is not None:
        import torch

        sae_model = getattr(sae, "_sae", sae)
        with torch.no_grad():
            encoded = sae_model.encode(selected).detach().float()
        if feature_ids is None:
            scores = encoded.abs().max(dim=0).values
            feature_ids = scores.topk(min(max_features, scores.numel())).indices.tolist()
        matrix = encoded[:, feature_ids].T.cpu().numpy()
        rows = [f"feature {int(i)}" for i in feature_ids]
        plot_title = title or f"SAE feature activations at {hook}"
    else:
        dense = selected.detach().float()
        if feature_ids is None:
            matrix = dense.norm(dim=-1).unsqueeze(0).cpu().numpy()
            rows = ["activation norm"]
        else:
            matrix = dense[:, feature_ids].T.cpu().numpy()
            rows = [f"dim {int(i)}" for i in feature_ids]
        plot_title = title or f"Activations at {hook}"

    fig = go.Figure(data=[go.Heatmap(z=matrix, x=tokens, y=rows, colorscale="Viridis")])
    fig.update_layout(
        title=plot_title,
        xaxis_title="token position",
        yaxis_title="feature",
        height=max(420, 22 * len(rows)),
    )
    return fig
