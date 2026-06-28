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
        raise ImportError(
            f"{package} is required. Install with: "
            "uv pip install --python .venv/bin/python -e '.[viz]'"
        ) from e


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


def _go():
    _require("plotly")
    import importlib

    return importlib.import_module("plotly.graph_objects")


def _nan_array(matrix: Any):
    import numpy as np

    return np.array(
        [[np.nan if cell is None else cell for cell in row] for row in matrix],
        dtype=float,
    )


def _compact_layout(fig: Any, **kwargs: Any) -> Any:
    """Apply a consistent compact, dark-background layout to a Plotly figure.

    Parameters
    ----------
    fig:
        A ``plotly.graph_objects.Figure`` instance to update in-place.
    **kwargs:
        Additional ``update_layout`` kwargs (title, xaxis_title, etc.) that
        supplement (not replace) the shared compact defaults.

    Returns
    -------
    The same figure, updated in-place.
    """
    defaults: dict[str, Any] = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.85)",
        font=dict(family="Inter, ui-sans-serif, sans-serif", size=11, color="#e5e7eb"),
        margin=dict(l=36, r=16, t=36, b=32),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        coloraxis_colorbar=dict(thickness=10, len=0.7),
        height=260,
    )
    defaults.update(kwargs)
    fig.update_layout(**defaults)
    return fig


def prompt_family_table(prompt_family: dict):
    go = _go()
    rows = prompt_family["variants"]
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Variant", "Role", "Prompt", "Target", "Foil"],
                    "fill_color": "#1e293b",
                    "font": {"color": "#e5e7eb", "size": 11},
                    "line_color": "rgba(148,163,184,0.2)",
                },
                cells={
                    "values": [
                        [row["label"] for row in rows],
                        [row.get("role", "") for row in rows],
                        [row["prompt"] for row in rows],
                        [row.get("target_token", "") for row in rows],
                        [row.get("foil_token", "") for row in rows],
                    ],
                    "fill_color": "#111827",
                    "font": {"color": "#94a3b8", "size": 10},
                    "line_color": "rgba(148,163,184,0.12)",
                },
            )
        ]
    )
    return _compact_layout(fig, title="Prompt family", height=220)


def tokenization_view(tokenization: dict):
    go = _go()
    variants = tokenization["variants"]
    max_len = max((len(row["tokens"]) for row in variants), default=0)
    headers = ["Variant"] + [str(i) for i in range(max_len)]
    values = [[row["label"] for row in variants]]
    for pos in range(max_len):
        values.append([row["tokens"][pos] if pos < len(row["tokens"]) else "" for row in variants])
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": headers,
                    "fill_color": "#1e293b",
                    "font": {"color": "#e5e7eb", "size": 11},
                    "line_color": "rgba(148,163,184,0.2)",
                },
                cells={
                    "values": values,
                    "fill_color": "#111827",
                    "font": {"color": "#94a3b8", "size": 10},
                    "line_color": "rgba(148,163,184,0.12)",
                },
            )
        ]
    )
    return _compact_layout(fig, title="Tokenization by prompt variant", height=220)


def behavior_logit_bars(behavior: dict):
    go = _go()
    variants = behavior["variants"]
    x = [row["label"] for row in variants]
    fig = go.Figure()
    fig.add_bar(
        x=x,
        y=[row["target_logit"] for row in variants],
        name="target",
        text=[row["target_token"] for row in variants],
        marker_color="#38bdf8",
    )
    fig.add_bar(
        x=x,
        y=[row["foil_logit"] for row in variants],
        name="foil",
        text=[row["foil_token"] for row in variants],
        marker_color="#fb7185",
    )
    return _compact_layout(
        fig,
        title="Target vs foil logits",
        yaxis_title="logit",
        barmode="group",
    )


def logit_margin_curve(logit_lens: dict):
    go = _go()
    fig = go.Figure()
    layers = logit_lens["layers"]
    for row in logit_lens["variants"]:
        fig.add_scatter(x=layers, y=row["margins"], mode="lines+markers", name=row["label"])
    return _compact_layout(
        fig,
        title="Logit-lens margin over depth",
        xaxis_title="layer",
        yaxis_title="target − foil margin",
    )


def causal_heatmap(data: dict, title: str | None = None):
    go = _go()
    z = _nan_array(data["matrix"])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=data["tokens"],
                y=[str(layer) for layer in data["layers"]],
                colorscale="RdBu",
                zmid=0,
            )
        ]
    )
    return _compact_layout(
        fig,
        title=title or data.get("title", "Causal heatmap"),
        xaxis_title="token",
        yaxis_title="layer",
    )


def causal_difference_heatmap(target: dict, control: dict):
    from mil.guided_demo import causal_difference

    diff = {
        "layers": target["layers"],
        "tokens": target["tokens"],
        "matrix": causal_difference(target["matrix"], control["matrix"]),
    }
    return causal_heatmap(diff, title="Target minus control causal effect")


def feature_activation_raster(raster: dict):
    go = _go()
    y = [f"{row['kind']} {row['feature_id']}" for row in raster["rows"]]
    z = _nan_array([row["values"] for row in raster["rows"]])
    fig = go.Figure(
        data=[go.Heatmap(z=z, x=raster["columns"], y=y, colorscale="Viridis")]
    )
    fig.update_layout(
        title="SAE feature activation raster",
        xaxis_title="prompt variant x token",
        yaxis_title="feature",
        height=max(420, 24 * len(y)),
    )
    return fig


def candidate_control_specificity_plot(specificity: dict):
    go = _go()
    rows = specificity["rows"]
    colors = ["#34d399" if row["kind"] == "candidate" else "#64748b" for row in rows]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[str(row["feature_id"]) for row in rows],
                y=[row["contrast"] for row in rows],
                marker_color=colors,
            )
        ]
    )
    return _compact_layout(
        fig,
        title="Candidate vs control feature contrast",
        xaxis_title="feature id",
        yaxis_title="negation−control contrast",
    )


def residual_trajectory_2d(trajectory: dict):
    go = _go()
    fig = go.Figure()
    for label, points in trajectory["variants"].items():
        fig.add_scatter(
            x=[point["x"] for point in points],
            y=[point["y"] for point in points],
            mode="lines+markers+text",
            text=[str(point["step"]) for point in points],
            textposition="top center",
            name=label,
        )
    return _compact_layout(
        fig,
        title="Residual trajectory (shared PCA)",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )


def residual_trajectory_3d(trajectory: dict):
    go = _go()
    fig = go.Figure()
    for label, points in trajectory["variants"].items():
        z = [point.get("z", point["step"]) for point in points]
        fig.add_scatter3d(
            x=[point["x"] for point in points],
            y=[point["y"] for point in points],
            z=z,
            mode="lines+markers",
            name=label,
        )
    fig.update_layout(title="Residual stream trajectory")
    return fig


def patch_before_after(patch_summary: dict):
    go = _go()
    rows = patch_summary["rows"]
    fig = go.Figure()
    fig.add_bar(
        x=[row["label"] for row in rows],
        y=[row["clean_margin"] for row in rows],
        name="clean",
        marker_color="#38bdf8",
    )
    fig.add_bar(
        x=[row["label"] for row in rows],
        y=[row["patched_margin"] for row in rows],
        name="ablated",
        marker_color="#f59e0b",
    )
    return _compact_layout(
        fig,
        title="Clean vs ablated margin",
        yaxis_title="target − foil margin",
        barmode="group",
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
