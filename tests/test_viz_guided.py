from __future__ import annotations

import pytest


plotly = pytest.importorskip("plotly")


def test_tokenization_view_handles_unequal_token_lengths():
    from mil.viz import tokenization_view

    fig = tokenization_view(
        {
            "variants": [
                {"label": "short", "tokens": ["A", "B"]},
                {"label": "long", "tokens": ["A", "B", "C"]},
            ]
        }
    )

    assert len(fig.data) == 1
    assert list(fig.data[0].header.values) == ["Variant", "0", "1", "2"]


def test_behavior_logit_visualization_renders_required_fields():
    from mil.viz import behavior_logit_bars

    fig = behavior_logit_bars(
        {
            "variants": [
                {
                    "label": "negated",
                    "target_token": " negative",
                    "foil_token": " positive",
                    "target_logit": 2.0,
                    "foil_logit": 1.0,
                    "margin": 1.0,
                }
            ]
        }
    )

    assert {trace.name for trace in fig.data} == {"target", "foil"}
    assert fig.layout.yaxis.title.text == "logit"


def test_causal_heatmap_has_layer_by_token_shape():
    from mil.viz import causal_heatmap

    fig = causal_heatmap(
        {
            "layers": [0, 1],
            "tokens": ["A", "B", "C"],
            "matrix": [[1.0, 2.0, 3.0], [0.5, 0.25, 0.0]],
        },
        title="target",
    )

    assert fig.data[0].z.shape == (2, 3)


def test_difference_heatmap_computes_target_control():
    from mil.viz import causal_difference_heatmap

    fig = causal_difference_heatmap(
        target={"layers": [0], "tokens": ["A", "B"], "matrix": [[2.0, 1.0]]},
        control={"layers": [0], "tokens": ["A", "B"], "matrix": [[0.5, 2.0]]},
    )

    assert fig.data[0].z.tolist() == [[1.5, -1.0]]


def test_feature_raster_preserves_feature_ids_and_prompt_variants():
    from mil.viz import feature_activation_raster

    fig = feature_activation_raster(
        {
            "rows": [
                {"kind": "candidate", "feature_id": 7, "values": [1.0, 0.0]},
                {"kind": "density_control", "feature_id": 11, "values": [0.2, 0.1]},
            ],
            "columns": ["negated:0", "control:0"],
        }
    )

    assert list(fig.data[0].y) == ["candidate 7", "density_control 11"]
    assert list(fig.data[0].x) == ["negated:0", "control:0"]
