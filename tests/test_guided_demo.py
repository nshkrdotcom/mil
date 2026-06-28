from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_guided_default_config_is_sae_compatible():
    from mil.guided_demo import default_guided_config

    cfg = default_guided_config()

    assert cfg.model_name == "pythia-70m-deduped"
    assert cfg.sae_release == "pythia-70m-deduped-res-sm"
    assert cfg.sae_id == cfg.hook
    assert "deduped" in cfg.model_name


def test_prompt_family_preserves_variant_labels():
    from mil.guided_demo import build_builtin_prompt_family

    family = build_builtin_prompt_family()
    labels = [variant["label"] for variant in family["variants"]]

    assert labels == ["negated", "paraphrased_negation", "non_negated_control", "neutral_decoy"]
    assert family["target_concept"] == "sentiment_negation"


def test_guided_manifest_loads(tmp_path: Path):
    from mil.guided_demo import load_guided_manifest

    manifest = {
        "config": {"model_name": "pythia-70m-deduped"},
        "artifacts": {"prompt_family": "prompt_family.json"},
    }
    (tmp_path / "demo_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "prompt_family.json").write_text("{}", encoding="utf-8")

    loaded = load_guided_manifest(tmp_path)

    assert loaded["config"]["model_name"] == "pythia-70m-deduped"
    assert loaded["resolved_artifacts"]["prompt_family"] == tmp_path / "prompt_family.json"


def test_app_helper_loads_precomputed_artifacts_without_cuda(tmp_path: Path):
    from apps.explorer import load_guided_artifacts

    (tmp_path / "demo_manifest.json").write_text(
        json.dumps({"artifacts": {"prompt_family": "prompt_family.json"}}),
        encoding="utf-8",
    )
    (tmp_path / "prompt_family.json").write_text(json.dumps({"variants": []}), encoding="utf-8")

    loaded = load_guided_artifacts(tmp_path)

    assert loaded["prompt_family"]["variants"] == []


def test_readme_guided_command_paths_exist():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "./scripts/bootstrap_uv.sh" in text
    assert ".venv/bin/streamlit run apps/explorer.py -- --demo guided" in text
    assert ".venv/bin/streamlit run apps/explorer.py -- --demo guided --view grid" in text
    assert Path("scripts/bootstrap_uv.sh").exists()
    assert Path("apps/explorer.py").exists()


def test_residual_projection_uses_one_shared_basis():
    from mil.guided_demo import residual_trajectory_projection

    trajectories = {
        "negated": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "control": np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
    }

    projected = residual_trajectory_projection(trajectories)

    assert projected["basis_shape"] == [3, 2]
    assert set(projected["variants"]) == {"negated", "control"}
    assert len(projected["variants"]["negated"]) == 2
    assert len(projected["variants"]["control"]) == 2


def test_causal_difference_pads_and_subtracts():
    from mil.guided_demo import causal_difference

    target = [[1.0, 2.0, 3.0], [0.5, 0.0, -0.5]]
    control = [[0.5, 1.0], [0.25, -0.25]]

    diff = causal_difference(target, control)

    assert diff == [[0.5, 1.0, None], [0.25, 0.25, None]]


def test_parse_app_args_accepts_grid_view():
    from apps.explorer import parse_app_args

    args = parse_app_args(["--demo", "guided", "--view", "grid"])

    assert args.demo == "guided"
    assert args.view == "grid"


def test_parse_app_args_scroll_is_default():
    from apps.explorer import parse_app_args

    args = parse_app_args([])

    assert args.view == "scroll"


def test_render_guided_summary_supports_compact_signature():
    import inspect

    from apps.explorer import _render_guided_summary

    sig = inspect.signature(_render_guided_summary)
    assert "compact" in sig.parameters


def test_behavior_table_rows_are_compact():
    from apps.explorer import _behavior_table_rows

    rows = _behavior_table_rows(
        {
            "variants": [
                {
                    "label": "negated",
                    "target_token": " negative",
                    "foil_token": " positive",
                    "margin": 0.84676,
                }
            ]
        }
    )

    assert rows == [
        {
            "variant": "negated",
            "target": " negative",
            "foil": " positive",
            "margin": 0.847,
        }
    ]


def test_feature_strip_summary_separates_candidate_and_control():
    from apps.explorer import _feature_strip_summary

    summary = _feature_strip_summary(
        {
            "rows": [
                {"kind": "candidate", "feature_id": 1, "contrast": 3.5},
                {"kind": "density_control", "feature_id": 2, "contrast": 0.2},
            ]
        }
    )

    assert summary["best_feature_id"] == "1"
    assert summary["best_contrast"] == "3.50"
    assert summary["control_max"] == "0.20"


def test_feature_strip_summary_handles_no_candidates():
    from apps.explorer import _feature_strip_summary

    summary = _feature_strip_summary({"rows": []})

    assert summary["best_feature_id"] == "—"
    assert summary["best_contrast"] == "—"
    assert summary["control_max"] == "0.00"
