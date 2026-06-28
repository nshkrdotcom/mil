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
