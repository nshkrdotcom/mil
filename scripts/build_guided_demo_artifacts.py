#!/usr/bin/env python3
"""Build the precomputed guided SELF-GROUND negation demo artifacts."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.guided_demo import (
    GUIDED_DEMO_DIR,
    build_builtin_prompt_family,
    causal_difference,
    config_dict,
    default_guided_config,
    residual_trajectory_projection,
)
from mil.tools import check_controls, get_activations, load_model, patch
from mil.viz import (
    attention,
    candidate_control_specificity_plot,
    causal_difference_heatmap,
    causal_heatmap,
    feature_activation_raster,
    feature_table,
    patch_before_after,
    residual_trajectory_2d,
    token_deltas,
)


def main(argv: list[str] | None = None) -> int:
    cfg = default_guided_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--hook", default=cfg.hook)
    parser.add_argument("--sae-release", default=cfg.sae_release)
    parser.add_argument("--sae-id", default=cfg.sae_id)
    parser.add_argument("--artifacts-dir", default=str(GUIDED_DEMO_DIR))
    parser.add_argument("--attention-layer", type=int, default=cfg.attention_layer)
    parser.add_argument("--attention-head", type=int, default=cfg.attention_head)
    args = parser.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    import torch

    cfg_dict = config_dict(
        dataclasses.replace(
            cfg,
            model_name=args.model_name,
            hook=args.hook,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            attention_layer=args.attention_layer,
            attention_head=args.attention_head,
            device=args.device,
        )
    )
    prompt_family = build_builtin_prompt_family()
    variants = prompt_family["variants"]
    prompts = [row["prompt"] for row in variants]

    print(f"artifact_dir={artifacts_dir}")
    print(f"model_name={args.model_name}")
    print(f"hook={args.hook}")
    print(f"device={args.device}")
    print(f"sae={args.sae_release}/{args.sae_id}")
    print(f"torch.__version__={torch.__version__}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if args.device == "cuda":
        print(f"torch.cuda.get_device_capability()={torch.cuda.get_device_capability()}")

    model = load_model(args.model_name, device=args.device)
    first_param = next(model._model.parameters())
    print(f"first_parameter_device={first_param.device}")
    if args.device == "cuda" and first_param.device.type != "cuda":
        raise SystemExit("FAIL: model parameter is not on cuda")

    tokenization = _build_tokenization(model, variants)
    behavior = _compute_behavior(model, variants)
    logit_lens = _compute_logit_lens(model, variants)
    target_heatmap = _compute_causal_heatmap(model, variants[0], args.hook)
    control_heatmap = _compute_causal_heatmap(model, variants[2], args.hook)
    causal_heatmaps = {
        "target": target_heatmap,
        "control": control_heatmap,
        "difference": {
            "layers": target_heatmap["layers"],
            "tokens": target_heatmap["tokens"],
            "matrix": causal_difference(target_heatmap["matrix"], control_heatmap["matrix"]),
        },
    }
    patch_summary, patch_result = _compute_patch_summary(model, variants, behavior, args.hook)
    residual_trajectory = _compute_residual_trajectory(model, variants)

    _write_json(artifacts_dir / "prompt_family.json", prompt_family)
    _write_json(artifacts_dir / "tokenization.json", tokenization)
    _write_json(artifacts_dir / "behavior_logits.json", behavior)
    _write_json(artifacts_dir / "logit_lens.json", logit_lens)
    _write_json(artifacts_dir / "causal_heatmap.json", causal_heatmaps)
    _write_json(artifacts_dir / "patch_summary.json", patch_summary)
    _write_json(artifacts_dir / "residual_trajectory.json", residual_trajectory)

    causal_heatmap(target_heatmap, "Target prompt causal effect").write_html(
        artifacts_dir / "causal_heatmap_target.html", include_plotlyjs="cdn"
    )
    causal_heatmap(control_heatmap, "Control prompt causal effect").write_html(
        artifacts_dir / "causal_heatmap_control.html", include_plotlyjs="cdn"
    )
    causal_difference_heatmap(target_heatmap, control_heatmap).write_html(
        artifacts_dir / "causal_heatmap_difference.html", include_plotlyjs="cdn"
    )
    patch_before_after(patch_summary).write_html(
        artifacts_dir / "patch_before_after.html", include_plotlyjs="cdn"
    )
    token_deltas(patch_result).write_html(
        artifacts_dir / "token_deltas.html", include_plotlyjs="cdn"
    )
    residual_trajectory_2d(residual_trajectory).write_html(
        artifacts_dir / "residual_trajectory.html", include_plotlyjs="cdn"
    )

    attention_path, attention_status = _export_attention(
        model=model,
        prompts=[prompts[0]],
        layer=args.attention_layer,
        head=args.attention_head,
        artifacts_dir=artifacts_dir,
    )
    print(f"attention={attention_path} ({attention_status})")

    manifest_artifacts = {
        "prompt_family": "prompt_family.json",
        "tokenization": "tokenization.json",
        "behavior_logits": "behavior_logits.json",
        "logit_lens": "logit_lens.json",
        "causal_heatmap": "causal_heatmap.json",
        "causal_heatmap_target": "causal_heatmap_target.html",
        "causal_heatmap_control": "causal_heatmap_control.html",
        "causal_heatmap_difference": "causal_heatmap_difference.html",
        "patch_summary": "patch_summary.json",
        "patch_before_after": "patch_before_after.html",
        "token_deltas": "token_deltas.html",
        "residual_trajectory": "residual_trajectory.json",
        "residual_trajectory_html": "residual_trajectory.html",
        "attention": attention_path.name,
    }

    feature_status = _maybe_export_feature_artifacts(
        model=model,
        variants=variants,
        hook=args.hook,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
        artifacts_dir=artifacts_dir,
    )
    manifest_artifacts.update(feature_status["artifacts"])
    print(f"feature_status={feature_status['status']}")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "guided_negation_walkthrough",
        "config": cfg_dict,
        "source": {
            "prompt_family": "built-in SELF-GROUND-style negation/control examples",
            "note": "The guided demo is intentionally small so the first page is immediately useful.",
        },
        "artifacts": manifest_artifacts,
    }
    _write_json(artifacts_dir / "demo_manifest.json", manifest)
    print(f"manifest={artifacts_dir / 'demo_manifest.json'}")
    return 0


def _build_tokenization(model: Any, variants: list[dict]) -> dict:
    return {
        "variants": [
            {
                "label": row["label"],
                "prompt": row["prompt"],
                "tokens": [str(tok) for tok in model._model.to_str_tokens(row["prompt"])],
            }
            for row in variants
        ]
    }


def _compute_behavior(model: Any, variants: list[dict]) -> dict:
    import torch

    rows = []
    for row in variants:
        tokens = model._model.to_tokens(row["prompt"])
        with torch.no_grad():
            logits = model._model(tokens)[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
        target_id = model._model.to_single_token(row["target_token"])
        foil_id = model._model.to_single_token(row["foil_token"])
        target_logit = float(logits[0, target_id].detach().float().item())
        foil_logit = float(logits[0, foil_id].detach().float().item())
        rows.append(
            {
                "label": row["label"],
                "role": row.get("role", ""),
                "target_token": row["target_token"],
                "foil_token": row["foil_token"],
                "target_token_id": int(target_id),
                "foil_token_id": int(foil_id),
                "target_logit": target_logit,
                "foil_logit": foil_logit,
                "target_probability": float(probs[0, target_id].item()),
                "foil_probability": float(probs[0, foil_id].item()),
                "margin": target_logit - foil_logit,
            }
        )
    return {"readout": "target - foil next-token logit margin", "variants": rows}


def _compute_logit_lens(model: Any, variants: list[dict]) -> dict:
    import torch

    n_layers = int(model._model.cfg.n_layers)
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    rows = []
    for row in variants:
        tokens = model._model.to_tokens(row["prompt"])
        target_id = model._model.to_single_token(row["target_token"])
        foil_id = model._model.to_single_token(row["foil_token"])
        margins = []
        with torch.no_grad():
            _, cache = model._model.run_with_cache(
                tokens,
                names_filter=lambda name: name in hook_names,
            )
            for hook in hook_names:
                resid = cache[hook][:, -1:, :]
                try:
                    resid = model._model.ln_final(resid)
                except Exception:
                    pass
                logits = model._model.unembed(resid)[:, -1, :]
                margins.append(float((logits[0, target_id] - logits[0, foil_id]).float().item()))
        rows.append({"label": row["label"], "margins": margins})
    return {"layers": list(range(n_layers)), "hook_points": hook_names, "variants": rows}


def _compute_causal_heatmap(model: Any, variant: dict, hook_template: str) -> dict:
    import torch

    n_layers = int(model._model.cfg.n_layers)
    tokens = model._model.to_tokens(variant["prompt"])
    token_labels = [str(tok) for tok in model._model.to_str_tokens(variant["prompt"])]
    target_id = model._model.to_single_token(variant["target_token"])
    foil_id = model._model.to_single_token(variant["foil_token"])

    with torch.no_grad():
        clean_logits = model._model(tokens)[:, -1, :]
    clean_margin = float((clean_logits[0, target_id] - clean_logits[0, foil_id]).float().item())
    matrix: list[list[float]] = []
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.{hook_template.split('.')[-1]}"
        row = []
        for pos in range(tokens.shape[1]):
            def ablate_position(activation, hook, pos=pos):
                out = activation.clone()
                out[:, pos, :] = 0
                return out

            with torch.no_grad():
                ablated_logits = model._model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_position)],
                )[:, -1, :]
            ablated_margin = float(
                (ablated_logits[0, target_id] - ablated_logits[0, foil_id]).float().item()
            )
            row.append(ablated_margin - clean_margin)
        matrix.append(row)

    return {
        "label": variant["label"],
        "prompt": variant["prompt"],
        "hook_template": hook_template,
        "metric": "ablated margin - clean margin",
        "clean_margin": clean_margin,
        "layers": list(range(n_layers)),
        "tokens": [f"{i}:{tok}" for i, tok in enumerate(token_labels)],
        "matrix": matrix,
    }


def _compute_patch_summary(
    model: Any,
    variants: list[dict],
    behavior: dict,
    hook: str,
) -> tuple[dict, Any]:
    targets = variants[:2]
    controls = variants[2:]
    result = patch(
        model,
        hook=hook,
        source=0.0,
        target_prompts=[row["prompt"] for row in targets],
        control_prompts=[row["prompt"] for row in controls],
        positions="last",
        metric="logit_diff_delta",
        target_tokens=[row["target_token"] for row in targets],
        foil_tokens=[row["foil_token"] for row in targets],
        control_tokens=[row["target_token"] for row in controls],
        control_foil_tokens=[row["foil_token"] for row in controls],
    )
    report = check_controls(result.target_mean, result.control_deltas)
    margins = {row["label"]: row["margin"] for row in behavior["variants"]}
    rows = []
    for variant, delta in zip(targets, result.target_deltas):
        clean = margins[variant["label"]]
        rows.append(
            {
                "label": variant["label"],
                "kind": "target",
                "clean_margin": clean,
                "patched_margin": clean + delta,
                "delta": delta,
            }
        )
    for variant, delta in zip(controls, result.control_deltas):
        clean = margins[variant["label"]]
        rows.append(
            {
                "label": variant["label"],
                "kind": "control",
                "clean_margin": clean,
                "patched_margin": clean + delta,
                "delta": delta,
            }
        )
    return (
        {
            "hook": hook,
            "intervention": "last-token residual ablation",
            "rows": rows,
            "control_report": dataclasses.asdict(report),
            "patch_result": dataclasses.asdict(result),
        },
        result,
    )


def _compute_residual_trajectory(model: Any, variants: list[dict]) -> dict:
    import torch

    n_layers = int(model._model.cfg.n_layers)
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    trajectories = {}
    for row in variants:
        tokens = model._model.to_tokens(row["prompt"])
        with torch.no_grad():
            _, cache = model._model.run_with_cache(
                tokens,
                names_filter=lambda name: name in hook_names,
            )
        trajectories[row["label"]] = np.stack(
            [cache[hook][0, -1, :].detach().float().cpu().numpy() for hook in hook_names],
            axis=0,
        )
    projected = residual_trajectory_projection(trajectories)
    projected["layers"] = list(range(n_layers))
    projected["token_position"] = "last"
    projected["readout"] = "hook_resid_post residual stream, shared PCA basis"
    return projected


def _maybe_export_feature_artifacts(
    *,
    model: Any,
    variants: list[dict],
    hook: str,
    sae_release: str,
    sae_id: str,
    device: str,
    artifacts_dir: Path,
) -> dict:
    try:
        from mil.tools import load_sae, rank_features
    except ImportError as exc:
        return _feature_unavailable(artifacts_dir, f"SAE unavailable: {exc}")

    try:
        target_acts = get_activations(
            model,
            prompts=[row["prompt"] for row in variants[:2]],
            hooks=[hook],
        )
        control_acts = get_activations(
            model,
            prompts=[row["prompt"] for row in variants[2:]],
            hooks=[hook],
        )
        all_acts = get_activations(
            model,
            prompts=[row["prompt"] for row in variants],
            hooks=[hook],
        )
        sae = load_sae(sae_release, sae_id, device=device)
        ranking = rank_features(sae, target_acts, control_acts, top_k=12)
        feature_table(ranking, top_k=12).write_html(
            artifacts_dir / "feature_table.html", include_plotlyjs="cdn"
        )
        raster, specificity = _build_feature_raster(model, sae, all_acts, variants, ranking)
        _write_json(artifacts_dir / "feature_specificity.json", specificity)
        feature_activation_raster(raster).write_html(
            artifacts_dir / "feature_raster.html", include_plotlyjs="cdn"
        )
        candidate_control_specificity_plot(specificity).write_html(
            artifacts_dir / "feature_specificity.html", include_plotlyjs="cdn"
        )
        return {
            "status": f"loaded {sae_release}/{sae_id}",
            "artifacts": {
                "feature_table": "feature_table.html",
                "feature_raster": "feature_raster.html",
                "feature_specificity": "feature_specificity.json",
                "feature_specificity_html": "feature_specificity.html",
            },
        }
    except Exception as exc:
        return _feature_unavailable(artifacts_dir, f"SAE unavailable: {type(exc).__name__}: {exc}")


def _build_feature_raster(
    model: Any,
    sae: Any,
    activations: Any,
    variants: list[dict],
    ranking: Any,
) -> tuple[dict, dict]:
    import torch

    hook_acts = activations._cache[sae.hook]
    variant_tokens = [model._model.to_str_tokens(row["prompt"]) for row in variants]
    encoded_by_variant = []
    columns = []
    for index, tokens in enumerate(variant_tokens):
        length = len(tokens)
        with torch.no_grad():
            encoded = sae._sae.encode(hook_acts[index : index + 1, :length, :]).squeeze(0)
        encoded_by_variant.append(encoded.detach().float())
        columns.extend(
            [f"{variants[index]['label']}:{pos}:{str(tok)}" for pos, tok in enumerate(tokens)]
        )

    target_encoded = torch.cat(encoded_by_variant[:2], dim=0)
    control_encoded = torch.cat(encoded_by_variant[2:], dim=0)
    all_encoded = torch.cat(encoded_by_variant, dim=0)
    target_score = target_encoded.max(dim=0).values
    control_score = control_encoded.max(dim=0).values
    contrast = target_score - control_score
    density = all_encoded.mean(dim=0)

    del ranking
    candidates = [int(i) for i in torch.topk(contrast, k=6).indices.tolist()]
    controls = _density_matched_controls(candidates, density, contrast, limit=len(candidates))
    selected = [("candidate", feature_id) for feature_id in candidates] + [
        ("density_control", feature_id) for feature_id in controls
    ]

    rows = []
    specificity_rows = []
    for kind, feature_id in selected:
        values = []
        for encoded in encoded_by_variant:
            values.extend(encoded[:, feature_id].detach().cpu().tolist())
        row = {
            "kind": kind,
            "feature_id": int(feature_id),
            "contrast": float(contrast[feature_id].item()),
            "mean_activation": float(density[feature_id].item()),
            "values": [float(v) for v in values],
        }
        rows.append(row)
        specificity_rows.append({k: row[k] for k in ("kind", "feature_id", "contrast", "mean_activation")})

    return (
        {
            "sae_id": sae.sae_id,
            "hook": sae.hook,
            "columns": columns,
            "rows": rows,
        },
        {
            "sae_id": sae.sae_id,
            "hook": sae.hook,
            "rows": specificity_rows,
        },
    )


def _density_matched_controls(
    candidates: list[int],
    density: Any,
    contrast: Any,
    *,
    limit: int,
) -> list[int]:
    import torch

    selected = set(candidates)
    controls = []
    abs_contrast = contrast.abs()
    for feature_id in candidates:
        score = (density - density[feature_id]).abs() + 10.0 * abs_contrast
        if selected:
            score[torch.tensor(sorted(selected), device=score.device)] = torch.inf
        chosen = int(torch.argmin(score).item())
        controls.append(chosen)
        selected.add(chosen)
        if len(controls) >= limit:
            break
    return controls


def _feature_unavailable(artifacts_dir: Path, text: str) -> dict:
    (artifacts_dir / "feature_unavailable.txt").write_text(text, encoding="utf-8")
    return {"status": text, "artifacts": {"feature_unavailable": "feature_unavailable.txt"}}


def _export_attention(
    *,
    model: Any,
    prompts: list[str],
    layer: int,
    head: int,
    artifacts_dir: Path,
) -> tuple[Path, str]:
    attention_hook = f"blocks.{layer}.attn.hook_pattern"
    activations = get_activations(model, prompts=prompts, hooks=[attention_hook])
    token_labels = model._model.to_str_tokens(prompts[0])
    try:
        render = attention(activations, layer=layer, head=head, tokens=token_labels)
        html = _html_document(_render_to_html(render))
        path = artifacts_dir / f"attention_layer{layer}_head{head}.html"
        path.write_text(html, encoding="utf-8")
        status = _screenshot_html(html, artifacts_dir / f"attention_layer{layer}_head{head}.png")
        return path, status
    except Exception as exc:
        path = artifacts_dir / f"attention_layer{layer}_head{head}.html"
        fallback = (
            "<!doctype html><html><body>"
            f"<p>CircuitsVis attention render failed: {type(exc).__name__}: {exc}</p>"
            "</body></html>"
        )
        path.write_text(fallback, encoding="utf-8")
        return path, f"fallback: {type(exc).__name__}: {exc}"


def _render_to_html(render: Any) -> str:
    if hasattr(render, "_repr_html_"):
        return str(render._repr_html_())
    if hasattr(render, "data"):
        return str(render.data)
    return str(render)


def _html_document(body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<style>body{margin:16px;font-family:sans-serif;}</style></head><body>"
        f"{body}</body></html>"
    )


def _screenshot_html(html: str, path: Path) -> str:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1200, "height": 760})
            page.set_content(html, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(3000)
            page.screenshot(path=str(path), full_page=True)
            browser.close()
        return "ok"
    except Exception as exc:
        path.with_suffix(".screenshot_error.txt").write_text(
            f"{type(exc).__name__}: {exc}",
            encoding="utf-8",
        )
        return f"failed: {type(exc).__name__}: {exc}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
