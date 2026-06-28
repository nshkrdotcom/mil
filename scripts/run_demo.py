#!/usr/bin/env python3
"""Run a real pythia-70m negation/control demo on CUDA and export artifacts."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.prompt_data import find_default_prompts_file, load_prompt_tasks, make_prompt_batch
from mil.tools import check_controls, get_activations, load_model, patch
from mil.viz import activation_heatmap, attention, feature_table, patch_bar, token_deltas


DEFAULT_HOOK = "blocks.2.hook_resid_post"
DEFAULT_SAE_RELEASE = "pythia-70m-deduped-res-sm"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="pythia-70m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts-file", default=str(find_default_prompts_file() or ""))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--hook", default=DEFAULT_HOOK)
    parser.add_argument("--positions", default="last", choices=["last", "all"])
    parser.add_argument("--attention-layer", type=int, default=0)
    parser.add_argument("--attention-head", type=int, default=0)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE)
    parser.add_argument("--sae-id", default=DEFAULT_HOOK)
    parser.add_argument("--skip-sae", action="store_true")
    args = parser.parse_args(argv)

    prompts_file = Path(args.prompts_file).expanduser()
    if not prompts_file.exists():
        raise SystemExit(
            "No prompts file found. Pass --prompts-file with Paul's SELF-GROUND "
            "phase3 task-bank JSONL path."
        )

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    import torch

    print(f"torch.__version__={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if args.device == "cuda":
        print(f"torch.cuda.get_device_capability()={torch.cuda.get_device_capability()}")

    tasks = load_prompt_tasks(prompts_file, limit=args.limit)
    batch = make_prompt_batch(tasks)
    print(f"prompts_file={prompts_file}")
    print(f"loaded_tasks={len(batch.tasks)}")
    print(f"first_task_id={batch.tasks[0].id}")
    print(f"hook={args.hook}")

    model = load_model(args.model_name, device=args.device)
    first_param = next(model._model.parameters())
    print(f"model_name={model.name}")
    print(f"first_parameter_device={first_param.device}")
    if args.device == "cuda" and first_param.device.type != "cuda":
        raise SystemExit("FAIL: model parameter is not on cuda")

    attention_hook = f"blocks.{args.attention_layer}.attn.hook_pattern"
    activations = get_activations(
        model,
        prompts=batch.target_prompts,
        hooks=[args.hook, attention_hook],
    )
    control_activations = get_activations(
        model,
        prompts=batch.control_prompts,
        hooks=[args.hook],
    )
    print("activation_summary=" + json.dumps(activations.summary, indent=2, sort_keys=True))

    result = patch(
        model,
        hook=args.hook,
        source=0.0,
        target_prompts=batch.target_prompts,
        control_prompts=batch.control_prompts,
        positions=args.positions,
        metric="logit_diff_delta",
        target_tokens=batch.target_tokens,
        foil_tokens=batch.foil_tokens,
        control_tokens=batch.control_target_tokens,
        control_foil_tokens=batch.control_foil_tokens,
    )
    report = check_controls(result.target_mean, result.control_deltas)
    print("PatchResult=" + json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
    print("ControlReport=" + json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True))

    (artifacts_dir / "patch_result.json").write_text(
        json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifacts_dir / "control_report.json").write_text(
        json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    token_labels = model._model.to_str_tokens(batch.target_prompts[0])
    attn_render = attention(
        activations,
        layer=args.attention_layer,
        head=args.attention_head,
        tokens=token_labels,
    )
    attention_html = _html_document(_render_to_html(attn_render))
    attention_path = artifacts_dir / "attention_layer0_head0.html"
    attention_path.write_text(attention_html, encoding="utf-8")
    screenshot_path = artifacts_dir / "attention_layer0_head0.png"
    screenshot_status = _screenshot_html(attention_html, screenshot_path)

    patch_path = artifacts_dir / "patch_bar.html"
    patch_bar(result).write_html(patch_path, include_plotlyjs="cdn")
    token_path = artifacts_dir / "token_deltas.html"
    token_deltas(result).write_html(token_path, include_plotlyjs="cdn")
    heatmap_path = artifacts_dir / "activation_heatmap.html"
    activation_heatmap(
        activations,
        args.hook,
        prompt_index=0,
        tokens=token_labels,
        title=f"Residual activations: {args.hook}",
    ).write_html(heatmap_path, include_plotlyjs="cdn")

    print(f"artifact_attention_html={attention_path}")
    print(f"artifact_attention_screenshot={screenshot_path} ({screenshot_status})")
    print(f"artifact_patch_bar={patch_path}")
    print(f"artifact_token_deltas={token_path}")
    print(f"artifact_activation_heatmap={heatmap_path}")

    sae_status = _maybe_export_sae_artifacts(
        args=args,
        activations=activations,
        control_activations=control_activations,
        token_labels=token_labels,
        artifacts_dir=artifacts_dir,
    )
    print(f"sae_status={sae_status}")
    return 0


def _maybe_export_sae_artifacts(
    *,
    args: argparse.Namespace,
    activations: Any,
    control_activations: Any,
    token_labels: list[str],
    artifacts_dir: Path,
) -> str:
    if args.skip_sae:
        return "skipped by --skip-sae"
    if "deduped" not in args.model_name and "deduped" in args.sae_release:
        text = (
            f"No matching public SAE wired for model {args.model_name!r} at {args.hook!r}. "
            f"SAELens release {args.sae_release!r} is for pythia-70m-deduped, so the demo "
            "does not encode non-deduped activations with it."
        )
        (artifacts_dir / "sae_unavailable.txt").write_text(text, encoding="utf-8")
        return text
    try:
        from mil.tools import load_sae, rank_features
    except ImportError as exc:
        text = f"SAE unavailable: {exc}"
        (artifacts_dir / "sae_unavailable.txt").write_text(text, encoding="utf-8")
        return text

    try:
        sae = load_sae(args.sae_release, args.sae_id, device=args.device)
        ranking = rank_features(sae, activations, control_activations, top_k=20)
        feature_table(ranking).write_html(
            artifacts_dir / "feature_table.html", include_plotlyjs="cdn"
        )
        activation_heatmap(
            activations,
            args.hook,
            prompt_index=0,
            tokens=token_labels,
            sae=sae,
            max_features=12,
            title=f"SAE feature activations: {args.sae_id}",
        ).write_html(artifacts_dir / "sae_activation_heatmap.html", include_plotlyjs="cdn")
    except Exception as exc:
        text = f"SAE unavailable: {type(exc).__name__}: {exc}"
        (artifacts_dir / "sae_unavailable.txt").write_text(text, encoding="utf-8")
        return text
    return (
        f"loaded {args.sae_release}/{args.sae_id}; exported feature_table.html and "
        "sae_activation_heatmap.html"
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
