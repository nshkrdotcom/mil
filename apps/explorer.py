from __future__ import annotations

# ruff: noqa: E402

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.guided_demo import GUIDED_DEMO_DIR, load_json
from mil.prompt_data import (
    builtin_tasks,
    find_default_prompts_file,
    load_prompt_tasks,
    make_prompt_batch,
)


DEFAULT_HOOK = "blocks.2.hook_resid_post"
DEFAULT_SAE_RELEASE = "pythia-70m-deduped-res-sm"


def load_guided_artifacts(artifact_dir: str | Path = GUIDED_DEMO_DIR) -> dict:
    root = Path(artifact_dir)
    manifest_path = root / "demo_manifest.json"
    manifest = load_json(manifest_path)
    loaded = {"manifest": manifest, "root": root}
    for key, rel_path in manifest.get("artifacts", {}).items():
        path = root / rel_path
        if path.suffix == ".json":
            loaded[key] = load_json(path)
        else:
            loaded[key] = path
    return loaded


def parse_app_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", choices=["guided", "free"], default="guided")
    parser.add_argument("--artifacts-dir", default=str(GUIDED_DEMO_DIR))
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    st, components = _require_streamlit()
    args = parse_app_args(argv)
    st.set_page_config(page_title="mil explorer", layout="wide")

    mode_options = ["Guided negation walkthrough", "Free exploration"]
    default_index = 0 if args.demo == "guided" else 1
    with st.sidebar:
        mode = st.selectbox("Demo mode", mode_options, index=default_index)

    if mode == "Guided negation walkthrough":
        render_guided_walkthrough(st, components, Path(args.artifacts_dir))
    else:
        render_free_exploration(st, components)


def render_guided_walkthrough(st: Any, components: Any, artifact_dir: Path) -> None:
    from mil.viz import (
        behavior_logit_bars,
        candidate_control_specificity_plot,
        causal_difference_heatmap,
        causal_heatmap,
        logit_margin_curve,
        patch_before_after,
        prompt_family_table,
        residual_trajectory_2d,
        tokenization_view,
    )

    with st.sidebar:
        st.markdown("### Researcher workflow")
        for label, anchor in [
            ("1. Behavior", "what-does-the-model-predict"),
            ("2. Localization", "where-is-the-causal-effect-localized"),
            ("3. Feature specificity", "which-features-components-are-specific"),
            ("4. Intervention", "what-changes-when-we-intervene"),
            ("5. Geometry", "optional-geometry-view-how-representations-move-through-depth"),
            ("6. Attention drill-down", "attention-drill-down"),
            ("7. What to try next", "what-to-try-next"),
        ]:
            st.markdown(f"- [{label}](#{anchor})")

    try:
        artifacts = load_guided_artifacts(artifact_dir)
    except Exception as exc:
        st.title("mil guided negation walkthrough")
        st.error(f"Guided demo artifacts were not found: {type(exc).__name__}: {exc}")
        st.markdown(
            "Build them with:\n\n"
            "```bash\n"
            ".venv/bin/python scripts/build_guided_demo_artifacts.py\n"
            "```"
        )
        return

    manifest = artifacts["manifest"]
    cfg = manifest["config"]
    st.title("Guided SELF-GROUND negation walkthrough")
    st.caption(
        f"Precomputed demo: `{cfg['model_name']}` at `{cfg['hook']}` with "
        f"`{cfg['sae_release']}/{cfg['sae_id']}`"
    )
    _render_guided_summary(st, artifacts)

    st.header("What question are we asking?")
    st.markdown(
        "We want to see whether the model has internal components that distinguish a "
        "negated statement from a non-negated control, and whether those components "
        "affect the next-token prediction. The controls matter: a component that also "
        "moves matched non-negated prompts is less specific to negation."
    )
    st.plotly_chart(prompt_family_table(artifacts["prompt_family"]), width="stretch")
    st.plotly_chart(tokenization_view(artifacts["tokenization"]), width="stretch")
    st.markdown(
        "Tokenization determines the positions available for hooks and interventions. "
        "Unequal token lengths are normal; compare positions by label and context, not "
        "only by column number."
    )

    st.header("What does the model predict?")
    st.markdown(
        "The behavioral readout is a target-vs-foil logit margin. A positive margin "
        "means the model assigns the target token a higher next-token logit than the "
        "foil. This is the quantity the later causal interventions try to move."
    )
    st.plotly_chart(behavior_logit_bars(artifacts["behavior_logits"]), width="stretch")
    st.dataframe(artifacts["behavior_logits"]["variants"], width="stretch")

    st.header("Where in the network does the answer emerge?")
    st.markdown(
        "A logit-lens curve projects intermediate residual streams through the final "
        "unembedding. It suggests depths where the readout becomes linearly available. "
        "This is not proof of a mechanism; it tells researchers where to start looking."
    )
    st.plotly_chart(logit_margin_curve(artifacts["logit_lens"]), width="stretch")

    st.header("Where is the causal effect localized?")
    st.markdown(
        "Absolute activation size is not enough. These heatmaps ablate one residual "
        "stream layer/token cell at a time and measure the change in the behavioral "
        "margin. Strong target effects that are absent from controls are better "
        "candidate sites for mechanism work."
    )
    left, mid, right = st.columns(3)
    with left:
        st.plotly_chart(
            causal_heatmap(artifacts["causal_heatmap"]["target"], "Target prompt"),
            width="stretch",
        )
    with mid:
        st.plotly_chart(
            causal_heatmap(artifacts["causal_heatmap"]["control"], "Control prompt"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            causal_difference_heatmap(
                artifacts["causal_heatmap"]["target"], artifacts["causal_heatmap"]["control"]
            ),
            width="stretch",
        )

    st.header("Which features/components are specific?")
    st.markdown(
        "A feature is more interesting if it fires on negation and paraphrased negation "
        "but not on the matched non-negated or decoy prompts. A feature is less "
        "interesting if density-matched controls behave similarly."
    )
    if "feature_table" in artifacts:
        _embed_html(components, artifacts["feature_table"], height=360)
    if "feature_raster" in artifacts:
        _embed_html(components, artifacts["feature_raster"], height=520)
    if "feature_specificity" in artifacts:
        st.plotly_chart(
            candidate_control_specificity_plot(artifacts["feature_specificity"]),
            width="stretch",
        )
    elif "feature_unavailable" in artifacts:
        st.info(Path(artifacts["feature_unavailable"]).read_text(encoding="utf-8"))

    st.header("What changes when we intervene?")
    st.markdown(
        "Interventions move the analysis from correlation toward causality. Here the "
        "clean margin is compared with the margin after ablating the selected residual "
        "stream site. The top-token deltas show broader side effects of the intervention."
    )
    report = artifacts["patch_summary"]["control_report"]
    control_message = (
        f"Control leakiness ratio {report['ratio']:.3f}: "
        f"control max {report['control_max']:.4f}, target delta {report['target_delta']:.4f}."
    )
    if report["flagged"]:
        st.error(f"FLAGGED - {control_message}")
    else:
        st.info(f"Controls below threshold - {control_message}")
    st.plotly_chart(patch_before_after(artifacts["patch_summary"]), width="stretch")
    _embed_html(components, artifacts["token_deltas"], height=520)

    st.header("Optional geometry view: how representations move through depth")
    st.markdown(
        "The trajectory view projects selected residual-stream states into one shared "
        "PCA basis. Geometry views are diagnostic, not standalone proof. They help "
        "show whether prompt variants separate over depth before committing to more "
        "expensive interventions."
    )
    st.plotly_chart(residual_trajectory_2d(artifacts["residual_trajectory"]), width="stretch")

    st.header("Attention drill-down")
    st.markdown(
        "Attention patterns are not explanations by themselves. Use them after causal "
        "localization to inspect routing hypotheses: which source tokens could be "
        "supplying information to a high-effect layer/token cell?"
    )
    _embed_html(components, artifacts["attention"], height=720)

    st.header("What to try next")
    st.markdown(
        "- Change prompt family and compare lexical negation with paraphrased negation.\n"
        "- Inspect the highest-effect cells in the layer/token causal heatmap.\n"
        "- Switch hooks and rerun with matched controls.\n"
        "- Inspect candidate SAE features against density-matched controls.\n"
        "- Export artifacts before changing the model or prompt source."
    )


def _render_guided_summary(st: Any, artifacts: dict) -> None:
    behavior_rows = artifacts["behavior_logits"]["variants"]
    negated = behavior_rows[0]
    patch_rows = artifacts["patch_summary"]["rows"]
    report = artifacts["patch_summary"]["control_report"]
    feature_rows = artifacts.get("feature_specificity", {}).get("rows", [])
    candidate_count = len([row for row in feature_rows if row.get("kind") == "candidate"])
    cols = st.columns(4)
    cols[0].metric("Negated margin", f"{negated['margin']:.3f}")
    cols[1].metric("Ablation delta", f"{patch_rows[0]['delta']:.3f}")
    cols[2].metric(
        "Control ratio",
        f"{report['ratio']:.3f}",
        delta="FLAGGED" if report["flagged"] else "below threshold",
        delta_color="inverse" if report["flagged"] else "normal",
    )
    cols[3].metric("SAE candidates", str(candidate_count))


def render_free_exploration(st: Any, components: Any) -> None:
    from mil.gpu import run_cuda_matmul_check
    from mil.tools import check_controls, get_activations, load_model, patch
    from mil.viz import activation_heatmap, attention, feature_table, patch_bar, token_deltas

    @st.cache_resource(show_spinner=True)
    def cached_model(model_name: str, device: str):
        return load_model(model_name, device=device)

    @st.cache_data(show_spinner=False)
    def cached_gpu_gate() -> tuple[bool, str]:
        result = run_cuda_matmul_check(size=1024, slow_threshold_ms=250.0)
        return result.ok, "\n".join(result.messages)

    def resolve_device(requested: str) -> str:
        if requested != "cuda":
            return "cpu"
        ok, output = cached_gpu_gate()
        if ok:
            return "cuda"
        st.warning("CUDA was requested but failed the GPU gate. Falling back to CPU.")
        with st.expander("GPU gate output"):
            st.code(output)
        return "cpu"

    default_prompts_file = find_default_prompts_file()
    with st.sidebar:
        requested_device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        device = resolve_device(requested_device)
        st.caption(f"Using `{device}`")
        model_name = st.text_input("Model", "pythia-70m-deduped")
        source_options = ["SELF-GROUND file", "Built-in examples", "Custom file"]
        source_index = 0 if default_prompts_file is not None else 1
        prompt_source = st.selectbox("Prompt source", source_options, index=source_index)
        if prompt_source == "Built-in examples":
            prompts_file = ""
        elif prompt_source == "SELF-GROUND file":
            prompts_file = str(default_prompts_file or "")
            st.text_input("Prompt file", prompts_file, disabled=True)
        else:
            prompts_file = st.text_input("Prompt file", str(default_prompts_file or ""))
        limit = st.slider("Prompt pairs", min_value=1, max_value=32, value=8)

    st.title("Free exploration")
    tasks = _load_tasks(prompt_source, prompts_file, limit)
    batch = make_prompt_batch(tasks)
    model = cached_model(model_name, device)
    st.caption(f"{model_name} parameter device: `{next(model._model.parameters()).device}`")
    hook = st.text_input("Hook", DEFAULT_HOOK)

    st.header("Attention")
    layer = st.number_input("Layer", min_value=0, max_value=12, value=0, step=1)
    head = st.number_input("Head", min_value=0, max_value=32, value=0, step=1)
    attention_hook = f"blocks.{layer}.attn.hook_pattern"
    attention_acts = get_activations(model, prompts=batch.target_prompts, hooks=[attention_hook])
    token_labels = model._model.to_str_tokens(batch.target_prompts[0])
    try:
        render = attention(attention_acts, int(layer), int(head), tokens=token_labels)
        components.html(_render_to_html(render), height=650, scrolling=True)
    except Exception as exc:
        st.warning(f"CircuitsVis render failed; inspect raw activations instead. {exc}")

    st.header("Patch / Controls")
    positions = st.selectbox("Positions", ["last", "all"], index=0)
    threshold = st.slider("Control threshold", min_value=0.1, max_value=1.5, value=0.8, step=0.05)
    if st.button("Run patch", type="primary"):
        activations = get_activations(model, prompts=batch.target_prompts, hooks=[hook])
        result = patch(
            model,
            hook=hook,
            source=0.0,
            target_prompts=batch.target_prompts,
            control_prompts=batch.control_prompts,
            positions=positions,
            metric="logit_diff_delta",
            target_tokens=batch.target_tokens,
            foil_tokens=batch.foil_tokens,
            control_tokens=batch.control_target_tokens,
            control_foil_tokens=batch.control_foil_tokens,
        )
        report = check_controls(result.target_mean, result.control_deltas, threshold=threshold)
        st.session_state["patch_result"] = result
        st.session_state["control_report"] = report
        st.session_state["hook_activations"] = activations

    if "patch_result" in st.session_state:
        result = st.session_state["patch_result"]
        report = st.session_state["control_report"]
        status = "FLAGGED" if report.flagged else "not flagged"
        st.error(
            f"Control leakiness {status}: ratio {report.ratio:.3f}, "
            f"control max {report.control_max:.4f}, target mean {report.target_delta:.4f}"
        ) if report.flagged else st.success(
            f"Control leakiness {status}: ratio {report.ratio:.3f}, "
            f"control max {report.control_max:.4f}, target mean {report.target_delta:.4f}"
        )
        st.plotly_chart(patch_bar(result), width="stretch")
        st.plotly_chart(token_deltas(result), width="stretch")
        st.plotly_chart(
            activation_heatmap(
                st.session_state["hook_activations"],
                hook,
                prompt_index=0,
                tokens=model._model.to_str_tokens(batch.target_prompts[0]),
            ),
            width="stretch",
        )
        with st.expander("PatchResult / ControlReport"):
            st.json({"PatchResult": dataclasses.asdict(result), "ControlReport": dataclasses.asdict(report)})

    st.header("SAE Features")
    if "deduped" not in model_name:
        st.info(
            "Use `pythia-70m-deduped` with "
            "`pythia-70m-deduped-res-sm/blocks.2.hook_resid_post` for SAE views."
        )
    elif st.button("Rank SAE features"):
        from mil.tools import load_sae, rank_features

        sae = load_sae(DEFAULT_SAE_RELEASE, hook, device=device)
        target_acts = get_activations(model, prompts=batch.target_prompts, hooks=[hook])
        control_acts = get_activations(model, prompts=batch.control_prompts, hooks=[hook])
        ranking = rank_features(sae, target_acts, control_acts, top_k=20)
        st.plotly_chart(feature_table(ranking), width="stretch")


def _load_tasks(source: str, file_path: str, limit: int):
    if source == "Built-in examples":
        return builtin_tasks(limit)
    return load_prompt_tasks(Path(file_path).expanduser(), limit=limit)


def _render_to_html(render: Any) -> str:
    if hasattr(render, "_repr_html_"):
        return str(render._repr_html_())
    if hasattr(render, "data"):
        return str(render.data)
    return str(render)


def _embed_html(components: Any, path: str | Path, *, height: int) -> None:
    html = Path(path).read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=True)


def _require_streamlit():
    try:
        import streamlit as st
        import streamlit.components.v1 as components
    except ImportError as exc:  # pragma: no cover - exercised by users without app extra.
        raise ImportError(
            "Streamlit is required. Install with: "
            "uv pip install --python .venv/bin/python -e '.[app]'"
        ) from exc
    return st, components


if __name__ == "__main__":
    main(sys.argv[1:])
