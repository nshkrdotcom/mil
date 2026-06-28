from __future__ import annotations

# ruff: noqa: E402

import dataclasses
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError as exc:  # pragma: no cover - exercised by users without app extra.
    raise ImportError("Streamlit is required. Install with: pip install mil[app]") from exc

from mil.gpu import run_cuda_matmul_check
from mil.prompt_data import (
    builtin_tasks,
    find_default_prompts_file,
    load_prompt_tasks,
    make_prompt_batch,
)
from mil.tools import check_controls, get_activations, load_model, patch
from mil.viz import activation_heatmap, attention, feature_table, patch_bar, token_deltas


DEFAULT_HOOK = "blocks.2.hook_resid_post"
DEFAULT_SAE_RELEASE = "pythia-70m-deduped-res-sm"


st.set_page_config(page_title="mil explorer", layout="wide")


@st.cache_resource(show_spinner=True)
def _cached_model(model_name: str, device: str):
    return load_model(model_name, device=device)


@st.cache_data(show_spinner=False)
def _cached_gpu_gate() -> tuple[bool, str]:
    result = run_cuda_matmul_check(size=1024, slow_threshold_ms=250.0)
    return result.ok, "\n".join(result.messages)


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return "cpu"
    ok, output = _cached_gpu_gate()
    if ok:
        return "cuda"
    st.warning("CUDA was requested but failed the GPU gate. Falling back to CPU.")
    with st.expander("GPU gate output"):
        st.code(output)
    return "cpu"


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


def _plotly_attention_fallback(activations, layer: int, head: int, tokens: list[str]):
    import plotly.graph_objects as go

    hook = f"blocks.{layer}.attn.hook_pattern"
    pattern = activations._cache[hook][0, head].detach().float().cpu().numpy()
    return go.Figure(data=[go.Heatmap(z=pattern, x=tokens, y=tokens, colorscale="Viridis")])


default_prompts_file = find_default_prompts_file()

with st.sidebar:
    requested_device = st.selectbox("Device", ["cuda", "cpu"], index=0)
    device = _resolve_device(requested_device)
    st.caption(f"Using `{device}`")

    model_name = st.text_input("Model", "pythia-70m")
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

st.title("mil explorer")

try:
    tasks = _load_tasks(prompt_source, prompts_file, limit)
    batch = make_prompt_batch(tasks)
except Exception as exc:
    st.error(f"Prompt loading failed: {type(exc).__name__}: {exc}")
    st.stop()

with st.spinner("Loading model"):
    model = _cached_model(model_name, device)

param_device = next(model._model.parameters()).device
st.caption(f"{model_name} parameter device: `{param_device}`")

hook = st.text_input("Hook", DEFAULT_HOOK)

st.header("Attention")
attn_cols = st.columns([1, 1, 3])
with attn_cols[0]:
    attention_layer = st.number_input("Layer", min_value=0, max_value=12, value=0, step=1)
with attn_cols[1]:
    attention_head = st.number_input("Head", min_value=0, max_value=32, value=0, step=1)
attention_hook = f"blocks.{attention_layer}.attn.hook_pattern"

try:
    attention_acts = get_activations(
        model,
        prompts=batch.target_prompts,
        hooks=[attention_hook],
    )
    token_labels = model._model.to_str_tokens(batch.target_prompts[0])
    try:
        render = attention(
            attention_acts,
            int(attention_layer),
            int(attention_head),
            tokens=token_labels,
        )
        components.html(_render_to_html(render), height=650, scrolling=True)
    except Exception as exc:
        st.warning(f"CircuitsVis render failed; using Plotly fallback. {type(exc).__name__}: {exc}")
        st.plotly_chart(
            _plotly_attention_fallback(
                attention_acts,
                int(attention_layer),
                int(attention_head),
                token_labels,
            ),
            use_container_width=True,
        )
except Exception as exc:
    st.error(f"Attention failed: {type(exc).__name__}: {exc}")

st.header("Patch / Controls")
patch_cols = st.columns([1, 1, 2])
with patch_cols[0]:
    positions = st.selectbox("Positions", ["last", "all"], index=0)
with patch_cols[1]:
    threshold = st.slider("Control threshold", min_value=0.1, max_value=1.5, value=0.8, step=0.05)
run_patch = st.button("Run patch", type="primary")

if run_patch:
    with st.spinner("Running patch and controls"):
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
    if report.flagged:
        st.error(
            f"Control leakiness {status}: ratio {report.ratio:.3f}, "
            f"control max {report.control_max:.4f}, target mean {report.target_delta:.4f}"
        )
    else:
        st.success(
            f"Control leakiness {status}: ratio {report.ratio:.3f}, "
            f"control max {report.control_max:.4f}, target mean {report.target_delta:.4f}"
        )
    st.plotly_chart(patch_bar(result), use_container_width=True)
    with st.expander("PatchResult / ControlReport"):
        st.json(
            {
                "PatchResult": dataclasses.asdict(result),
                "ControlReport": dataclasses.asdict(report),
            }
        )

st.header("SAE Features")
if "deduped" not in model_name:
    st.info(
        "No matching public SAE is wired for non-deduped pythia-70m at this hook. "
        "Use a pythia-70m-deduped model to enable the SAELens release."
    )
elif st.button("Rank SAE features"):
    try:
        from mil.tools import load_sae, rank_features

        with st.spinner("Loading SAE and ranking features"):
            sae = load_sae(DEFAULT_SAE_RELEASE, hook, device=device)
            target_acts = get_activations(model, prompts=batch.target_prompts, hooks=[hook])
            control_acts = get_activations(model, prompts=batch.control_prompts, hooks=[hook])
            ranking = rank_features(sae, target_acts, control_acts, top_k=20)
        st.plotly_chart(feature_table(ranking), use_container_width=True)
        st.plotly_chart(
            activation_heatmap(
                target_acts,
                hook,
                prompt_index=0,
                tokens=model._model.to_str_tokens(batch.target_prompts[0]),
                sae=sae,
                max_features=12,
            ),
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"SAE feature ranking failed: {type(exc).__name__}: {exc}")

st.header("Token Effects")
if "patch_result" in st.session_state:
    st.plotly_chart(token_deltas(st.session_state["patch_result"]), use_container_width=True)
else:
    st.caption("Run patch to populate token effects.")

st.header("Activation Heatmap")
if "hook_activations" in st.session_state:
    st.plotly_chart(
        activation_heatmap(
            st.session_state["hook_activations"],
            hook,
            prompt_index=0,
            tokens=model._model.to_str_tokens(batch.target_prompts[0]),
        ),
        use_container_width=True,
    )
else:
    st.caption("Run patch to populate hook activations.")
