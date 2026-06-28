from __future__ import annotations

# ruff: noqa: E402

import argparse
import dataclasses
import html
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

# Heights used for embedded HTML artifacts.
_HTML_HEIGHT_COMPACT = 300
_HTML_HEIGHT_NORMAL = 380
_HTML_HEIGHT_TALL = 440


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


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
    parser.add_argument("--view", choices=["scroll", "grid"], default="scroll")
    parser.add_argument("--artifacts-dir", default=str(GUIDED_DEMO_DIR))
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    st, components = _require_streamlit()
    args = parse_app_args(argv)
    st.set_page_config(page_title="mil explorer", layout="wide")

    from mil.app_style import apply_app_style

    apply_app_style(st)

    # ── Sidebar ───────────────────────────────────────────────────────────
    mode_options = ["Guided negation walkthrough", "Free exploration"]
    default_mode_index = 0 if args.demo == "guided" else 1

    view_options = ["Scroll narrative", "Dashboard grid"]
    default_view_index = 1 if args.view == "grid" else 0

    with st.sidebar:
        mode = st.selectbox("Demo mode", mode_options, index=default_mode_index)
        if mode == "Guided negation walkthrough":
            view = st.selectbox(
                "Presentation view",
                view_options,
                index=default_view_index,
                help=(
                    "Scroll narrative — compact educational walkthrough.\n"
                    "Dashboard grid — above-the-fold dense overview for demos."
                ),
            )
        else:
            view = "Scroll narrative"

    # ── Route ─────────────────────────────────────────────────────────────
    if mode == "Guided negation walkthrough":
        if view == "Dashboard grid":
            _set_density_mode(st, "grid")
            render_guided_grid(st, components, Path(args.artifacts_dir))
        else:
            _set_density_mode(st, "scroll")
            render_guided_walkthrough(st, components, Path(args.artifacts_dir))
    else:
        _set_density_mode(st, "scroll")
        render_free_exploration(st, components)


# ---------------------------------------------------------------------------
# Scroll-narrative guided walkthrough
# ---------------------------------------------------------------------------


def render_guided_walkthrough(st: Any, components: Any, artifact_dir: Path) -> None:
    from mil.app_style import callout, explain, section_kicker
    from mil.viz import (
        behavior_logit_bars,
        candidate_control_specificity_plot,
        causal_difference_heatmap,
        causal_heatmap,
        logit_margin_curve,
        patch_before_after,
        prompt_family_table,
        residual_trajectory_2d,
    )

    _render_guided_sidebar_nav(st)

    artifacts = _load_artifacts_or_error(st, artifact_dir)
    if artifacts is None:
        return

    cfg = artifacts["manifest"]["config"]

    # ── Header ────────────────────────────────────────────────────────────
    st.title("Guided SELF-GROUND negation walkthrough")
    st.caption(
        f"Precomputed · `{cfg['model_name']}` · `{cfg['hook']}` · "
        f"`{cfg['sae_release']}/{cfg['sae_id']}`"
    )
    _render_guided_summary(st, artifacts)

    # ── Step 0: prompt family ─────────────────────────────────────────────
    section_kicker(st, "Step 0 — Setup")
    st.subheader("What question are we asking?")
    explain(
        st,
        "Does the model have internal components that distinguish a negated statement "
        "from a non-negated control, and do those components causally affect next-token "
        "prediction? Controls narrow the search: a component that fires on both negated "
        "and non-negated prompts is less specific to negation.",
    )
    with st.expander("Prompt family & tokenization", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                _compact_plotly(prompt_family_table(artifacts["prompt_family"]), height=210),
                width="stretch",
            )
        with col_b:
            _render_tokenization_chips(st, artifacts["tokenization"])
        explain(
            st,
            "Tokenization determines hook positions.  Unequal token lengths are normal; "
            "compare positions by label and context, not column number.",
        )

    # ── Step 1: behavior ──────────────────────────────────────────────────
    section_kicker(st, "Step 1 — Behavior")
    st.subheader("What does the model predict?")
    explain(
        st,
        "<b>What it shows:</b> target-vs-foil logit margin — positive means the model "
        "predicts the target token over the foil.  "
        "<b>Why it matters:</b> later interventions try to move this margin.  "
        "<b>What supports the hypothesis:</b> negated prompts show a clear positive margin "
        "while controls show the opposite.",
    )
    col_beh, col_lens = st.columns([1.3, 1])
    with col_beh:
        st.plotly_chart(
            _compact_plotly(behavior_logit_bars(artifacts["behavior_logits"]), height=280),
            width="stretch",
        )
    with col_lens:
        st.dataframe(
            _behavior_table_rows(artifacts["behavior_logits"]),
            width="stretch",
            height=260,
            hide_index=True,
            column_config={
                "variant": st.column_config.TextColumn("variant", width="medium"),
                "target": st.column_config.TextColumn("target", width="medium"),
                "foil": st.column_config.TextColumn("foil", width="medium"),
                "margin": st.column_config.NumberColumn("margin", width="small"),
            },
        )

    # ── Step 1b: logit lens ───────────────────────────────────────────────
    section_kicker(st, "Step 1b — Depth readout")
    st.subheader("Where in the network does the answer emerge?")
    explain(
        st,
        "<b>What it shows:</b> logit-lens projects intermediate residual streams through "
        "the final unembedding, suggesting depths where the readout becomes linearly "
        "available.  <b>Caution:</b> this is not proof of a mechanism — it tells you "
        "where to start looking.",
    )
    st.plotly_chart(
        _compact_plotly(logit_margin_curve(artifacts["logit_lens"]), height=280),
        width="stretch",
    )

    # ── Step 2: causal localization ───────────────────────────────────────
    section_kicker(st, "Step 2 — Causal localization")
    st.subheader("Where is the causal effect localized?")
    explain(
        st,
        "<b>What it shows:</b> ablation of one residual-stream layer/token cell and "
        "the resulting change in behavioral margin.  "
        "<b>What supports the hypothesis:</b> a strong target effect absent from controls.  "
        "<b>What weakens it:</b> a similarly strong effect in the control heatmap.",
    )
    tab_target, tab_control, tab_diff, tab_interp = st.tabs(
        ["Target", "Control", "Target − Control", "How to read it"]
    )
    with tab_target:
        st.plotly_chart(
            _compact_plotly(causal_heatmap(artifacts["causal_heatmap"]["target"], "Target"), height=300),
            width="stretch",
        )
    with tab_control:
        st.plotly_chart(
            _compact_plotly(causal_heatmap(artifacts["causal_heatmap"]["control"], "Control"), height=300),
            width="stretch",
        )
    with tab_diff:
        st.plotly_chart(
            _compact_plotly(
                causal_difference_heatmap(
                    artifacts["causal_heatmap"]["target"],
                    artifacts["causal_heatmap"]["control"],
                ),
                height=300,
            ),
            width="stretch",
        )
    with tab_interp:
        callout(
            st,
            "<b>How to read the causal heatmaps:</b> each cell is one (layer, token-position) "
            "ablation.  Red = ablation reduces the target margin (the site contributes positively "
            "to the negation signal).  Blue = ablation increases the margin.  "
            "The difference heatmap subtracts the control from the target — high positive cells "
            "that disappear in the difference are not negation-specific.",
        )

    # ── Step 3: feature specificity ───────────────────────────────────────
    section_kicker(st, "Step 3 — SAE feature specificity")
    st.subheader("Which SAE features are specific to negation?")
    explain(
        st,
        "<b>What supports the hypothesis:</b> candidate features fire on negated and "
        "paraphrased-negation tokens but not on non-negated controls or decoys.  "
        "<b>What weakens it:</b> density-matched controls showing the same pattern.",
    )

    feat_tabs = st.tabs(["Candidate contrast", "Feature raster", "How to read it"])
    with feat_tabs[0]:
        if "feature_specificity" in artifacts:
            st.plotly_chart(
                _feature_specificity_plotly(
                    candidate_control_specificity_plot(artifacts["feature_specificity"]),
                    height=340,
                ),
                width="stretch",
            )
        elif "feature_unavailable" in artifacts:
            st.info(Path(artifacts["feature_unavailable"]).read_text(encoding="utf-8"))
    with feat_tabs[1]:
        if "feature_raster" in artifacts:
            _embed_html(components, artifacts["feature_raster"], height=_HTML_HEIGHT_NORMAL)
        else:
            st.info("Feature raster not available.")
    with feat_tabs[2]:
        callout(
            st,
            "<b>How to read this:</b> each row is an SAE feature.  Candidate rows should light "
            "up on negated and paraphrased-negation tokens more than on non-negated or decoy "
            "tokens.  Density controls are included so a high activation alone does not look "
            "meaningful — the contrast relative to matched controls is what matters.",
        )

    if "feature_table" in artifacts:
        with st.expander("Feature table (top candidates)", expanded=False):
            _embed_html(components, artifacts["feature_table"], height=_HTML_HEIGHT_COMPACT)

    # ── Step 4: intervention ──────────────────────────────────────────────
    section_kicker(st, "Step 4 — Intervention")
    st.subheader("What changes when we intervene?")
    explain(
        st,
        "<b>What it shows:</b> the logit margin before vs. after ablating the selected "
        "residual-stream site.  "
        "<b>Why it matters:</b> this moves the analysis from correlation toward causality.  "
        "<b>Control leakiness</b> ratio tests whether the ablation is specific.",
    )
    report = artifacts["patch_summary"]["control_report"]
    ratio_str = f"{report['ratio']:.3f}"
    ctrl_max = f"{report['control_max']:.4f}"
    tgt_delta = f"{report['target_delta']:.4f}"
    ctrl_msg = f"ratio {ratio_str} · control max {ctrl_max} · target delta {tgt_delta}"
    if report["flagged"]:
        st.error(f"FLAGGED · {ctrl_msg}")
    else:
        st.info(f"Controls below threshold · {ctrl_msg}")

    int_tabs = st.tabs(["Clean vs ablated", "Token deltas", "Control report"])
    with int_tabs[0]:
        st.plotly_chart(
            _compact_plotly(patch_before_after(artifacts["patch_summary"]), height=280),
            width="stretch",
        )
    with int_tabs[1]:
        _embed_html(components, artifacts["token_deltas"], height=_HTML_HEIGHT_NORMAL)
    with int_tabs[2]:
        st.json(report)

    # ── Step 5: geometry ──────────────────────────────────────────────────
    section_kicker(st, "Step 5 — Geometry (optional)")
    st.subheader("How do representations move through depth?")
    explain(
        st,
        "<b>What it shows:</b> residual-stream states projected into one shared PCA basis.  "
        "<b>Caution:</b> geometry views are diagnostic, not standalone proof.  "
        "<b>What to look for:</b> prompt variants that separate over depth before "
        "committing to more expensive interventions.",
    )
    st.plotly_chart(
        _compact_plotly(residual_trajectory_2d(artifacts["residual_trajectory"]), height=300),
        width="stretch",
    )

    # ── Step 6: attention ─────────────────────────────────────────────────
    section_kicker(st, "Step 6 — Attention drill-down")
    st.subheader("Attention drill-down")
    explain(
        st,
        "Attention patterns are not explanations by themselves.  Use them after causal "
        "localization to inspect routing hypotheses: which source tokens could be supplying "
        "information to a high-effect layer/token cell?",
    )
    with st.expander("Attention pattern viewer", expanded=False):
        _render_attention_artifact(st, components, artifacts, height=_HTML_HEIGHT_TALL)

    # ── Step 7: next steps ────────────────────────────────────────────────
    section_kicker(st, "Step 7 — What to try next")
    callout(
        st,
        "<ul class='mil-next-list'>"
        "<li>Change prompt family — compare lexical vs paraphrased negation.</li>"
        "<li>Inspect the highest-effect cells in the layer/token causal heatmap.</li>"
        "<li>Switch hooks and rerun with matched controls.</li>"
        "<li>Inspect candidate SAE features against density-matched controls.</li>"
        "<li>Export artifacts before changing the model or prompt source.</li>"
        "</ul>",
    )


# ---------------------------------------------------------------------------
# Dashboard grid guided view
# ---------------------------------------------------------------------------


def render_guided_grid(st: Any, components: Any, artifact_dir: Path) -> None:
    """Above-the-fold dense overview for live demos / presentations."""
    from mil.app_style import apply_dense_grid_style

    apply_dense_grid_style(st)
    _render_guided_sidebar_nav(st)

    artifacts = _load_artifacts_or_error(st, artifact_dir)
    if artifacts is None:
        return

    cfg = artifacts["manifest"]["config"]

    from mil.viz import (
        behavior_logit_bars,
        candidate_control_specificity_plot,
        causal_difference_heatmap,
        causal_heatmap,
        logit_margin_curve,
        patch_before_after,
        residual_trajectory_2d,
    )

    # ── Compact header row ────────────────────────────────────────────────
    title_col, meta_col = st.columns([1.1, 2.9], gap="small")
    with title_col:
        st.markdown("## Guided negation walkthrough")
    with meta_col:
        st.markdown(
            f'<div class="mil-grid-note mil-grid-note-top">'
            f'Precomputed · <code>{cfg["model_name"]}</code> · '
            f'<code>{cfg["hook"]}</code> · '
            f'<code>{cfg["sae_release"]}/{cfg["sae_id"]}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Mini metrics ──────────────────────────────────────────────────────
    _render_guided_summary(st, artifacts, compact=True)
    _hairline(st)

    # ── Row 1: Behavior · Logit-lens · Target−Control causal heatmap ─────
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.plotly_chart(
            _grid_fig(behavior_logit_bars(artifacts["behavior_logits"]), height=245),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            _grid_fig(logit_margin_curve(artifacts["logit_lens"]), height=245),
            width="stretch",
        )
    with col3:
        st.plotly_chart(
            _grid_heatmap_fig(
                causal_difference_heatmap(
                    artifacts["causal_heatmap"]["target"],
                    artifacts["causal_heatmap"]["control"],
                ),
                height=245,
            ),
            width="stretch",
        )

    # ── Row 2: SAE specificity · Intervention · Residual trajectory ───────
    col4, col5, col6 = st.columns(3, gap="small")
    with col4:
        if "feature_specificity" in artifacts:
            st.plotly_chart(
                _grid_fig(
                    candidate_control_specificity_plot(artifacts["feature_specificity"]),
                    height=245,
                ),
                width="stretch",
            )
        elif "feature_unavailable" in artifacts:
            st.info(Path(artifacts["feature_unavailable"]).read_text(encoding="utf-8"))
    with col5:
        st.plotly_chart(
            _grid_fig(patch_before_after(artifacts["patch_summary"]), height=245),
            width="stretch",
        )
    with col6:
        st.plotly_chart(
            _grid_fig(residual_trajectory_2d(artifacts["residual_trajectory"]), height=245),
            width="stretch",
        )

    # ── Feature extraction strip ──────────────────────────────────────────
    _render_feature_extraction_strip(st, artifacts)
    _grid_drilldown_gap(st)

    # ── Drill-down expander (collapsed by default) ────────────────────────
    with st.expander("Drill-down: attention · full heatmaps · token deltas", expanded=False):
        tab_attn, tab_heatmaps, tab_tokens = st.tabs(
            ["Attention", "Causal heatmaps", "Token deltas"]
        )
        with tab_attn:
            _render_attention_artifact(st, components, artifacts, height=320)
        with tab_heatmaps:
            hm_t, hm_c = st.columns(2, gap="small")
            with hm_t:
                st.plotly_chart(
                    _grid_heatmap_fig(
                        causal_heatmap(artifacts["causal_heatmap"]["target"], "Target"),
                        height=230,
                    ),
                    width="stretch",
                )
            with hm_c:
                st.plotly_chart(
                    _grid_heatmap_fig(
                        causal_heatmap(artifacts["causal_heatmap"]["control"], "Control"),
                        height=230,
                    ),
                    width="stretch",
                )
        with tab_tokens:
            _embed_html(components, artifacts["token_deltas"], height=320)


# ---------------------------------------------------------------------------
# Shared helpers (guided)
# ---------------------------------------------------------------------------


def _render_guided_sidebar_nav(st: Any) -> None:
    with st.sidebar:
        st.markdown("### Researcher workflow")
        for label, anchor in [
            ("1. Behavior", "what-does-the-model-predict"),
            ("2. Localization", "where-is-the-causal-effect-localized"),
            ("3. Feature specificity", "which-sae-features-are-specific-to-negation"),
            ("4. Intervention", "what-changes-when-we-intervene"),
            ("5. Geometry", "how-do-representations-move-through-depth"),
            ("6. Attention", "attention-drill-down"),
            ("7. Next steps", "what-to-try-next"),
        ]:
            st.markdown(f"- [{label}](#{anchor})")


def _load_artifacts_or_error(st: Any, artifact_dir: Path) -> dict | None:
    """Load precomputed guided artifacts; render an error card on failure."""
    try:
        return load_guided_artifacts(artifact_dir)
    except Exception as exc:
        st.error(f"Guided demo artifacts were not found: {type(exc).__name__}: {exc}")
        st.markdown(
            "Build them with:\n\n"
            "```bash\n"
            ".venv/bin/python scripts/build_guided_demo_artifacts.py\n"
            "```"
        )
        return None


def _set_density_mode(st: Any, mode: str) -> None:
    if mode not in {"scroll", "grid"}:
        raise ValueError(f"Unknown density mode: {mode}")
    st.markdown(
        f'<div id="mil-density-mode" data-mode="{mode}"></div>',
        unsafe_allow_html=True,
    )


def _render_guided_summary(st: Any, artifacts: dict, *, compact: bool = False) -> None:
    """Render the four key-metric summary row.

    Parameters
    ----------
    compact:
        When True, renders custom mini metric cards (zero extra padding) instead
        of native ``st.metric`` widgets.  Use in grid/dashboard mode.
    """
    from mil.app_style import mini_metrics

    behavior_rows = artifacts["behavior_logits"]["variants"]
    negated = behavior_rows[0]
    patch_rows = artifacts["patch_summary"]["rows"]
    report = artifacts["patch_summary"]["control_report"]
    feature_rows = artifacts.get("feature_specificity", {}).get("rows", [])
    candidate_count = len([r for r in feature_rows if r.get("kind") == "candidate"])

    if compact:
        mini_metrics(
            st,
            [
                {"label": "Negated margin", "value": f"{negated['margin']:.3f}"},
                {"label": "Ablation delta", "value": f"{patch_rows[0]['delta']:.3f}"},
                {
                    "label": "Control ratio",
                    "value": f"{report['ratio']:.3f}",
                    "delta": "FLAGGED" if report["flagged"] else "below threshold",
                    "delta_bad": report["flagged"],
                },
                {"label": "SAE candidates", "value": str(candidate_count)},
            ],
        )
    else:
        cols = st.columns(4, gap="small")
        cols[0].metric("Negated margin", f"{negated['margin']:.3f}")
        cols[1].metric("Ablation delta", f"{patch_rows[0]['delta']:.3f}")
        cols[2].metric(
            "Control ratio",
            f"{report['ratio']:.3f}",
            delta="FLAGGED" if report["flagged"] else "below threshold",
            delta_color="inverse" if report["flagged"] else "normal",
        )
        cols[3].metric("SAE candidates", str(candidate_count))


def _render_feature_extraction_strip(st: Any, artifacts: dict) -> None:
    """Compact educational strip showing SAE feature separation summary."""
    if "feature_specificity" not in artifacts:
        return

    summary = _feature_strip_summary(artifacts["feature_specificity"])
    st.markdown(
        f"""
        <div class="mil-feature-strip">
          <div>
            <div class="mil-strip-label">Top candidate feature</div>
            <div class="mil-strip-main">feature {summary["best_feature_id"]}</div>
          </div>
          <div>
            <div class="mil-strip-label">Candidate contrast</div>
            <div class="mil-strip-value">{summary["best_contrast"]}</div>
          </div>
          <div>
            <div class="mil-strip-label">Max density-control contrast</div>
            <div class="mil-strip-value">{summary["control_max"]}</div>
          </div>
          <div class="mil-strip-note">
            Candidate SAE features should separate negated and paraphrased-negation
            prompts from non-negated controls and decoys.  A large gap between
            candidate contrast and control contrast suggests specificity.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hairline(st: Any) -> None:
    """Render a thin 1px divider line."""
    st.markdown('<div class="mil-hairline"></div>', unsafe_allow_html=True)


def _grid_drilldown_gap(st: Any) -> None:
    """Reserve vertical space between the feature strip and drill-down expander."""
    st.markdown('<div class="mil-grid-drilldown-gap"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data helpers (testable without Streamlit)
# ---------------------------------------------------------------------------


def _feature_strip_summary(feature_specificity: dict) -> dict[str, str]:
    """Summarize feature specificity data for the extraction strip.

    Returns a dict with string values ready for display:
    ``best_feature_id``, ``best_contrast``, ``control_max``.
    """
    rows = feature_specificity.get("rows", [])
    candidates = [r for r in rows if r.get("kind") == "candidate"]
    controls = [r for r in rows if r.get("kind") != "candidate"]
    best = max(candidates, key=lambda r: r["contrast"]) if candidates else None
    control_max = max((abs(r["contrast"]) for r in controls), default=0.0)
    return {
        "best_feature_id": str(best["feature_id"]) if best else "—",
        "best_contrast": f"{best['contrast']:.2f}" if best else "—",
        "control_max": f"{control_max:.2f}",
    }


def _behavior_table_rows(behavior: dict) -> list[dict]:
    """Return compact display rows for the behavior logit table."""
    return [
        {
            "variant": row["label"],
            "target": row["target_token"],
            "foil": row["foil_token"],
            "margin": round(row["margin"], 3),
        }
        for row in behavior["variants"]
    ]


def _prompt_family_rows(prompt_family: dict) -> list[dict]:
    """Return compact display rows for the prompt family table."""
    return [
        {
            "variant": row["label"],
            "role": row.get("role", ""),
            "target": row.get("target_token", ""),
            "foil": row.get("foil_token", ""),
            "prompt": row["prompt"],
        }
        for row in prompt_family["variants"]
    ]


def _render_tokenization_chips(st: Any, tokenization: dict) -> None:
    """Render tokenization as wrapped chips instead of a clipped table."""
    rows = []
    for variant in tokenization["variants"]:
        chips = []
        for index, token in enumerate(variant["tokens"]):
            chips.append(
                "<span class='mil-token-chip'> "
                f"<span class='mil-token-index'>{index}:</span>"
                f"<span class='mil-token-text'>{html.escape(str(token))}</span>"
                " </span>"
            )
        rows.append(
            "<div class='mil-token-row'>"
            f"<div class='mil-token-label'>{html.escape(str(variant['label']))}</div>"
            f"<div class='mil-token-wrap'>{' '.join(chips)}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='mil-token-panel'>"
        "<div class='mil-mini-title'>Tokenization by prompt variant</div>"
        f"{''.join(rows)}"
        "</div>",
        unsafe_allow_html=True,
    )


def _compact_plotly(fig: Any, *, height: int = 280, title: str | None = None) -> Any:
    """Apply compact chart sizing for scroll-narrative mode.

    Does *not* change colors or data — only height/margins/font.
    """
    updates: dict[str, Any] = dict(
        height=height,
        margin={"l": 36, "r": 18, "t": 36, "b": 36},
        font={"family": "Inter, ui-sans-serif, sans-serif", "size": 10, "color": "#e5e7eb"},
        title_font={"size": 12, "color": "#e5e7eb"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 9},
        },
    )
    if title is not None:
        updates["title"] = title
    fig.update_layout(**updates)
    fig.update_xaxes(
        tickfont=dict(color="#94a3b8"),
        title_font=dict(color="#94a3b8"),
    )
    fig.update_yaxes(
        tickfont=dict(color="#94a3b8"),
        title_font=dict(color="#94a3b8"),
    )
    return fig


def _feature_specificity_plotly(fig: Any, *, height: int = 340) -> Any:
    fig = _compact_plotly(fig, height=height)
    fig.update_layout(margin=dict(l=92, r=18, t=42, b=96))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9, color="#94a3b8"), automargin=True)
    fig.update_yaxes(tickfont=dict(size=9, color="#94a3b8"), automargin=True)
    return fig


def _grid_fig(fig: Any, *, height: int = 245) -> Any:
    """Ultra-tight Plotly layout for dashboard grid cells."""
    fig.update_layout(
        height=height,
        margin=dict(l=42, r=14, t=48, b=58),
        font=dict(family="Inter, ui-sans-serif, sans-serif", size=9, color="#e5e7eb"),
        title_font=dict(size=11, color="#e5e7eb"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=8),
        ),
    )
    fig.update_xaxes(
        tickfont=dict(size=8, color="#94a3b8"),
        title_font=dict(size=8, color="#94a3b8"),
    )
    fig.update_yaxes(
        tickfont=dict(size=8, color="#94a3b8"),
        title_font=dict(size=8, color="#94a3b8"),
    )
    return fig


def _grid_heatmap_fig(fig: Any, *, height: int = 245) -> Any:
    """Ultra-tight Plotly layout for heatmaps in dashboard grid."""
    fig.update_layout(
        height=height,
        margin=dict(l=44, r=14, t=48, b=64),
        font=dict(family="Inter, ui-sans-serif, sans-serif", size=9, color="#e5e7eb"),
        title_font=dict(size=11, color="#e5e7eb"),
    )
    fig.update_xaxes(
        tickfont=dict(size=7, color="#94a3b8"),
        title_font=dict(size=8, color="#94a3b8"),
    )
    fig.update_yaxes(
        tickfont=dict(size=8, color="#94a3b8"),
        title_font=dict(size=8, color="#94a3b8"),
    )
    return fig


# ---------------------------------------------------------------------------
# Free exploration mode
# ---------------------------------------------------------------------------


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
            st.json(
                {"PatchResult": dataclasses.asdict(result), "ControlReport": dataclasses.asdict(report)}
            )

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


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


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


def _embed_html(
    components: Any,
    path: str | Path,
    *,
    height: int,
    scrolling: bool = False,
) -> None:
    html = Path(path).read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=scrolling)


def _render_attention_artifact(
    st: Any,
    components: Any,
    artifacts: dict,
    *,
    height: int,
) -> None:
    if "attention_heatmap" in artifacts:
        _embed_html(components, artifacts["attention_heatmap"], height=height)
        return
    if "attention_screenshot" in artifacts:
        st.image(str(artifacts["attention_screenshot"]))
        return
    _embed_html(components, artifacts["attention"], height=height)


def _require_streamlit():
    try:
        import streamlit as st
        import streamlit.components.v1 as components
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Streamlit is required. Install with: "
            "uv pip install --python .venv/bin/python -e '.[app]'"
        ) from exc
    return st, components


if __name__ == "__main__":
    main(sys.argv[1:])
