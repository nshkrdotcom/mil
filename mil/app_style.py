"""Centralized styling and layout helpers for the mil Streamlit app.

Rules:
- All CSS lives here.  No inline style= or st.markdown("<style>…") calls elsewhere.
- Only stable Streamlit structural selectors are used (data-testid attributes and
  element-type selectors).  Generated class names like .css-abc123 are forbidden.
- Helper functions that emit HTML panels accept `st` as a parameter so they are
  independently testable without importing streamlit at module load time.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Base design tokens + compact document CSS
# ---------------------------------------------------------------------------

APP_CSS = """
<style>
:root {
  --mil-bg: #0b0f17;
  --mil-panel: #111827;
  --mil-panel-2: #151d2b;
  --mil-border: rgba(148, 163, 184, 0.20);
  --mil-muted: #94a3b8;
  --mil-text: #e5e7eb;
  --mil-accent: #38bdf8;
  --mil-good: #34d399;
  --mil-warn: #f59e0b;
  --mil-bad: #fb7185;
}

/* ── Structural selectors only — no generated class names ── */

[data-testid="stAppViewContainer"] {
  background: var(--mil-bg);
}

[data-testid="stHeader"] {
  height: 44px !important;
  min-height: 44px !important;
  background: rgba(11, 15, 23, 0.96) !important;
  border-bottom: 1px solid var(--mil-border);
}

[data-testid="stHeader"]::before {
  content: "mil explorer";
  position: absolute;
  left: 0.85rem;
  top: 0.72rem;
  color: var(--mil-muted);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

[data-testid="stToolbar"] {
  height: 44px !important;
  min-height: 44px !important;
}

[data-testid="stSidebarHeader"] {
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}

[data-testid="stSidebarUserContent"] {
  padding-top: 0.55rem !important;
}

[data-testid="stSidebar"] {
  background: #111827;
  border-right: 1px solid var(--mil-border);
  min-width: 220px !important;
  max-width: 265px !important;
}

[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
  font-size: 0.70rem !important;
}

[data-testid="stSidebar"] h3 {
  font-size: 0.80rem !important;
  margin-top: 0.55rem !important;
}

.main .block-container,
[data-testid="stMainBlockContainer"] {
  padding-top: 3.15rem !important;
  padding-bottom: 0.45rem !important;
  padding-left: 0.85rem !important;
  padding-right: 0.85rem !important;
  max-width: 1800px !important;
}

/* ── Typography ── */

h1 {
  color: var(--mil-text) !important;
  font-size: 1.55rem !important;
  line-height: 1.12 !important;
  margin-bottom: 0.2rem !important;
}

h2 {
  color: var(--mil-text) !important;
  font-size: 1.0rem !important;
  line-height: 1.2 !important;
  margin-top: 0.85rem !important;
  margin-bottom: 0.3rem !important;
}

h3 {
  color: var(--mil-text) !important;
  font-size: 0.85rem !important;
  margin-top: 0.65rem !important;
  margin-bottom: 0.2rem !important;
}

p, li, div[data-testid="stMarkdownContainer"] {
  color: var(--mil-text);
  font-size: 0.78rem;
  line-height: 1.35;
}

div[data-testid="stMarkdownContainer"] > p {
  margin-bottom: 0.3rem !important;
}

/* ── Compact metrics ── */

[data-testid="stMetric"] {
  background: var(--mil-panel);
  border: 1px solid var(--mil-border);
  border-radius: 12px;
  padding: 0.52rem 0.68rem;
}

[data-testid="stMetricLabel"] {
  font-size: 0.67rem;
  color: var(--mil-muted);
}

[data-testid="stMetricValue"] {
  font-size: 1.02rem;
}

/* ── Semantic panel classes ── */

.mil-card {
  background: var(--mil-panel);
  border: 1px solid var(--mil-border);
  border-radius: 14px;
  padding: 0.72rem 0.82rem;
  margin-bottom: 0.65rem;
}

.mil-card-tight {
  background: var(--mil-panel);
  border: 1px solid var(--mil-border);
  border-radius: 12px;
  padding: 0.50rem 0.60rem;
  margin-bottom: 0.50rem;
}

.mil-section-kicker {
  color: var(--mil-accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  font-size: 0.60rem;
  margin-bottom: 0.15rem;
}

.mil-explain {
  color: var(--mil-muted);
  font-size: 0.74rem;
  line-height: 1.33;
  margin: 0.12rem 0 0.76rem 0;
}

.mil-callout {
  border-left: 3px solid var(--mil-accent);
  background: rgba(56, 189, 248, 0.08);
  padding: 0.42rem 0.62rem;
  border-radius: 8px;
  color: var(--mil-text);
  margin-bottom: 0.55rem;
}

.mil-next-list {
  margin: 0;
  padding-left: 1.1rem;
}

.mil-grid-note {
  color: var(--mil-muted);
  font-size: 0.66rem;
  line-height: 1.22;
}

.mil-grid-note-top {
  margin-top: 0.35rem;
}

/* ── Embedded HTML frames ── */

iframe {
  border-radius: 10px;
}

/* ── Default vertical / horizontal gaps ── */

div[data-testid="stVerticalBlock"] {
  gap: 0.38rem;
}

div[data-testid="stHorizontalBlock"] {
  gap: 0.55rem;
}

hr {
  margin: 0.45rem 0 !important;
  border-color: rgba(148, 163, 184, 0.22) !important;
}

/* ── Tables / dataframes ── */

[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
}

/* ── Typography helpers ── */

.mil-mini-title {
  font-size: 0.76rem;
  font-weight: 700;
  margin-bottom: 0.22rem;
  color: var(--mil-text);
}

.mil-mini-copy {
  color: var(--mil-muted);
  font-size: 0.64rem;
  line-height: 1.22;
}

/* ── Custom mini metric cards (used in compact=True summary row) ── */

.mil-mini-metric {
  background: #111827;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 9px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 64px;
  padding: 0.34rem 0.50rem;
  margin-bottom: 0.50rem;
  overflow: hidden;
}

.mil-mini-metric-label {
  color: #94a3b8;
  font-size: 0.58rem;
  line-height: 1;
  margin-bottom: 0.16rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.mil-mini-metric-value {
  color: #e5e7eb;
  font-size: 0.88rem;
  font-weight: 700;
  line-height: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.mil-mini-metric-delta {
  display: inline-block;
  margin-top: 0.16rem;
  color: #34d399;
  background: rgba(52, 211, 153, 0.12);
  border-radius: 999px;
  padding: 0.04rem 0.26rem;
  font-size: 0.52rem;
  line-height: 1;
  max-width: max-content;
  white-space: nowrap;
}

.mil-mini-metric-delta.flagged {
  color: #fb7185;
  background: rgba(251, 113, 133, 0.12);
}

/* ── Feature extraction strip ── */

.mil-feature-strip {
  display: grid;
  grid-template-columns: 1.1fr 0.8fr 0.9fr 2.4fr;
  gap: 0.44rem;
  align-items: center;
  background: linear-gradient(90deg, rgba(52,211,153,0.09), rgba(56,189,248,0.05));
  border: 1px solid rgba(148,163,184,0.20);
  border-radius: 10px;
  padding: 0.36rem 0.52rem;
  margin: 0.38rem 0 0.90rem;
  position: relative;
  z-index: 1;
}

.mil-strip-label {
  color: #94a3b8;
  font-size: 0.52rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.10rem;
}

.mil-strip-main,
.mil-strip-value {
  color: #e5e7eb;
  font-size: 0.76rem;
  font-weight: 700;
}

.mil-strip-note {
  color: #94a3b8;
  font-size: 0.60rem;
  line-height: 1.18;
}

/* ── Hairline divider ── */

.mil-hairline {
  height: 1px;
  background: rgba(148, 163, 184, 0.18);
  margin: 0.28rem 0;
}

.mil-grid-drilldown-gap {
  height: 0.65rem;
}

.mil-token-panel {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid var(--mil-border);
  border-radius: 10px;
  padding: 0.56rem 0.62rem;
}

.mil-token-row {
  margin-bottom: 0.48rem;
}

.mil-token-row:last-child {
  margin-bottom: 0;
}

.mil-token-label {
  color: var(--mil-muted);
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  margin-bottom: 0.20rem;
  text-transform: uppercase;
}

.mil-token-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 0.22rem;
}

.mil-token-chip {
  background: #1e293b;
  border: 1px solid rgba(148, 163, 184, 0.20);
  border-radius: 6px;
  color: #e5e7eb;
  display: inline-flex;
  max-width: 100%;
  min-height: 1.34rem;
  padding: 0.12rem 0.32rem;
  white-space: normal;
  word-break: break-word;
}

.mil-token-index {
  color: var(--mil-accent);
  font-size: 0.54rem;
  margin-right: 0.18rem;
}

.mil-token-text {
  font-size: 0.68rem;
  line-height: 1.15;
}
</style>
"""

# ---------------------------------------------------------------------------
# Dense overlay CSS — injected ONLY in dashboard grid mode
# ---------------------------------------------------------------------------

DENSE_GRID_CSS = """
<style>
[data-testid="stHeader"] {
  height: 0 !important;
  min-height: 0 !important;
  background: transparent !important;
}

[data-testid="stHeader"]::before {
  content: none !important;
  display: none !important;
}

[data-testid="stToolbar"] {
  display: none !important;
}

[data-testid="stSidebarHeader"] {
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}

[data-testid="stSidebarUserContent"] {
  padding-top: 0.42rem !important;
}

.main .block-container,
[data-testid="stMainBlockContainer"] {
  padding-top: 0.38rem !important;
  padding-bottom: 0.20rem !important;
  padding-left: 0.55rem !important;
  padding-right: 0.55rem !important;
  max-width: none !important;
}

div[data-testid="stVerticalBlock"] {
  gap: 0.34rem !important;
}

div[data-testid="stHorizontalBlock"] {
  gap: 0.42rem !important;
}

div[data-testid="column"],
div[data-testid="stColumn"] {
  padding-left: 0.10rem !important;
  padding-right: 0.10rem !important;
}

[data-testid="stMetric"] {
  padding: 0.34rem 0.48rem !important;
  min-height: 50px !important;
}

[data-testid="stMetricLabel"] {
  font-size: 0.60rem !important;
}

[data-testid="stMetricValue"] {
  font-size: 0.88rem !important;
}

h1 {
  font-size: 1.22rem !important;
  margin-bottom: 0.10rem !important;
}

h2, h3 {
  margin-top: 0.28rem !important;
  margin-bottom: 0.14rem !important;
  font-size: 0.82rem !important;
}

.mil-section-kicker {
  font-size: 0.52rem !important;
  margin-bottom: 0.06rem !important;
  margin-top: 0.18rem !important;
}

.mil-explain {
  font-size: 0.64rem !important;
  line-height: 1.20 !important;
  margin: 0.04rem 0 0.20rem 0 !important;
}

div[data-testid="stMarkdownContainer"] > p {
  margin-bottom: 0.18rem !important;
}

hr {
  margin: 0.28rem 0 !important;
}

.mil-hairline {
  margin: 0.18rem 0;
}

.mil-mini-metric {
  height: 64px !important;
  margin-bottom: 0.56rem !important;
}

.mil-feature-strip {
  margin-top: 0.36rem !important;
  margin-bottom: 0.92rem !important;
}

.mil-grid-drilldown-gap {
  height: 0.70rem !important;
}

[data-testid="stExpander"] {
  margin-top: 0.22rem !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_app_style(st: Any) -> None:
    """Inject the shared base CSS. Call once after set_page_config."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def apply_dense_grid_style(st: Any) -> None:
    """Inject the ultra-dense overlay CSS for dashboard grid mode.

    Must be called *after* apply_app_style so the overrides win.
    """
    st.markdown(DENSE_GRID_CSS, unsafe_allow_html=True)


def mini_metrics(st: Any, metrics: list[dict[str, str]]) -> None:
    """Render a row of compact custom metric cards.

    Parameters
    ----------
    st:
        The Streamlit module.
    metrics:
        List of dicts with keys: ``label``, ``value``, and optional ``delta``
        and ``delta_bad`` (bool; turns the delta pill red when True).
    """
    cols = st.columns(len(metrics), gap="small")
    for col, metric in zip(cols, metrics, strict=True):
        with col:
            delta = metric.get("delta", "")
            is_bad = metric.get("delta_bad", False)
            delta_class = "mil-mini-metric-delta flagged" if is_bad else "mil-mini-metric-delta"
            delta_html = (
                f'<span class="{delta_class}">{delta}</span>' if delta else ""
            )
            col.markdown(
                f"""
                <div class="mil-mini-metric">
                  <div class="mil-mini-metric-label">{metric["label"]}</div>
                  <div class="mil-mini-metric-value">{metric["value"]}</div>
                  {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def card(
    st: Any,
    title: str,
    body: str | None = None,
    *,
    kicker: str | None = None,
    tight: bool = False,
) -> None:
    """Render a styled information card panel."""
    css_class = "mil-card-tight" if tight else "mil-card"
    kicker_html = (
        f'<div class="mil-section-kicker">{kicker}</div>' if kicker else ""
    )
    body_html = f'<div class="mil-explain">{body}</div>' if body else ""
    st.markdown(
        f"""
        <div class="{css_class}">
          {kicker_html}
          <div class="mil-mini-title">{title}</div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(st: Any, text: str) -> None:
    """Render a left-bordered accent callout block."""
    st.markdown(
        f'<div class="mil-callout">{text}</div>',
        unsafe_allow_html=True,
    )


def section_kicker(st: Any, label: str) -> None:
    """Render a small uppercase section label (accent color)."""
    st.markdown(
        f'<div class="mil-section-kicker">{label}</div>',
        unsafe_allow_html=True,
    )


def explain(st: Any, text: str) -> None:
    """Render a muted explanatory paragraph."""
    st.markdown(
        f'<div class="mil-explain">{text}</div>',
        unsafe_allow_html=True,
    )
