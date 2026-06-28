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
# Design tokens and CSS
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

/* ── Structural selectors only ── */

[data-testid="stAppViewContainer"] {
  background: var(--mil-bg);
}

[data-testid="stSidebar"] {
  background: #111827;
  border-right: 1px solid var(--mil-border);
}

.main .block-container {
  padding-top: 1.4rem;
  padding-bottom: 1.2rem;
  max-width: 1500px;
}

/* ── Typography ── */

h1 {
  font-size: 1.65rem !important;
  line-height: 1.15 !important;
  margin-bottom: 0.25rem !important;
}

h2 {
  font-size: 1.05rem !important;
  line-height: 1.2 !important;
  margin-top: 1.05rem !important;
  margin-bottom: 0.35rem !important;
}

h3 {
  font-size: 0.88rem !important;
  margin-top: 0.75rem !important;
}

p, li, div[data-testid="stMarkdownContainer"] {
  font-size: 0.78rem;
  line-height: 1.35;
}

/* ── Compact metrics ── */

[data-testid="stMetric"] {
  background: var(--mil-panel);
  border: 1px solid var(--mil-border);
  border-radius: 12px;
  padding: 0.55rem 0.7rem;
}

[data-testid="stMetricLabel"] {
  font-size: 0.68rem;
  color: var(--mil-muted);
}

[data-testid="stMetricValue"] {
  font-size: 1.05rem;
}

/* ── Semantic panel classes ── */

.mil-card {
  background: var(--mil-panel);
  border: 1px solid var(--mil-border);
  border-radius: 14px;
  padding: 0.75rem 0.85rem;
  margin-bottom: 0.7rem;
}

.mil-card-tight {
  background: var(--mil-panel);
  border: 1px solid var(--mil-border);
  border-radius: 12px;
  padding: 0.55rem 0.65rem;
  margin-bottom: 0.55rem;
}

.mil-section-kicker {
  color: var(--mil-accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  font-size: 0.62rem;
  margin-bottom: 0.18rem;
}

.mil-explain {
  color: var(--mil-muted);
  font-size: 0.76rem;
  line-height: 1.35;
  margin: 0.15rem 0 0.55rem 0;
}

.mil-callout {
  border-left: 3px solid var(--mil-accent);
  background: rgba(56, 189, 248, 0.08);
  padding: 0.45rem 0.65rem;
  border-radius: 8px;
  color: var(--mil-text);
  margin-bottom: 0.6rem;
}

.mil-grid-note {
  color: var(--mil-muted);
  font-size: 0.68rem;
  line-height: 1.25;
}

/* ── Embedded HTML frames ── */

iframe {
  border-radius: 12px;
}

/* ── Tighten default vertical / horizontal gaps ── */

div[data-testid="stVerticalBlock"] {
  gap: 0.45rem;
}

div[data-testid="stHorizontalBlock"] {
  gap: 0.65rem;
}

/* ── Tables / dataframes ── */

[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
}

/* ── Dashboard ("grid") mode shell ── */

.mil-dashboard-shell {
  height: calc(100vh - 5.2rem);
  overflow: hidden;
}

/* ── Typography helpers used in HTML panels ── */

.mil-mini-title {
  font-size: 0.78rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
  color: var(--mil-text);
}

.mil-mini-copy {
  color: var(--mil-muted);
  font-size: 0.66rem;
  line-height: 1.25;
}
</style>
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_app_style(st: Any) -> None:
    """Inject the shared CSS into the Streamlit app.  Call once, near the top."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def card(
    st: Any,
    title: str,
    body: str | None = None,
    *,
    kicker: str | None = None,
    tight: bool = False,
) -> None:
    """Render a styled information card panel.

    Parameters
    ----------
    st:
        The Streamlit module (passed in to keep this function testable without
        importing streamlit at module level).
    title:
        Card heading text.
    body:
        Optional muted explanatory paragraph rendered below the title.
    kicker:
        Optional uppercase accent label rendered above the title.
    tight:
        When True use the compact ``.mil-card-tight`` variant.
    """
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
