from __future__ import annotations


def test_app_css_uses_stable_selectors():
    from mil.app_style import APP_CSS

    assert '[data-testid="stAppViewContainer"]' in APP_CSS
    assert '[data-testid="stSidebar"]' in APP_CSS
    assert ".css-" not in APP_CSS
    assert "mil-card" in APP_CSS


def test_dense_grid_css_is_available_and_stable():
    from mil.app_style import APP_CSS, DENSE_GRID_CSS

    # Dense CSS must widen the container significantly
    assert "max-width: 1920px" in DENSE_GRID_CSS or "max-width: none" in DENSE_GRID_CSS
    # Dense CSS must override base padding
    assert "padding-top: 0.38rem" in DENSE_GRID_CSS
    # Mini metric and feature strip classes must be in base APP_CSS
    assert "mil-mini-metric" in APP_CSS
    assert "mil-feature-strip" in APP_CSS
    assert "mil-hairline" in APP_CSS
    # No generated selectors in either
    assert ".css-" not in APP_CSS
    assert ".css-" not in DENSE_GRID_CSS


def test_apply_app_style_calls_markdown(monkeypatch):
    from mil.app_style import APP_CSS, apply_app_style

    calls = []

    class FakeSt:
        def markdown(self, text, unsafe_allow_html=False):
            calls.append((text, unsafe_allow_html))

    apply_app_style(FakeSt())
    assert len(calls) == 1
    assert calls[0][0] == APP_CSS
    assert calls[0][1] is True


def test_apply_dense_grid_style_calls_markdown(monkeypatch):
    from mil.app_style import DENSE_GRID_CSS, apply_dense_grid_style

    calls = []

    class FakeSt:
        def markdown(self, text, unsafe_allow_html=False):
            calls.append((text, unsafe_allow_html))

    apply_dense_grid_style(FakeSt())
    assert len(calls) == 1
    assert calls[0][0] == DENSE_GRID_CSS
    assert calls[0][1] is True
