from __future__ import annotations

import pytest

from mil.tools.controls import check_controls


def test_clean_controls_not_flagged():
    report = check_controls(target_delta=1.0, control_deltas=[0.1, -0.2], threshold=0.8)

    assert not report.flagged
    assert report.ratio == pytest.approx(0.2)
    assert report.threshold == pytest.approx(0.8)


def test_leaky_controls_flagged_by_abs_value():
    report = check_controls(target_delta=-1.0, control_deltas=[0.1, -0.9], threshold=0.8)

    assert report.flagged
    assert report.control_max == pytest.approx(-0.9)
    assert report.gap == pytest.approx(0.1)

