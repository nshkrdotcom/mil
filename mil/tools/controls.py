"""Control-specificity checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlReport:
    target_delta: float
    control_max: float
    gap: float
    ratio: float
    threshold: float
    flagged: bool

    def _mil_summary(self) -> dict:
        return {
            "type": "ControlReport",
            "target_delta": self.target_delta,
            "control_max": self.control_max,
            "gap": self.gap,
            "ratio": self.ratio,
            "threshold": self.threshold,
            "flagged": self.flagged,
        }


def check_controls(
    target_delta: float,
    control_deltas: list[float],
    threshold: float = 0.8,
) -> ControlReport:
    if not control_deltas:
        return ControlReport(
            target_delta=target_delta,
            control_max=0.0,
            gap=target_delta,
            ratio=0.0,
            threshold=threshold,
            flagged=False,
        )
    control_max = max(control_deltas, key=abs)
    gap = abs(target_delta) - abs(control_max)
    ratio = abs(control_max) / abs(target_delta) if target_delta != 0 else float("inf")
    flagged = abs(control_max) >= threshold * abs(target_delta)
    return ControlReport(
        target_delta=target_delta,
        control_max=control_max,
        gap=gap,
        ratio=ratio,
        threshold=threshold,
        flagged=flagged,
    )
