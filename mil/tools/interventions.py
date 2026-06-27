"""Intervention tools."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

from mil.tools import logged


@dataclass
class PatchResult:
    hook: str
    mode: str
    metric: str
    target_deltas: list[float]
    control_deltas: list[float]
    target_mean: float
    control_mean: float | None

    def _mil_summary(self) -> dict:
        return {
            "type": "PatchResult",
            "hook": self.hook,
            "mode": self.mode,
            "metric": self.metric,
            "target_mean": self.target_mean,
            "control_mean": self.control_mean,
            "num_targets": len(self.target_deltas),
            "num_controls": len(self.control_deltas),
        }


def _expand_token_ids(model: Any, spec: list[int | str] | None, n: int, metric: str, label: str) -> list[int]:
    if metric == "logit_gap_delta":
        return []
    if spec is None:
        raise ValueError(f"{label} is required for metric={metric!r}")
    if len(spec) == 1:
        spec = spec * n
    if len(spec) != n:
        raise ValueError(f"{label} must have length 1 or match prompt count")
    return [tok if isinstance(tok, int) else model._model.to_single_token(tok) for tok in spec]


def _metric_from_logits(
    model: Any,
    logits: Any,
    prompts: list[str],
    metric: str,
    tokens_spec: list[int | str] | None,
    foil_spec: list[int | str] | None,
) -> list[float]:
    import torch

    last = logits[:, -1, :]
    if metric == "logit_gap_delta":
        top2 = torch.topk(last, k=2, dim=-1).values
        return (top2[:, 0] - top2[:, 1]).tolist()
    token_ids = _expand_token_ids(model, tokens_spec, len(prompts), metric, "target_tokens/control_tokens")
    rows = torch.arange(len(prompts), device=last.device)
    values = last[rows, torch.tensor(token_ids, device=last.device)]
    if metric == "target_logit_delta":
        return values.tolist()
    if metric == "logit_diff_delta":
        foil_ids = _expand_token_ids(model, foil_spec, len(prompts), metric, "foil_tokens")
        foil_values = last[rows, torch.tensor(foil_ids, device=last.device)]
        return (values - foil_values).tolist()
    raise ValueError(f"Unknown patch metric: {metric!r}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@logged
def patch(
    model: "ModelHandle",
    hook: str,
    source: "ActivationsHandle | float",
    target_prompts: list[str],
    control_prompts: list[str] | None = None,
    positions: str | list[int] = "all",
    metric: str = "target_logit_delta",
    target_tokens: list[int | str] | None = None,
    control_tokens: list[int | str] | None = None,
    foil_tokens: list[int | str] | None = None,
    control_foil_tokens: list[int | str] | None = None,
) -> PatchResult:
    import torch

    hook_name = hook

    def run_metric(prompts: list[str], token_spec, foil_spec) -> list[float]:
        tokens = model._model.to_tokens(prompts)
        with torch.no_grad():
            logits = model._model(tokens)
        return _metric_from_logits(model, logits, prompts, metric, token_spec, foil_spec)

    def hook_fn(activation, hook_point):
        if isinstance(source, Real):
            return torch.full_like(activation, source)
        src_act = source._cache[hook_name]
        out = activation.clone()
        if positions == "last":
            out[:, -1, :] = src_act[:, -1, :]
        elif positions == "all":
            min_len = min(out.shape[1], src_act.shape[1])
            out[:, :min_len, :] = src_act[:, :min_len, :]
        else:
            for pos in positions:
                out[:, pos, :] = src_act[:, pos, :]
        return out

    def run_patched(prompts: list[str], token_spec, foil_spec) -> list[float]:
        tokens = model._model.to_tokens(prompts)
        with torch.no_grad():
            logits = model._model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
        return _metric_from_logits(model, logits, prompts, metric, token_spec, foil_spec)

    baseline_target = run_metric(target_prompts, target_tokens, foil_tokens)
    patched_target = run_patched(target_prompts, target_tokens, foil_tokens)
    target_deltas = [p - b for p, b in zip(patched_target, baseline_target)]

    control_deltas: list[float] = []
    if control_prompts:
        ctrl_tokens = control_tokens if control_tokens is not None else target_tokens
        ctrl_foils = control_foil_tokens if control_foil_tokens is not None else foil_tokens
        baseline_ctrl = run_metric(control_prompts, ctrl_tokens, ctrl_foils)
        patched_ctrl = run_patched(control_prompts, ctrl_tokens, ctrl_foils)
        control_deltas = [p - b for p, b in zip(patched_ctrl, baseline_ctrl)]

    return PatchResult(
        hook=hook_name,
        mode="ablation" if isinstance(source, Real) else "activation_patch",
        metric=metric,
        target_deltas=target_deltas,
        control_deltas=control_deltas,
        target_mean=_mean(target_deltas),
        control_mean=_mean(control_deltas) if control_deltas else None,
    )
