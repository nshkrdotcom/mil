"""Intervention tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

from mil.tools import logged

if TYPE_CHECKING:
    from mil.tools.activations import ActivationsHandle
    from mil.tools.models import ModelHandle


@dataclass
class PatchResult:
    hook: str
    mode: str
    metric: str
    target_deltas: list[float]
    control_deltas: list[float]
    target_mean: float
    control_mean: float | None
    token_deltas: list[dict] = field(default_factory=list)

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
            "num_token_deltas": len(self.token_deltas),
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


def _top_token_deltas(model: Any, clean_logits: Any, patched_logits: Any, top_k: int = 10) -> list[dict]:
    import torch

    deltas = (patched_logits[:, -1, :] - clean_logits[:, -1, :]).float().mean(0)
    k = min(top_k, deltas.numel())
    pos_values, pos_indices = torch.topk(deltas, k=k)
    neg_values, neg_indices = torch.topk(-deltas, k=k)
    rows = []
    for value, index in zip(pos_values.tolist(), pos_indices.tolist()):
        rows.append(
            {
                "token_id": int(index),
                "token": _token_to_string(model, int(index)),
                "delta": float(value),
                "direction": "positive",
            }
        )
    for value, index in zip(neg_values.tolist(), neg_indices.tolist()):
        rows.append(
            {
                "token_id": int(index),
                "token": _token_to_string(model, int(index)),
                "delta": float(-value),
                "direction": "negative",
            }
        )
    return rows


def _token_to_string(model: Any, token_id: int) -> str:
    tokenizer_model = model._model
    for method in ("to_single_str_token", "to_string"):
        fn = getattr(tokenizer_model, method, None)
        if fn is None:
            continue
        try:
            return str(fn(token_id))
        except Exception:
            pass
    return str(token_id)


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

    def run_logits(prompts: list[str]):
        tokens = model._model.to_tokens(prompts)
        with torch.no_grad():
            return model._model(tokens)

    def hook_fn(activation, hook=None):
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

    def run_patched_logits(prompts: list[str]):
        tokens = model._model.to_tokens(prompts)
        with torch.no_grad():
            return model._model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

    baseline_target_logits = run_logits(target_prompts)
    patched_target_logits = run_patched_logits(target_prompts)
    baseline_target = _metric_from_logits(
        model, baseline_target_logits, target_prompts, metric, target_tokens, foil_tokens
    )
    patched_target = _metric_from_logits(
        model, patched_target_logits, target_prompts, metric, target_tokens, foil_tokens
    )
    target_deltas = [p - b for p, b in zip(patched_target, baseline_target)]

    control_deltas: list[float] = []
    if control_prompts:
        ctrl_tokens = control_tokens if control_tokens is not None else target_tokens
        ctrl_foils = control_foil_tokens if control_foil_tokens is not None else foil_tokens
        baseline_ctrl_logits = run_logits(control_prompts)
        patched_ctrl_logits = run_patched_logits(control_prompts)
        baseline_ctrl = _metric_from_logits(
            model, baseline_ctrl_logits, control_prompts, metric, ctrl_tokens, ctrl_foils
        )
        patched_ctrl = _metric_from_logits(
            model, patched_ctrl_logits, control_prompts, metric, ctrl_tokens, ctrl_foils
        )
        control_deltas = [p - b for p, b in zip(patched_ctrl, baseline_ctrl)]

    return PatchResult(
        hook=hook_name,
        mode="ablation" if isinstance(source, Real) else "activation_patch",
        metric=metric,
        target_deltas=target_deltas,
        control_deltas=control_deltas,
        target_mean=_mean(target_deltas),
        control_mean=_mean(control_deltas) if control_deltas else None,
        token_deltas=_top_token_deltas(model, baseline_target_logits, patched_target_logits),
    )
