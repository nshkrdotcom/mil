"""Activation capture tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mil.tools import logged

if TYPE_CHECKING:
    from mil.tools.models import ModelHandle


@dataclass
class ActivationsHandle:
    prompts: list[str]
    hooks: list[str]
    model_name: str
    summary: dict
    _cache: Any = field(repr=False)

    def _mil_summary(self) -> dict:
        return {
            "type": "ActivationsHandle",
            "model": self.model_name,
            "hooks": self.hooks,
            "n_prompts": len(self.prompts),
            "summary": self.summary,
        }

    def at(self, hook: str) -> Any:
        return self._cache[hook]

    @classmethod
    def from_cache(
        cls,
        *,
        cache: Any,
        prompts: list[str],
        hooks: list[str],
        model_name: str,
    ) -> "ActivationsHandle":
        return cls(
            prompts=prompts,
            hooks=hooks,
            model_name=model_name,
            summary=_summarize_cache(cache, hooks),
            _cache=cache,
        )


def _summarize_tensor(t: Any) -> dict:
    entry = {"shape": list(t.shape), "dtype": str(t.dtype)}
    if hasattr(t, "device"):
        entry["device"] = str(t.device)
    try:
        tf = t.float()
        entry["mean"] = float(tf.mean())
        entry["std"] = float(tf.std())
    except Exception:
        pass
    return entry


def _summarize_cache(cache: Any, hooks: list[str]) -> dict:
    summary = {}
    for hook in hooks:
        summary[hook] = _summarize_tensor(cache[hook]) if hook in cache else {"error": "hook not found"}
    return summary


@logged
def get_activations(
    model: "ModelHandle",
    prompts: list[str] | None,
    hooks: list[str],
    cache: Any | None = None,
) -> ActivationsHandle:
    if prompts is None and cache is None:
        raise ValueError("get_activations requires prompts unless an existing cache is provided.")
    if cache is not None:
        return ActivationsHandle.from_cache(
            cache=cache,
            prompts=prompts or [],
            hooks=hooks,
            model_name=model.name,
        )

    import torch

    tokens = model._model.to_tokens(prompts)
    with torch.no_grad():
        _, cache = model._model.run_with_cache(tokens, names_filter=lambda name: name in hooks)
    return ActivationsHandle(
        prompts=prompts or [],
        hooks=hooks,
        model_name=model.name,
        summary=_summarize_cache(cache, hooks),
        _cache=cache,
    )
