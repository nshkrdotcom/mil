"""Model loading tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mil.tools import logged


@dataclass
class ModelHandle:
    name: str
    device: str
    _model: Any = field(repr=False)

    def _mil_summary(self) -> dict:
        return {"type": "ModelHandle", "name": self.name, "device": self.device}

    def __getattr__(self, item: str):
        return getattr(self._model, item)


@logged
def load_model(name: str, device: str = "cpu") -> ModelHandle:
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise ImportError("TransformerLens is required for load_model().") from e
    model = HookedTransformer.from_pretrained(name, device=device)
    return ModelHandle(name=name, device=device, _model=model)
