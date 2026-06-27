"""Tool logging and exports."""

from __future__ import annotations

import functools
import sys
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def logged(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        try:
            import mil

            args_summary = {}
            names = fn.__code__.co_varnames
            for i, arg in enumerate(args):
                key = names[i] if i < fn.__code__.co_argcount else f"arg{i}"
                args_summary[key] = _summarize(arg)
            for key, value in kwargs.items():
                args_summary[key] = _summarize(value)
            mil._get_backend().log(
                {
                    "kind": "tool_call",
                    "tool": fn.__name__,
                    "args": args_summary,
                    "result": _summarize(result),
                    "duration_ms": round(elapsed_ms, 1),
                    "when": time.time(),
                }
            )
        except Exception as e:
            print(
                f"mil logging failed in {fn.__name__}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        return result

    return wrapper  # type: ignore[return-value]


def _summarize(obj: Any) -> Any:
    if hasattr(obj, "_mil_summary"):
        return obj._mil_summary()
    if hasattr(obj, "shape"):
        try:
            return {"type": type(obj).__qualname__, "shape": list(obj.shape)}
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) <= 5:
        return [_summarize(x) for x in obj]
    try:
        text = repr(obj)
    except Exception as e:
        text = f"<repr error: {type(e).__name__}>"
    return {"type": type(obj).__qualname__, "repr": text[:100]}


from mil.tools.activations import ActivationsHandle, get_activations
from mil.tools.compare import compare
from mil.tools.controls import ControlReport, check_controls
from mil.tools.interventions import PatchResult, patch
from mil.tools.models import ModelHandle, load_model

try:
    from mil.tools.features import (
        FeatureHits,
        FeatureRanking,
        SAEHandle,
        decoder_projection,
        load_sae,
        rank_features,
        steer_feature,
        top_features_at,
    )
except Exception:
    pass
