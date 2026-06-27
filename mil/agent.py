"""Agent schema export."""

from __future__ import annotations

import inspect
import importlib
from typing import Any


def build_tool_schemas() -> list[dict]:
    from mil.tools import activations, controls, interventions, models

    compare_mod = importlib.import_module("mil.tools.compare")

    tool_fns = [
        models.load_model,
        activations.get_activations,
        interventions.patch,
        controls.check_controls,
        compare_mod.compare,
    ]
    try:
        from mil.tools import features

        tool_fns += [
            features.load_sae,
            features.rank_features,
            features.top_features_at,
            features.steer_feature,
            features.decoder_projection,
        ]
    except Exception:
        pass

    schemas = []
    for fn in tool_fns:
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            prop: dict[str, Any] = {"description": f"Parameter `{name}`."}
            ann = param.annotation
            if ann is not inspect.Parameter.empty:
                if _annotation_has_handle(ann):
                    if "float" in str(ann) or "Real" in str(ann):
                        prop["anyOf"] = [
                            {
                                "type": "string",
                                "description": "Opaque handle_id from a prior tool result.",
                            },
                            {"type": "number"},
                        ]
                    else:
                        prop["type"] = "string"
                        prop["description"] = f"Opaque handle_id for `{name}` from a prior tool result."
                else:
                    prop["type"] = _annotation_to_json_type(ann)
            if param.default is inspect.Parameter.empty:
                required.append(name)
            properties[name] = prop
        schemas.append(
            {
                "name": fn.__name__,
                "description": inspect.getdoc(fn) or "",
                "input_schema": {"type": "object", "properties": properties, "required": required},
            }
        )
    return schemas


def _annotation_has_handle(ann: Any) -> bool:
    ann_str = str(ann)
    return any(marker in ann_str for marker in ("ModelHandle", "ActivationsHandle", "SAEHandle"))


def _annotation_to_json_type(ann: Any) -> str:
    ann_str = str(ann)
    if "str" in ann_str:
        return "string"
    if "int" in ann_str:
        return "integer"
    if "float" in ann_str:
        return "number"
    if "bool" in ann_str:
        return "boolean"
    if "list" in ann_str or "List" in ann_str:
        return "array"
    return "object"
