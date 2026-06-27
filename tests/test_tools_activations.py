from __future__ import annotations

import pytest
import torch


class FakeModelHandle:
    name = "fake"

    def __init__(self):
        self._model = self

    def to_tokens(self, prompts):
        if prompts is None:
            raise TypeError("bad prompts")
        return torch.tensor([[1, 2, 3] for _ in prompts])

    def run_with_cache(self, tokens, names_filter):
        cache = {
            "blocks.0.hook_resid_post": torch.ones(tokens.shape[0], tokens.shape[1], 4)
        }
        return None, cache


def test_get_activations_runs_and_summarizes():
    from mil.tools.activations import get_activations

    acts = get_activations(
        model=FakeModelHandle(),
        prompts=["a", "b"],
        hooks=["blocks.0.hook_resid_post"],
    )

    assert acts.summary["blocks.0.hook_resid_post"]["shape"] == [2, 3, 4]
    assert acts.summary["blocks.0.hook_resid_post"]["dtype"] == "torch.float32"


def test_get_activations_wraps_existing_cache_without_running():
    from mil.tools.activations import get_activations

    cache = {"h": torch.zeros(2, 3, 4)}
    acts = get_activations(FakeModelHandle(), prompts=["a", "b"], hooks=["h"], cache=cache)

    assert acts.at("h").shape == (2, 3, 4)


def test_get_activations_requires_prompts_without_cache():
    from mil.tools.activations import get_activations

    with pytest.raises(ValueError, match="requires prompts"):
        get_activations(FakeModelHandle(), prompts=None, hooks=["h"])

