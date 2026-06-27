from __future__ import annotations

import torch


class FakeTLModel:
    W_U = torch.eye(5)

    def to_single_token(self, tok):
        mapping = {" A": 0, " B": 1, " C": 2}
        return mapping[tok] if tok in mapping else int(tok)

    def to_tokens(self, prompts):
        return torch.zeros(len(prompts), 3, dtype=torch.long)

    def __call__(self, tokens):
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], 5)
        logits[:, -1, 0] = 1.0
        logits[:, -1, 1] = 0.5
        return logits

    def run_with_hooks(self, tokens, fwd_hooks):
        activation = torch.ones(tokens.shape[0], tokens.shape[1], 5)
        for _, hook_fn in fwd_hooks:
            activation = hook_fn(activation, None)
        logits = self(tokens).clone()
        logits[:, -1, :] += activation[:, -1, :]
        return logits


class FakeModelHandle:
    name = "fake"

    def __init__(self):
        self._model = FakeTLModel()


def test_patch_target_logit_delta_with_control_foils():
    from mil.tools.interventions import patch

    result = patch(
        FakeModelHandle(),
        hook="h",
        source=0.0,
        target_prompts=["p"],
        control_prompts=["c"],
        metric="logit_diff_delta",
        target_tokens=[" A"],
        control_tokens=[" B"],
        foil_tokens=[" B"],
        control_foil_tokens=[" C"],
    )

    assert result.metric == "logit_diff_delta"
    assert len(result.target_deltas) == 1
    assert len(result.control_deltas) == 1
