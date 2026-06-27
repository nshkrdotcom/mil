from __future__ import annotations

from types import SimpleNamespace

import torch


class FakeSAE:
    W_dec = torch.eye(4)

    def encode(self, acts):
        return acts


class FakeTLModel:
    W_U = torch.eye(4)

    def to_single_token(self, tok):
        mapping = {" A": 0, " B": 1}
        return mapping[tok] if tok in mapping else int(tok)

    def to_tokens(self, prompts):
        return torch.zeros(len(prompts), 3, dtype=torch.long)

    def __call__(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 4)

    def run_with_hooks(self, tokens, fwd_hooks):
        activation = torch.zeros(tokens.shape[0], tokens.shape[1], 4)
        for _, hook_fn in fwd_hooks:
            activation = hook_fn(activation, None)
        logits = self(tokens).clone()
        logits[:, -1, :] = activation[:, -1, :]
        return logits


def _acts(values):
    from mil.tools.activations import ActivationsHandle

    return ActivationsHandle(
        prompts=["p"] * values.shape[0],
        hooks=["h"],
        model_name="fake",
        summary={},
        _cache={"h": values},
    )


def test_rank_features_uses_two_handles():
    from mil.tools.features import SAEHandle, rank_features

    sae = SAEHandle("rel", "id", "h", FakeSAE())
    a = _acts(torch.tensor([[[2.0, 1.0, 0.0, 0.0]]]))
    b = _acts(torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]))

    ranking = rank_features(sae, a, b, top_k=1)
    assert ranking.contrast_sizes == (1, 1)
    assert ranking.top_features[0]["feature_id"] == 0


def test_top_features_at_returns_feature_hits():
    from mil.tools.features import FeatureHits, SAEHandle, top_features_at

    sae = SAEHandle("rel", "id", "h", FakeSAE())
    hits = top_features_at(sae, _acts(torch.tensor([[[0.1, 3.0, 0.2, 0.0]]])), 0, 0, 1)

    assert isinstance(hits, FeatureHits)
    assert hits.top_features[0]["feature_id"] == 1


def test_steer_feature_applies_positioned_decoder_vector():
    from mil.tools.features import SAEHandle, steer_feature

    sae = SAEHandle("rel", "id", "h", FakeSAE())
    model = SimpleNamespace(name="fake", _model=FakeTLModel())

    result = steer_feature(
        model,
        sae,
        feature_id=0,
        scale=2.0,
        prompts=["p"],
        hook="h",
        positions="last",
        target_tokens=[" A"],
    )

    assert result.mode == "feature_steering"
    assert result.target_deltas == [2.0]
