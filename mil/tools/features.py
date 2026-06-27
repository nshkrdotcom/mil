"""SAE feature tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mil.tools import logged
from mil.tools.interventions import PatchResult, _mean, _metric_from_logits


@dataclass
class FeatureRanking:
    sae_id: str
    hook: str
    contrast_sizes: tuple[int, int]
    top_features: list[dict]
    summary: dict | None = None

    def _mil_summary(self) -> dict:
        return {
            "type": "FeatureRanking",
            "sae_id": self.sae_id,
            "hook": self.hook,
            "top_feature": self.top_features[0] if self.top_features else None,
        }


@dataclass
class FeatureHits:
    sae_id: str
    hook: str
    prompt_index: int
    position: int
    top_features: list[dict]

    def _mil_summary(self) -> dict:
        return {
            "type": "FeatureHits",
            "sae_id": self.sae_id,
            "hook": self.hook,
            "prompt_index": self.prompt_index,
            "position": self.position,
            "top_feature": self.top_features[0] if self.top_features else None,
        }


@dataclass
class SAEHandle:
    release: str
    sae_id: str
    hook: str
    _sae: Any

    def _mil_summary(self) -> dict:
        return {
            "type": "SAEHandle",
            "release": self.release,
            "sae_id": self.sae_id,
            "hook": self.hook,
        }


@logged
def load_sae(release: str, sae_id: str) -> SAEHandle:
    try:
        from sae_lens import SAE
    except ImportError as e:
        raise ImportError("SAELens is required. Install with: pip install mil[sae]") from e
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id)
    return SAEHandle(release=release, sae_id=sae_id, hook=sae.cfg.hook_name, _sae=sae)


@logged
def rank_features(
    sae: SAEHandle,
    activations_a: "ActivationsHandle",
    activations_b: "ActivationsHandle",
    top_k: int = 20,
) -> FeatureRanking:
    import torch

    acts_a = activations_a._cache[sae.hook][:, -1, :]
    acts_b = activations_b._cache[sae.hook][:, -1, :]
    with torch.no_grad():
        encoded_a = sae._sae.encode(acts_a)
        encoded_b = sae._sae.encode(acts_b)
    mean_a = encoded_a.float().mean(0)
    mean_b = encoded_b.float().mean(0)
    scores = mean_a - mean_b
    indices = scores.topk(top_k).indices.tolist()
    top_features = [
        {
            "feature_id": int(i),
            "score": float(scores[i]),
            "mean_a": float(mean_a[i]),
            "mean_b": float(mean_b[i]),
        }
        for i in indices
    ]
    return FeatureRanking(
        sae_id=sae.sae_id,
        hook=sae.hook,
        contrast_sizes=(len(activations_a.prompts), len(activations_b.prompts)),
        top_features=top_features,
    )


@logged
def top_features_at(
    sae: SAEHandle,
    activations: "ActivationsHandle",
    prompt_index: int,
    position: int = -1,
    top_k: int = 20,
) -> FeatureHits:
    import torch

    act = activations._cache[sae.hook][prompt_index, position, :].unsqueeze(0)
    with torch.no_grad():
        encoded = sae._sae.encode(act).squeeze(0)
    values, indices = encoded.topk(top_k)
    return FeatureHits(
        sae_id=sae.sae_id,
        hook=sae.hook,
        prompt_index=prompt_index,
        position=position,
        top_features=[
            {"feature_id": int(i), "activation": float(v)}
            for i, v in zip(indices.tolist(), values.tolist())
        ],
    )


@logged
def decoder_projection(
    model: "ModelHandle",
    sae: SAEHandle,
    feature_id: int,
    target_token: int | str,
    foil_token: int | str | None = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    target_id = target_token if isinstance(target_token, int) else model._model.to_single_token(target_token)
    direction = model._model.W_U[:, target_id]
    if foil_token is not None:
        foil_id = foil_token if isinstance(foil_token, int) else model._model.to_single_token(foil_token)
        direction = direction - model._model.W_U[:, foil_id]
    decoder = sae._sae.W_dec[feature_id]
    return {
        "feature_id": feature_id,
        "dot": float(torch.dot(decoder, direction)),
        "cosine": float(F.cosine_similarity(decoder, direction, dim=0)),
    }


@logged
def steer_feature(
    model: "ModelHandle",
    sae: SAEHandle,
    feature_id: int,
    scale: float,
    prompts: list[str],
    hook: str | None = None,
    positions: str | list[int] = "last",
    metric: str = "target_logit_delta",
    target_tokens: list[int | str] | None = None,
    foil_tokens: list[int | str] | None = None,
) -> PatchResult:
    import torch

    hook_name = hook or sae.hook
    decoder = sae._sae.W_dec[feature_id]

    def add_decoder(activation, hook_point):
        vec = decoder.to(device=activation.device, dtype=activation.dtype)
        if positions == "all":
            return activation + scale * vec
        out = activation.clone()
        if positions == "last":
            out[:, -1, :] += scale * vec
        else:
            for pos in positions:
                out[:, pos, :] += scale * vec
        return out

    tokens = model._model.to_tokens(prompts)
    with torch.no_grad():
        clean_logits = model._model(tokens)
        steered_logits = model._model.run_with_hooks(tokens, fwd_hooks=[(hook_name, add_decoder)])
    clean_values = _metric_from_logits(model, clean_logits, prompts, metric, target_tokens, foil_tokens)
    steered_values = _metric_from_logits(model, steered_logits, prompts, metric, target_tokens, foil_tokens)
    deltas = [s - c for s, c in zip(steered_values, clean_values)]
    return PatchResult(
        hook=hook_name,
        mode="feature_steering",
        metric=metric,
        target_deltas=deltas,
        control_deltas=[],
        target_mean=_mean(deltas),
        control_mean=None,
    )


from mil.tools.activations import ActivationsHandle  # noqa: E402
