"""Baselines for asking "is it the direction?": the magnitude-matched random control.

Record NXDirection's exact per-(layer, token) coefficients over the prompt, then replay them along random unit
directions. Recomputing the coefficient per layer is NOT matched: the real edit removes the d-component (adjacent
layers' directions are nearly parallel, so later layers fire less), a random edit does not, and every later layer
would fire at full strength (4-8x the real dose in step-0 measurements).
"""

from __future__ import annotations

import torch


def random_unit(shape, seed: int) -> torch.Tensor:
    r = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    return r / r.norm(dim=-1, keepdim=True)


@torch.no_grad()
def matched_random(nx, conversations, *, seed: int, layers=None, alpha=None, gate=None, **generate_kwargs) -> list[str]:
    enc = nx.encode(conversations)
    dev = nx.model.device
    ids, att = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    coefs: dict[int, torch.Tensor] = {}
    with nx.protect(enc.untrusted_mask, layers=layers, alpha=alpha, gate=gate, record=coefs):
        nx.model(ids, attention_mask=att, position_ids=(att.cumsum(-1) - 1).clamp(min=0), use_cache=False)
    R = random_unit(nx.directions.d.shape, seed)
    with nx.protect(enc.untrusted_mask, layers=layers, alpha=alpha, gate=gate, replay=(coefs, R)):
        return nx._generate(enc, **generate_kwargs)
