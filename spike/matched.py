"""Magnitude-matched random control: replay NXDirection's exact per-(layer, token) edit sizes along random directions.

Recomputing the coefficient per layer is NOT matched: the real edit removes the d-component (adjacent-layer
directions have cos ~0.98, so later layers fire less), while a random edit leaves it, so every layer fires at full
strength (~4-8x the real dose). Record once, replay exactly.
"""

from contextlib import contextmanager

import torch

import hook as hook_mod
from hook import nx_hooks


@contextmanager
def patched_edit(fn):
    real = hook_mod.edit
    hook_mod.edit = fn
    try:
        yield
    finally:
        hook_mod.edit = real


def layer_of(fit_d: torch.Tensor, d: torch.Tensor) -> int:
    return int(torch.argmax(fit_d.to(d.device, d.dtype) @ d))


@torch.no_grad()
def record_coefs(model, ids, d, layers, alpha, mask, bias=None) -> dict[int, torch.Tensor]:
    """Run the real intervention once over the prompt and return alpha*ReLU(h·d - b) per layer, per untrusted row."""
    coefs: dict[int, torch.Tensor] = {}
    real = hook_mod.edit

    def recording(h, dl, a, b):
        coefs[layer_of(d, dl)] = (a * torch.relu(h @ dl - b)).detach().cpu()
        return real(h, dl, a, b)

    with patched_edit(recording), nx_hooks(model, d, layers, alpha, mask, bias=bias):
        model(ids)
    return coefs


def replay_edit(coefs: dict[int, torch.Tensor], d: torch.Tensor, rnd: torch.Tensor):
    """Edit that subtracts the recorded coefficient along rnd[layer] instead of d[layer]."""

    def replay(h, dl, a, b):
        l = layer_of(d, dl)
        c = coefs[l].to(h.device)
        assert c.shape[0] == h.shape[0], f"layer {l}: recorded {c.shape[0]} rows, got {h.shape[0]}"
        return h - c[:, None] * rnd[l].to(h.device, h.dtype)

    return replay


def random_unit(shape, seed: int) -> torch.Tensor:
    r = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    return r / r.norm(dim=-1, keepdim=True)
