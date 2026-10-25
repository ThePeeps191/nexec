"""Span-gated NXDirection intervention on the residual stream of HF decoder models.

For every untrusted position i at each chosen layer l (on the decoder layer's output, i.e. resid_post):

    h_i <- h_i - alpha * ReLU(h_i . d_l - b_l) * d_l

Positions are absolute indices into the (left-padded) prompt the mask was built for. The hook tracks them itself,
so chunked prefill and sliding-window caches (whose ``get_seq_length`` can be capped) are handled; decode steps fall
past the end of the mask and are never edited.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


def edit(h: torch.Tensor, d: torch.Tensor, alpha: float, b: float = 0.0) -> torch.Tensor:
    """h: [N, D], d: [D] unit. Remove alpha x the part of h.d above b (fp32 math)."""
    h = h.float()
    d = d.float()
    coef = alpha * torch.relu(h @ d - b)
    return h - coef[:, None] * d


@dataclass
class HookState:
    edits: int = 0       # untrusted token-rows processed, summed over layers
    offset: int = 0      # absolute position of the current forward call's first token
    next: int = 0        # absolute position the next call continues from
    started: bool = False


def _layers(model):
    return model.model.layers


@contextmanager
def protect(model, *, d: torch.Tensor, layers: list[int], alpha: float, untrusted_mask: torch.Tensor,
            bias: torch.Tensor | None = None, record: dict[int, torch.Tensor] | None = None,
            replay: tuple[dict[int, torch.Tensor], torch.Tensor] | None = None):
    """Install the intervention for the duration of the block. Yields a :class:`HookState`.

    ``untrusted_mask`` is ``[T]`` or ``[B, T]``, aligned to the prompt's (left-padded) ``input_ids``.
    ``record`` (a dict) receives, per layer, the ``[B, T]`` coefficients alpha*ReLU(h.d - b) actually applied.
    ``replay=(coefs, R)`` applies recorded coefficients along directions ``R`` instead (magnitude-matched control).
    """
    n_layers = len(_layers(model))
    bad = [l for l in layers if not 0 <= l < n_layers]
    if bad:
        raise ValueError(f"layer indices {bad} out of range for a {n_layers}-layer model")
    mask = untrusted_mask.to(torch.bool).cpu()
    if mask.dim() == 1:
        mask = mask[None]
    state = HookState()

    def before_model(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        n = cache.get_seq_length() if cache is not None else 0
        if n == 0:
            state.offset = 0
        elif state.started:
            state.offset = state.next
        else:
            state.offset = n  # a cache prefilled before entering the context
        state.started = True

    def after_model(module, args, kwargs, output):
        x = kwargs.get("input_ids", args[0] if args else None)
        if x is None:
            x = kwargs["inputs_embeds"]
        state.next = state.offset + x.shape[1]

    def make_layer_hook(l: int):
        def hook(module, args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"decoder layer {l} returned {type(output).__name__}; expected a tensor")
            B, T, _ = output.shape
            if mask.shape[0] != B:
                raise ValueError(f"untrusted_mask has batch {mask.shape[0]} but the model got batch {B}")
            pos = torch.arange(state.offset, state.offset + T)
            inside = pos < mask.shape[1]
            sel = torch.zeros(B, T, dtype=torch.bool)
            sel[:, inside] = mask[:, pos[inside]]
            if not sel.any():
                return None
            b_idx, t_idx = sel.nonzero(as_tuple=True)
            p_idx = pos[t_idx]
            dev = output.device
            rows = output[b_idx.to(dev), t_idx.to(dev)].float()
            if replay is not None:
                coefs, R = replay
                coef = coefs[l][b_idx, p_idx].to(dev, torch.float32)
                new = rows - coef[:, None] * R[l].to(dev, torch.float32)
            else:
                dl = d[l].to(dev, torch.float32)
                bl = float(bias[l]) if bias is not None else 0.0
                coef = alpha * torch.relu(rows @ dl - bl)
                new = rows - coef[:, None] * dl
                if record is not None:
                    rec = record.setdefault(l, torch.zeros(mask.shape, dtype=torch.float32))
                    rec[b_idx, p_idx] = coef.cpu()
            out = output.clone()
            out[b_idx.to(dev), t_idx.to(dev)] = new.to(output.dtype)
            state.edits += len(b_idx)
            return out

        return hook

    base = model.model
    handles = [base.register_forward_pre_hook(before_model, with_kwargs=True),
               base.register_forward_hook(after_model, with_kwargs=True)]
    handles += [_layers(model)[l].register_forward_hook(make_layer_hook(l)) for l in layers]
    try:
        yield state
    finally:
        for h in handles:
            h.remove()
