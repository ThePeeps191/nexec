"""Span-gated NXDirection intervention: h <- h - alpha * ReLU(h·d - b) * d on untrusted positions."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch


def edit(h: torch.Tensor, d: torch.Tensor, alpha: float, b: float) -> torch.Tensor:
    """h: [N, D] fp32, d: [D] unit. Removes alpha x the part of h·d above b."""
    coef = alpha * torch.relu(h @ d - b)
    return h - coef[:, None] * d


@dataclass
class HookState:
    offset: int = 0   # absolute position of the current forward call's first token
    edits: int = 0    # token-rows processed (summed over layers)


@contextmanager
def nx_hooks(model, d: torch.Tensor, layers: list[int], alpha: float, untrusted_mask: torch.Tensor,
             bias: torch.Tensor | None = None):
    state = HookState()
    mask = untrusted_mask.to(torch.bool).cpu()
    base = model.model

    def set_offset(module, args, kwargs):
        x = kwargs.get("input_ids", args[0] if args else None)
        if x is None:
            x = kwargs["inputs_embeds"]
        assert x.shape[0] == 1, "spike hook supports batch size 1"
        cache = kwargs.get("past_key_values")
        state.offset = cache.get_seq_length() if cache is not None else 0

    def make_layer_hook(l: int):
        def hook(module, args, output):
            t = output.shape[1]
            pos = torch.arange(state.offset, state.offset + t)
            sel = torch.zeros(t, dtype=torch.bool)
            inside = pos < len(mask)
            sel[inside] = mask[pos[inside]]
            if not sel.any():
                return None
            rows = sel.nonzero().squeeze(1).to(output.device)
            dl = d[l].to(output.device, torch.float32)
            b = float(bias[l]) if bias is not None else 0.0
            out = output.clone()
            out[0, rows] = edit(output[0, rows].float(), dl, alpha, b).to(output.dtype)
            state.edits += len(rows)
            return out
        return hook

    handles = [base.register_forward_pre_hook(set_offset, with_kwargs=True)]
    handles += [base.layers[l].register_forward_hook(make_layer_hook(l)) for l in layers]
    try:
        yield state
    finally:
        for h in handles:
            h.remove()
