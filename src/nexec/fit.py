"""Fit per-layer instruction directions: normalized difference of mean resid_post activations between tokens that
act as the user's instruction and tokens that are the content it operates on (NXDirection)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nexec.corpus import CORPUS_VERSION, Example, corpus_sha256, load_corpus
from nexec.directions import Directions, DirectionsConfig, template_sha256
from nexec.provenance import render

DOC_PREFIX = "Here is a document:\n"
OUTLIER_FACTOR = 10.0  # exclude tokens whose norm exceeds this multiple of the prompt's median norm at that layer


@dataclass
class FitResult:
    d: torch.Tensor           # [L, D] unit rows
    mu_instr: torch.Tensor    # [L, D]
    mu_non: torch.Tensor      # [L, D]
    n_instr: int
    n_non: int
    outlier_frac: list[float]  # per layer: share of labeled tokens excluded as norm outliers


def token_labels(tok, ex: Example, context: str = "user") -> tuple[list[int], list[str | None]]:
    """Render the example as a chat and label each token by the span it lies entirely inside (else None)."""
    messages = []
    for i, t in enumerate(ex.turns):
        content = [DOC_PREFIX, t.text] if (context == "document" and i == 0) else [t.text]
        messages.append({"role": t.role, "content": content})
    r = render(tok, messages)
    starts = []
    for t, ranges in zip(ex.turns, r.part_char_ranges, strict=True):
        a, b = ranges[-1]
        seg = r.text[a:b]
        lead = t.text.index(seg) if seg and seg != t.text else 0  # template may trim leading whitespace
        starts.append(a - lead)
    labels = []
    for s, e in r.offsets:
        while s < e and r.text[s].isspace():  # byte-level BPE folds the preceding space into the token
            s += 1
        label = None
        if s < e:
            for t, base in zip(ex.turns, starts, strict=True):
                hit = next((sp.label for sp in t.spans if base + sp.start <= s and e <= base + sp.end), None)
                if hit:
                    label = hit
                    break
        labels.append(label)
    return r.input_ids, labels


def _layers(model):
    return model.model.layers


@torch.no_grad()
def capture(model, input_ids) -> torch.Tensor:
    """resid_post (decoder-layer outputs) at every layer for one prompt -> [L, T, D] fp32 on CPU."""
    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, device=device).reshape(1, -1)
    outs = []

    def grab(module, args, output):
        outs.append(output[0].float().cpu())

    handles = [layer.register_forward_hook(grab) for layer in _layers(model)]
    try:
        model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(outs)


@torch.no_grad()
def fit_labeled(model, items) -> FitResult:
    """items: iterable of (input_ids, labels) with labels in {instruction, non_instruction, boundary, None}."""
    sums, counts, totals = {}, {}, {}
    n = {"instruction": 0, "non_instruction": 0}
    for ids, labels in items:
        h = capture(model, ids)
        norms = h.norm(dim=-1)
        keep = norms <= OUTLIER_FACTOR * norms.median(dim=1, keepdim=True).values
        for cls in n:
            sel = torch.tensor([lab == cls for lab in labels])
            n[cls] += int(sel.sum())
            w = (keep & sel).float()
            s, c = torch.einsum("lt,ltd->ld", w, h), w.sum(1)
            sums[cls] = s if cls not in sums else sums[cls] + s
            counts[cls] = c if cls not in counts else counts[cls] + c
            totals[cls] = float(sel.sum()) + totals.get(cls, 0.0)
    mu_i = sums["instruction"] / counts["instruction"][:, None]
    mu_n = sums["non_instruction"] / counts["non_instruction"][:, None]
    diff = mu_i - mu_n
    kept = counts["instruction"] + counts["non_instruction"]
    outlier = (1 - kept / (totals["instruction"] + totals["non_instruction"])).tolist()
    return FitResult(diff / diff.norm(dim=-1, keepdim=True), mu_i, mu_n, n["instruction"], n["non_instruction"],
                     [round(x, 4) for x in outlier])


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean())


@torch.no_grad()
def _heldout_diagnostics(model, tok, examples, fit: FitResult, context: str) -> dict:
    proj = {"instruction": [], "non_instruction": []}
    for ex in examples:
        ids, labels = token_labels(tok, ex, context)
        p = torch.einsum("ltd,ld->lt", capture(model, ids), fit.d)
        for cls in proj:
            idx = [i for i, lab in enumerate(labels) if lab == cls]
            proj[cls].append(p[:, idx])
    pi, pn = (torch.cat(proj[c], dim=1) for c in ("instruction", "non_instruction"))
    L = fit.d.shape[0]
    return {
        "heldout_auc": [round(_auc(pi[l], pn[l]), 4) for l in range(L)],
        "heldout_frac_non_pos": [round(float((pn[l] > 0).float().mean()), 4) for l in range(L)],
        "heldout_frac_instr_pos": [round(float((pi[l] > 0).float().mean()), 4) for l in range(L)],
    }


def fit_directions(model, tok, model_id: str, *, examples: list[Example] | None = None, context: str = "user",
                   layers: list[int] | None = None, alpha: float = 1.0, gate: str = "raw") -> Directions:
    """Fit on the train split; record diagnostics on the heldout split."""
    custom = examples is not None
    examples = examples if custom else load_corpus()
    train = [e for e in examples if e.split == "train"]
    heldout = [e for e in examples if e.split == "heldout"]
    fit = fit_labeled(model, (token_labels(tok, e, context) for e in train))
    L = fit.d.shape[0]
    diag = {
        "n_instr": fit.n_instr, "n_non": fit.n_non, "n_train_examples": len(train),
        "n_heldout_examples": len(heldout),
        "mu_instr_proj": [round(float(fit.mu_instr[l] @ fit.d[l]), 4) for l in range(L)],
        "mu_non_proj": [round(float(fit.mu_non[l] @ fit.d[l]), 4) for l in range(L)],
        "outlier_frac": fit.outlier_frac,
    }
    if heldout:
        diag.update(_heldout_diagnostics(model, tok, heldout, fit, context))
    cfg = DirectionsConfig(
        model_id=model_id, model_revision=getattr(model.config, "_commit_hash", None),
        chat_template_sha256=template_sha256(tok), layers=list(range(L)) if layers is None else list(layers),
        alpha=alpha, gate=gate, fit_context=context,
        corpus_version="custom" if custom else CORPUS_VERSION, corpus_sha256="" if custom else corpus_sha256(),
        diagnostics=diag)
    return Directions(fit.d, fit.mu_instr, fit.mu_non, cfg)
