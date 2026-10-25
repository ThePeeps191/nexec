"""Fit per-layer instruction directions (difference of means on resid_post)."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

OUTLIER_FACTOR = 10.0  # exclude tokens whose ‖h‖ exceeds this multiple of the prompt's median norm


@dataclass
class Fit:
    d: torch.Tensor          # [L, D] unit rows
    mu_instr: torch.Tensor   # [L, D]
    mu_non: torch.Tensor     # [L, D]
    n_instr: int
    n_non: int


def layers_of(model):
    return model.model.layers


@torch.no_grad()
def capture(model, input_ids) -> torch.Tensor:
    """resid_post at every decoder layer for one prompt -> [L, T, D] fp32 on CPU."""
    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, device=device).reshape(1, -1)
    outs = []

    def grab(module, args, output):
        assert isinstance(output, torch.Tensor), f"decoder layer returned {type(output)}"
        outs.append(output[0].float().cpu())

    handles = [layer.register_forward_hook(grab) for layer in layers_of(model)]
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(outs)


@torch.no_grad()
def fit_labeled(model, items) -> Fit:
    """items: iterable of (input_ids, labels) with labels in {instruction, non_instruction, boundary, None}."""
    sums = {"instruction": None, "non_instruction": None}
    counts = {"instruction": None, "non_instruction": None}
    n = {"instruction": 0, "non_instruction": 0}
    for ids, labels in items:
        h = capture(model, ids)                                    # [L, T, D]
        norms = h.norm(dim=-1)                                     # [L, T]
        keep = norms <= OUTLIER_FACTOR * norms.median(dim=1, keepdim=True).values
        for cls in sums:
            sel = torch.tensor([lab == cls for lab in labels])
            n[cls] += int(sel.sum())
            w = (keep & sel).float()                               # [L, T]
            s, c = torch.einsum("lt,ltd->ld", w, h), w.sum(1)
            sums[cls] = s if sums[cls] is None else sums[cls] + s
            counts[cls] = c if counts[cls] is None else counts[cls] + c
    mu_i = sums["instruction"] / counts["instruction"][:, None]
    mu_n = sums["non_instruction"] / counts["non_instruction"][:, None]
    diff = mu_i - mu_n
    return Fit(diff / diff.norm(dim=-1, keepdim=True), mu_i, mu_n, n["instruction"], n["non_instruction"])


DOC_PREFIX = "Here is a document:\n"


def fit(model, tok, examples, context: str = "user") -> Fit:
    from render import token_labels

    prefix = DOC_PREFIX if context == "document" else ""
    return fit_labeled(model, (token_labels(tok, ex, prefix=prefix) for ex in examples))


def save_fit(f: Fit, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({"d": f.d.contiguous(), "mu_instr": f.mu_instr.contiguous(), "mu_non": f.mu_non.contiguous()},
              str(path), metadata={"n_instr": str(f.n_instr), "n_non": str(f.n_non)})


def load_fit(path) -> Fit:
    from safetensors import safe_open

    t = load_file(str(path))
    with safe_open(str(path), "pt") as fh:
        meta = fh.metadata()
    return Fit(t["d"], t["mu_instr"], t["mu_non"], int(meta["n_instr"]), int(meta["n_non"]))


def out_dir(model_id: str, context: str) -> Path:
    from common import short

    return Path(__file__).parent / "out" / short(model_id) / context


def main():
    from common import load
    from corpus import load_corpus

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", choices=["user", "document"], default="user")
    args = ap.parse_args()
    model, tok = load(args.model)
    train = [ex for ex in load_corpus(Path(__file__).with_name("corpus_pilot.jsonl")) if ex.split == "train"]
    f = fit(model, tok, train, args.context)
    path = out_dir(args.model, args.context) / "fit.safetensors"
    save_fit(f, path)
    print(f"{args.model} [{args.context}] n_instr={f.n_instr} n_non={f.n_non} -> {path}")


if __name__ == "__main__":
    main()
