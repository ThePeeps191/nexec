"""Per-layer diagnostics for fitted instruction directions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import MODELS, load, short
from corpus import load_corpus
from fit import DOC_PREFIX, capture, load_fit, out_dir
from render import render, token_labels

HERE = Path(__file__).parent
DOC_QA_LEAD = "Answer the question using the document.\n\nDocument:\n"


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(pos > neg) + 0.5 P(tie), via pairwise comparison (sets are small)."""
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean())


def labeled_projections(model, tok, examples, d, prefix):
    """-> {label: [L, N] projections} over corpus tokens."""
    acc = {"instruction": [], "non_instruction": []}
    for ex in examples:
        ids, labels = token_labels(tok, ex, prefix=prefix)
        proj = torch.einsum("ltd,ld->lt", capture(model, ids), d)
        for cls in acc:
            idx = [i for i, lab in enumerate(labels) if lab == cls]
            acc[cls].append(proj[:, idx])
    return {k: torch.cat(v, dim=1) for k, v in acc.items()}


def natural_projections(model, tok, d):
    """Projections of untrusted document tokens in a doc-QA prompt -> ([L, N], token strings)."""
    projs, toks = [], []
    for line in (HERE / "natural_content.jsonl").read_text().splitlines():
        row = json.loads(line)
        r = render(tok, [(DOC_QA_LEAD, False), (row["passage"], True), ("\n\nQuestion: " + row["question"], False)])
        idx = [i for i, m in enumerate(r.untrusted_mask) if m]
        proj = torch.einsum("ltd,ld->lt", capture(model, r.input_ids), d)
        projs.append(proj[:, idx])
        toks += [tok.decode([r.input_ids[i]]) for i in idx]
    return torch.cat(projs, dim=1), toks


def diagnose(model_id: str, context: str):
    model, tok = load(model_id)
    f = load_fit(out_dir(model_id, context) / "fit.safetensors")
    d = f.d
    corpus = load_corpus(HERE / "corpus_pilot.jsonl")
    prefix = DOC_PREFIX if context == "document" else ""
    train = labeled_projections(model, tok, [e for e in corpus if e.split == "train"], d, prefix)
    held = labeled_projections(model, tok, [e for e in corpus if e.split == "heldout"], d, prefix)
    nat, nat_toks = natural_projections(model, tok, d)

    rows = []
    L = d.shape[0]
    for l in range(L):
        mu_i, mu_n = float(f.mu_instr[l] @ d[l]), float(f.mu_non[l] @ d[l])
        mid = (mu_i + mu_n) / 2
        k = int(nat[l].abs().argmax())
        rows.append({
            "layer": l,
            "mu_instr_proj": round(mu_i, 3),
            "mu_non_proj": round(mu_n, 3),
            "sep_over_std": round((mu_i - mu_n) / float(torch.cat([train["instruction"][l], train["non_instruction"][l]]).std()), 3),
            "frac_non_pos": round(float((train["non_instruction"][l] > 0).float().mean()), 3),
            "frac_instr_pos": round(float((train["instruction"][l] > 0).float().mean()), 3),
            "heldout_auc": round(auc(held["instruction"][l], held["non_instruction"][l]), 3),
            "natural_frac_pos": round(float((nat[l] > 0).float().mean()), 3),
            "natural_frac_above_mid": round(float((nat[l] > mid).float().mean()), 3),
            "natural_median_proj": round(float(nat[l].median()), 3),
            "cos_prev": round(float(d[l] @ d[l - 1]), 3) if l else "",
            "max_abs_proj_token": repr(nat_toks[k]),
        })
    outd = out_dir(model_id, context)
    with open(outd / "diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = 7
    fig, axes = plt.subplots((L + cols - 1) // cols, cols, figsize=(3 * cols, 2.2 * ((L + cols - 1) // cols)))
    for l, ax in enumerate(axes.flat):
        if l >= L:
            ax.axis("off")
            continue
        for data, lab, c in [(train["non_instruction"][l], "content", "#2a6fb0"),
                             (train["instruction"][l], "instruction", "#b03a2e"),
                             (nat[l], "natural doc", "#7a7a7a")]:
            ax.hist(data.numpy(), bins=40, alpha=0.5, density=True, label=lab, color=c)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"L{l}  auc={rows[l]['heldout_auc']}", fontsize=8)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(f"{short(model_id)} [{context}]: h·d per layer")
    fig.tight_layout()
    fig.savefig(outd / "diagnostics.png", dpi=110)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--context", choices=["user", "document"], default="user")
    args = ap.parse_args()
    rows = diagnose(args.model, args.context)
    keys = ["layer", "mu_instr_proj", "mu_non_proj", "sep_over_std", "frac_non_pos", "heldout_auc",
            "natural_frac_pos", "natural_frac_above_mid", "natural_median_proj", "cos_prev"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))


if __name__ == "__main__":
    main()
