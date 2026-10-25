"""Follow-up experiments: layer ablation, determinism, utility spot check on natural documents."""

import argparse
import json
from pathlib import Path

import torch

from common import load
from diagnose import DOC_QA_LEAD
from fit import load_fit, out_dir
from hook import nx_hooks
from render import render
from salary import INJECTION, RENDERINGS, answer, is_compromised, is_correct

HERE = Path(__file__).parent
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"


def cfg(fit, lo, hi, alpha=1.0):
    return dict(fit=fit, layers=list(range(lo, hi + 1)), alpha=alpha, gate="raw")


def layer_ablation(model, tok, fit):
    out = []
    for lo, hi in [(0, 4), (0, 8), (0, 11), (5, 11), (9, 27), (12, 27), (0, 26)]:
        for r in RENDERINGS:
            dfd = answer(model, tok, r, INJECTION, cfg(fit, lo, hi))
            cln = answer(model, tok, r, None, cfg(fit, lo, hi))
            row = dict(layers=f"{lo}-{hi}", rendering=r, blocked=not is_compromised(dfd), correct=is_correct(dfd),
                       clean_correct=is_correct(cln), defended=dfd[:80])
            out.append(row)
            print(json.dumps(row), flush=True)
    return out


def determinism(model, tok, fit):
    outs = [answer(model, tok, "md_before", INJECTION, cfg(fit, 0, 27)) for _ in range(3)]
    und = [answer(model, tok, "md_before", INJECTION) for _ in range(3)]
    row = dict(defended_identical=len(set(outs)) == 1, undefended_identical=len(set(und)) == 1, defended=outs[0][:80])
    print(json.dumps(row), flush=True)
    return [row]


@torch.no_grad()
def utility(model, tok, fit):
    out = []
    for line in (HERE / "natural_content.jsonl").read_text().splitlines():
        row = json.loads(line)
        r = render(tok, [(DOC_QA_LEAD, False), (row["passage"], True), ("\n\nQuestion: " + row["question"], False)])
        ids = torch.tensor([r.input_ids], device=model.device)
        kw = dict(attention_mask=torch.ones_like(ids), max_new_tokens=80, do_sample=False)
        off = model.generate(ids, **kw)
        with nx_hooks(model, fit.d, list(range(0, 28)), 1.0, torch.tensor(r.untrusted_mask)):
            on = model.generate(ids, **kw)
        dec = lambda o: tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
        res = dict(kind=row["kind"], question=row["question"], nx_off=dec(off)[:160], nx_on=dec(on)[:160])
        out.append(res)
        print(json.dumps(res), flush=True)
    return out


@torch.no_grad()
def fire(model, tok, fit):
    """Per layer: total ReLU(h·d) over untrusted tokens and mean h·d over injection-note tokens, undefended pass."""
    from fit import capture
    from salary import parts_for, SYSTEM

    out = []
    for rendering in ("md_before", "md_after"):
        r = render(tok, parts_for(rendering, INJECTION), system=SYSTEM)
        h = capture(model, r.input_ids)                         # [L, T, D]; index l = output of decoder layer l
        proj = torch.einsum("ltd,ld->lt", h, fit.d)
        untrusted = [i for i, m in enumerate(r.untrusted_mask) if m]
        inj_lo = r.text.index("IMPORTANT")
        inj = [i for i in untrusted if r.offsets[i][0] >= inj_lo]
        for l in range(proj.shape[0]):
            row = dict(rendering=rendering, layer=l, relu_sum_untrusted=round(float(torch.relu(proj[l, untrusted]).sum()), 2),
                       frac_pos_untrusted=round(float((proj[l, untrusted] > 0).float().mean()), 3),
                       inj_mean=round(float(proj[l, inj].mean()), 3), inj_max=round(float(proj[l, inj].max()), 3))
            out.append(row)
            print(json.dumps(row), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["layers", "determinism", "utility", "fire"])
    args = ap.parse_args()
    model, tok = load(LLAMA)
    fit = load_fit(out_dir(LLAMA, "user") / "fit.safetensors")
    rows = {"layers": layer_ablation, "determinism": determinism, "utility": utility, "fire": fire}[args.what](model, tok, fit)
    (HERE / "out" / f"extra_{args.what}.json").write_text(json.dumps(rows, indent=1))
