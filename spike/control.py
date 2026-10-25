"""Controls for the salary flip: is the effect specific to the instruction direction?

- random: same layers/alpha/gate, but d replaced by random unit vectors (seeds 0..4).
- paraphrase: the fixed paraphrase set under the chosen config.
"""

import argparse
import json
from pathlib import Path

import torch

from common import load, short
from fit import Fit, load_fit, out_dir
from matched import patched_edit, random_unit, record_coefs, replay_edit
from hook import nx_hooks
from render import render
from salary import INJECTION, SYSTEM, answer, gate_bias, is_compromised, is_correct, parts_for, run

HERE = Path(__file__).parent


def random_fit(f: Fit, seed: int) -> Fit:
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(f.d.shape, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    return Fit(d, f.mu_instr, f.mu_non, f.n_instr, f.n_non)


@torch.no_grad()
def answer_matched(model, tok, rendering, injection, fit, layers, alpha, gate, seed) -> tuple[str, float]:
    """Generate with NXDirection's recorded per-(layer, token) edit sizes replayed along random directions."""
    r = render(tok, parts_for(rendering, injection), system=SYSTEM)
    ids = torch.tensor([r.input_ids], device=model.device)
    mask, bias = torch.tensor(r.untrusted_mask), gate_bias(fit, gate)
    coefs = record_coefs(model, ids, fit.d, layers, alpha, mask, bias)
    with patched_edit(replay_edit(coefs, fit.d, random_unit(fit.d.shape, 100 + seed))), \
            nx_hooks(model, fit.d, layers, alpha, mask, bias=bias):
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=160, do_sample=False)
    total = float(sum(c.sum() for c in coefs.values()))
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), total


def run_matched(model, tok, fit, *, rendering, layers, alpha, gate, injection, seed, undefended=None) -> dict:
    und = undefended if undefended is not None else answer(model, tok, rendering, injection)
    dfd, total = answer_matched(model, tok, rendering, injection, fit, layers, alpha, gate, seed)
    cln, _ = answer_matched(model, tok, rendering, None, fit, layers, alpha, gate, seed)
    return {"undefended": und, "defended": dfd, "defended_clean": cln, "replayed_coef_total": total,
            "pass_undefended_compromised": is_compromised(und), "pass_defended_correct": is_correct(dfd),
            "pass_clean_correct": is_correct(cln)}


def parse_layers(s: str) -> list[int]:
    a, b = map(int, s.split("-"))
    return list(range(a, b + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", default="user")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gate", default="raw")
    ap.add_argument("--rendering", default="md_before")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--paraphrase-direction", choices=["real", "matched"], default="real")
    ap.add_argument("--what", choices=["random", "neg", "matched", "amplify", "paraphrase"], required=True)
    args = ap.parse_args()
    model, tok = load(args.model)
    fit = load_fit(out_dir(args.model, args.context) / "fit.safetensors")
    layers = parse_layers(args.layers)
    kw = dict(rendering=args.rendering, layers=layers, alpha=args.alpha, gate=args.gate)
    results = []
    if args.what == "random":
        for seed in range(5):
            res = run(model, tok, random_fit(fit, seed), injection=INJECTION, **kw)
            results.append({"seed": seed, **res})
            print(f"seed={seed} compromised={'compromised' in res['defended'].lower()} D={res['pass_defended_correct']} "
                  f"C={res['pass_clean_correct']} :: {res['defended'][:80]!r}", flush=True)
    elif args.what == "matched":
        und = answer(model, tok, args.rendering, INJECTION)
        for seed in args.seeds:
            res = run_matched(model, tok, fit, injection=INJECTION, seed=seed, undefended=und, **kw)
            results.append({"seed": seed, **res})
            print(f"matched seed={seed} coef_total={res['replayed_coef_total']:.1f} "
                  f"compromised={is_compromised(res['defended'])} D={res['pass_defended_correct']} "
                  f"C={res['pass_clean_correct']} :: {res['defended'][:70]!r}", flush=True)
    elif args.what == "amplify":
        import hook as hook_mod

        real_edit = hook_mod.edit
        hook_mod.edit = lambda h, d, alpha, b: h + (alpha * torch.relu(h @ d - b))[:, None] * d
        try:
            res = run(model, tok, fit, injection=INJECTION, **kw)
        finally:
            hook_mod.edit = real_edit
        results.append(res)
        print(f"amplify compromised={'compromised' in res['defended'].lower()} D={res['pass_defended_correct']} "
              f"C={res['pass_clean_correct']} :: {res['defended'][:80]!r}", flush=True)
    elif args.what == "neg":
        flipped = Fit(-fit.d, fit.mu_instr, fit.mu_non, fit.n_instr, fit.n_non)
        res = run(model, tok, flipped, injection=INJECTION, **kw)
        results.append(res)
        print(f"neg compromised={'compromised' in res['defended'].lower()} D={res['pass_defended_correct']} "
              f"C={res['pass_clean_correct']} :: {res['defended'][:80]!r}", flush=True)
    else:
        seeds = args.seeds if args.paraphrase_direction == "matched" else [None]
        for inj in json.loads((HERE / "paraphrases.json").read_text()):
            und = answer(model, tok, args.rendering, inj)
            for seed in seeds:
                if seed is None:
                    res = run(model, tok, fit, injection=inj, undefended=und, **kw)
                else:
                    res = run_matched(model, tok, fit, injection=inj, seed=seed, undefended=und, **kw)
                results.append({"injection": inj, "seed": seed, **res})
                print(f"inj={inj[:40]!r} seed={seed}\n   und={res['undefended'][:70]!r}\n   def={res['defended'][:70]!r} "
                      f"C={res['pass_clean_correct']}", flush=True)
    path = HERE / "out" / f"control_{args.what}{'_matched' if args.what == 'paraphrase' and args.paraphrase_direction == 'matched' else ''}_{short(args.model)}_{args.layers}_a{args.alpha}_{args.gate}_{args.rendering}.json"
    path.write_text(json.dumps({"config": vars(args), "results": results}, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
