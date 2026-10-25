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
from salary import INJECTION, run

HERE = Path(__file__).parent


def random_fit(f: Fit, seed: int) -> Fit:
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(f.d.shape, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    return Fit(d, f.mu_instr, f.mu_non, f.n_instr, f.n_non)


def make_matched_edit(fit: Fit, seed: int):
    """Same tokens and per-token edit size as NXDirection, but along a random unit direction per layer."""
    rnd = random_fit(fit, 100 + seed).d

    def matched_edit(h, d, alpha, b):
        l = int(torch.argmax(fit.d.to(d.device) @ d))  # identify the layer by its direction
        coef = alpha * torch.relu(h @ d - b)
        return h - coef[:, None] * rnd[l].to(h.device)

    return matched_edit


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
        import hook as hook_mod

        real_edit = hook_mod.edit
        for seed in range(5):
            rnd = random_fit(fit, 100 + seed).d
            norms = []

            def matched_edit(h, d, alpha, b, rnd=rnd, norms=norms):
                # Same tokens and per-token edit size as NXDirection, random direction for this layer.
                l = int(torch.argmax((fit.d.to(d.device) @ d)))  # identify layer by its direction
                coef = alpha * torch.relu(h @ d - b)
                norms.append(float(coef.abs().mean()))
                return h - coef[:, None] * rnd[l].to(h.device)

            hook_mod.edit = matched_edit
            try:
                res = run(model, tok, fit, injection=INJECTION, **kw)
            finally:
                hook_mod.edit = real_edit
            results.append({"seed": seed, **res})
            print(f"matched seed={seed} mean|coef|={sum(norms)/max(len(norms),1):.2f} "
                  f"compromised={'compromised' in res['defended'].lower()} D={res['pass_defended_correct']} "
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
        import hook as hook_mod

        real_edit = hook_mod.edit
        if args.paraphrase_direction == "matched":
            hook_mod.edit = make_matched_edit(fit, 0)
        for inj in json.loads((HERE / "paraphrases.json").read_text()):
            res = run(model, tok, fit, injection=inj, **kw)
            results.append({"injection": inj, **res})
            print(f"inj={inj[:40]!r}\n   und={res['undefended'][:70]!r}\n   def={res['defended'][:70]!r} C={res['pass_clean_correct']}", flush=True)
        hook_mod.edit = real_edit
    path = HERE / "out" / f"control_{args.what}{'_matched' if args.what == 'paraphrase' and args.paraphrase_direction == 'matched' else ''}_{short(args.model)}_{args.layers}_a{args.alpha}_{args.gate}_{args.rendering}.json"
    path.write_text(json.dumps({"config": vars(args), "results": results}, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
