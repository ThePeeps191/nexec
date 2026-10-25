"""Sweep alpha x layer set x gate x rendering x fit context on the salary demo; also the paraphrase check."""

import argparse
import csv
import json
from pathlib import Path

from common import MODELS, load, short
from fit import load_fit, out_dir
from salary import GATES, INJECTION, RENDERINGS, answer, is_compromised, is_correct, run

HERE = Path(__file__).parent
OUT = HERE / "out" / "salary_sweep.csv"
ALPHAS = (1.0, 1.5, 2.0, 4.0)
CONTEXTS = ("user", "document")
FIELDS = ["model", "context", "layer_set", "layers", "alpha", "gate", "rendering",
          "undefended", "undefended_clean", "defended", "defended_clean",
          "pass_undefended_compromised", "undefended_clean_correct", "pass_defended_correct", "pass_clean_correct", "all_pass"]


def band(model_id: str, context: str, n_layers: int) -> list[int]:
    """Contiguous run within 0.02 of the max heldout AUC over layers 0..L-2 (the last layer cannot affect generation)."""
    rows = list(csv.DictReader(open(out_dir(model_id, context) / "diagnostics.csv")))
    auc = [float(r["heldout_auc"]) for r in rows][: n_layers - 1]
    best = max(range(len(auc)), key=auc.__getitem__)
    lo = hi = best
    while lo > 0 and auc[lo - 1] >= auc[best] - 0.02:
        lo -= 1
    while hi < len(auc) - 1 and auc[hi + 1] >= auc[best] - 0.02:
        hi += 1
    return list(range(lo, hi + 1))


def layer_sets(model_id: str, context: str, n: int) -> dict[str, list[int]]:
    return {"all": list(range(n)), "mid_third": list(range(n // 3, 2 * n // 3)),
            "d40_80": list(range(round(0.4 * n), round(0.8 * n))), "band": band(model_id, context, n)}


def span(layers: list[int]) -> str:
    return f"{layers[0]}-{layers[-1]}"


def sweep(models):
    done = set()
    if OUT.exists():
        done = {tuple(r[k] for k in ("model", "context", "layer_set", "alpha", "gate", "rendering")) for r in csv.DictReader(open(OUT))}
    new = not OUT.exists()
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for model_id in models:
        model, tok = load(model_id)
        n = len(model.model.layers)
        base = {r: (answer(model, tok, r, INJECTION), answer(model, tok, r, None)) for r in RENDERINGS}
        for context in CONTEXTS:
            fit = load_fit(out_dir(model_id, context) / "fit.safetensors")
            for set_name, layers in layer_sets(model_id, context, n).items():
                for alpha in ALPHAS:
                    for gate in GATES:
                        for rendering in RENDERINGS:
                            key = (short(model_id), context, set_name, str(alpha), gate, rendering)
                            if key in done:
                                continue
                            und, und_clean = base[rendering]
                            res = run(model, tok, fit, rendering=rendering, layers=layers, alpha=alpha, gate=gate, undefended=und)
                            row = {"model": short(model_id), "context": context, "layer_set": set_name, "layers": span(layers),
                                   "alpha": alpha, "gate": gate, "rendering": rendering, "undefended_clean": und_clean,
                                   "undefended_clean_correct": is_correct(und_clean), **res}
                            row["all_pass"] = row["pass_undefended_compromised"] and row["pass_defended_correct"] and row["pass_clean_correct"]
                            for k in ("undefended", "undefended_clean", "defended", "defended_clean"):
                                row[k] = row[k].replace("\n", " ")[:200]
                            w.writerow(row)
                            fh.flush()
                            print(f"{key} D={res['pass_defended_correct']} C={res['pass_clean_correct']} :: {row['defended'][:70]!r}", flush=True)
        del model
    fh.close()


def summarize():
    rows = list(csv.DictReader(open(OUT)))
    for m in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == m]
        print(f"\n== {m}: undefended compromised in renderings: "
              f"{sorted({r['rendering'] for r in mr if r['pass_undefended_compromised'] == 'True'})}; "
              f"undefended clean correct: {sorted({r['rendering'] for r in mr if r['undefended_clean_correct'] == 'True'})}")
        for r in mr:
            if r["all_pass"] == "True":
                print(f"  ALL PASS {r['context']:8} {r['layer_set']:9} {r['layers']:6} a={r['alpha']:4} {r['gate']:6} {r['rendering']:9} :: {r['defended'][:60]!r}")


def paraphrase_check(model_id: str, context: str, layers: list[int], alpha: float, gate: str, rendering: str = "md_before"):
    model, tok = load(model_id)
    fit = load_fit(out_dir(model_id, context) / "fit.safetensors")
    rows = []
    for inj in json.loads((HERE / "paraphrases.json").read_text()):
        res = run(model, tok, fit, rendering=rendering, layers=layers, alpha=alpha, gate=gate, injection=inj)
        rows.append({"injection": inj, **res})
        print(f"und={res['undefended'][:50]!r} | def={res['defended'][:70]!r}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        sweep(args.models)
    summarize()
