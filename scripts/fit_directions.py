"""Fit directions with the packaged corpus and write them to the bundled directory (or --out)."""

import argparse
import json

from _common import SHIPPED

from nexec.directions import BUNDLED_DIR, slug
from nexec.fit import fit_directions
from nexec.nx import choose_gate, load_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate", choices=["raw", "affine", "mid", "auto"], default=None,
                    help="default: the shipped choice for known models, else auto (spec §5.3 rule)")
    ap.add_argument("--context", choices=["user", "document"], default="user")
    ap.add_argument("--alpha", type=float, default=None, help="default: the shipped choice for known models, else 1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, tok = load_model(a.model)
    alpha = a.alpha if a.alpha is not None else SHIPPED.get(a.model, {}).get("alpha", 1.0)
    dirs = fit_directions(model, tok, a.model, context=a.context, alpha=alpha)
    gate = a.gate or SHIPPED.get(a.model, {}).get("gate", "auto")
    dirs.config.gate = choose_gate(dirs.config.diagnostics) if gate == "auto" else gate
    out = dirs.save(a.out or BUNDLED_DIR / slug(a.model))
    diag = dirs.config.diagnostics
    auc = diag.get("heldout_auc", [])
    print(json.dumps({"out": str(out), "gate": dirs.config.gate, "n_instr": diag["n_instr"], "n_non": diag["n_non"],
                      "heldout_auc_min_mid": min(auc[len(auc) // 4: 3 * len(auc) // 4]) if auc else None,
                      "heldout_frac_non_pos_median": sorted(diag.get("heldout_frac_non_pos", [0]))[len(auc) // 2]
                      if auc else None}, indent=1))
