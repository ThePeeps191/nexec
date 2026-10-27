"""Step 3.5 selection: apply the pre-registered rule in docs/findings/2026-10-step3.5-residual.md.

Per candidate (gate, alpha; all layers): dev2 rule ASR (mean over emoji / language / code families), step-3 dev ASR
(Claude-judged, mean over tasks) and dev clean utility. Dev tags from step 3 are reused where the settings match
(all_a1 / all_a2 were generated with each model's shipped gate).

    uv run python scripts/select_v35.py --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals.bipia import dev2
from evals.bipia.pipeline import MAX_NEW, NVIDIA_DROP, _items, generate, prejudge, score_tag
from evals.bipia.run import read_records, run
from nexec import NX

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/findings/step3.5-data"
GRID = {"meta-llama/Llama-3.2-3B-Instruct": [("raw", 1.0), ("raw", 2.0), ("raw", 4.0),
                                             ("affine", 1.0), ("affine", 2.0), ("affine", 4.0)],
        "google/gemma-2-2b-it": [("affine", 1.0), ("affine", 2.0)]}
SHIPPED_GATE = {"meta-llama/Llama-3.2-3B-Instruct": "raw", "google/gemma-2-2b-it": "affine"}


def dev_tag(model, gate, alpha):
    if gate == SHIPPED_GATE[model] and alpha in (1.0, 2.0):
        return f"all_a{alpha:g}"  # step-3 dev grid
    return f"v35_{gate}_a{alpha:g}"


def dev2_rate(path, by_uid) -> dict:
    c = defaultdict(lambda: [0, 0])
    for r in read_records(path):
        s = by_uid[r["uid"]]
        f = dev2.family(s)
        c[f][0] += dev2.success(s, r["output"])
        c[f][1] += 1
    fam = {f: k / n for f, (k, n) in sorted(c.items())}
    return {"families": fam, "mean": sum(fam.values()) / len(fam), "n": sum(n for _, n in c.values())}


def choose(model, base_u, cands: dict) -> str:
    cap = max(0.02, NVIDIA_DROP.get(model, 0.02))
    ok = {t: c for t, c in cands.items() if base_u - c["dev_utility"] <= cap + 1e-9}
    if not ok:
        return "shipped"
    best = min(c["objective"] for c in ok.values())
    tied = [t for t, c in ok.items() if c["objective"] == best]
    return sorted(tied, key=lambda t: (ok[t]["alpha"], ok[t]["gate"] != SHIPPED_GATE[model]))[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    m, short = a.model, a.model.split("/")[-1]
    nx = NX.from_pretrained(m)
    ss = dev2.build()
    by_uid = {s.uid: s for s in ss}
    run(nx, "undefended", _items(ss), OUT / "gen" / short / "dev2_undefended.jsonl.gz", max_new_tokens=MAX_NEW)
    for gate, alpha in GRID[m]:
        nx.directions.config.gate = gate
        run(nx, "nx", _items(ss), OUT / "gen" / short / f"dev2_{gate}_a{alpha:g}.jsonl.gz", alpha=alpha,
            max_new_tokens=MAX_NEW)
        generate(nx, m, "dev", "nx", dev_tag(m, gate, alpha), alpha=alpha)
    nx.directions.config.gate = SHIPPED_GATE[m]
    tags = ["undefended"] + [dev_tag(m, g, al) for g, al in GRID[m]]
    prejudge([(m, t) for t in tags], "dev")
    base = score_tag(m, "dev", "undefended")
    cands = {}
    for gate, alpha in GRID[m]:
        d = score_tag(m, "dev", dev_tag(m, gate, alpha))
        r2 = dev2_rate(OUT / "gen" / short / f"dev2_{gate}_a{alpha:g}.jsonl.gz", by_uid)
        cands[f"{gate}_a{alpha:g}"] = {"gate": gate, "alpha": alpha, "dev2_rule_asr": r2, "dev_asr": d["asr"]["mean"],
                                       "dev_utility": d["utility"]["mean"],
                                       "objective": (r2["mean"] + d["asr"]["mean"]) / 2}
    summary = {"model": m, "baseline": {"dev_asr": base["asr"]["mean"], "dev_utility": base["utility"]["mean"],
                                        "dev2_rule_asr": dev2_rate(OUT / "gen" / short / "dev2_undefended.jsonl.gz",
                                                                   by_uid)},
               "candidates": cands, "choice": choose(m, base["utility"]["mean"], cands)}
    (OUT / f"select_{short}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: {"obj": round(100 * c["objective"], 1), "dev2": round(100 * c["dev2_rule_asr"]["mean"], 1),
                          "dev_asr": round(100 * c["dev_asr"], 1), "util": round(100 * c["dev_utility"], 1)}
                      for k, c in cands.items()}, indent=1))
    print("baseline util", base["utility"]["mean"], "choice:", summary["choice"])
