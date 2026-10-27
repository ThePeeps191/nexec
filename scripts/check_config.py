"""Ordinary-document checks for a candidate (gate, alpha): gate firing on 10 natural documents (mid third of layers,
spec §8 test 4), doc-QA answers identical with NX on vs off, and the salary demo answer. The clean salary check uses
the md_after layout (Gemma's demo); undefended Llama already misses that one, its demo layout is md_before.

    PYTHONPATH=. uv run python scripts/check_config.py --model meta-llama/Llama-3.2-3B-Instruct --configs raw:1 affine:4
"""

import argparse
import json
from pathlib import Path

import torch

from evals.cases import is_correct, natural_documents, salary_conversation
from nexec import NX

OUT = Path(__file__).resolve().parents[1] / "docs/findings/step3.5-data"
GEN = dict(max_new_tokens=160)


@torch.no_grad()
def check(nx, gate, alpha):
    nx.directions.config.gate = gate
    L = len(nx.model.model.layers)
    mid = range(L // 3, 2 * L // 3)
    docs = []
    for doc in natural_documents():
        enc = nx.encode(doc["conversation"])
        rec = {}
        with nx.protect(enc.untrusted_mask, alpha=alpha, record=rec):
            nx.model(enc.input_ids.to(nx.model.device), attention_mask=enc.attention_mask.to(nx.model.device),
                     logits_to_keep=1)
        m = enc.untrusted_mask
        frac = sum(float((rec[l][m] > 0).float().mean()) for l in mid) / len(mid)
        off = nx.generate(doc["conversation"], defend=False, **GEN)[0]
        on = nx.generate(doc["conversation"], alpha=alpha, **GEN)[0]
        docs.append({"kind": doc["kind"], "fire_mid": frac, "identical": off == on, "nx_off": off, "nx_on": on})
    salary = {layout: is_correct(nx.generate(salary_conversation(layout), alpha=alpha, **GEN)[0])
              for layout in ("md_after", "md_before")}
    salary_clean = is_correct(nx.generate(salary_conversation("md_after", None), alpha=alpha, **GEN)[0])
    return {"gate": gate, "alpha": alpha, "fire_mid_mean": sum(d["fire_mid"] for d in docs) / len(docs),
            "fire_mid_max": max(d["fire_mid"] for d in docs), "identical": sum(d["identical"] for d in docs),
            "n_docs": len(docs), "salary_repaired": salary, "salary_clean_correct": salary_clean, "docs": docs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", required=True, help="gate:alpha, e.g. raw:1 affine:4")
    a = ap.parse_args()
    nx = NX.from_pretrained(a.model)
    res = [check(nx, c.split(":")[0], float(c.split(":")[1])) for c in a.configs]
    out = OUT / f"check_{a.model.split('/')[-1]}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for r in res:
        print(r["gate"], r["alpha"], f"fire_mid {100 * r['fire_mid_mean']:.1f}% (max {100 * r['fire_mid_max']:.1f}%)",
              f"identical {r['identical']}/{r['n_docs']}", "salary", r["salary_repaired"],
              "clean", r["salary_clean_correct"])
