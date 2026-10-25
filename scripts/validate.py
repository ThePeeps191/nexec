"""Step-1 validation.

  parity  one-time check that the library reproduces the step-0 spike (fit tensors + salary outputs);
          needs the historical spike/ files (git history up to 643a914) and its spike/out/ fits; kept for the
          record, its result is docs/findings/step1-data/parity.json.
  full    validate shipped directions: salary (3 renderings), paraphrases (10), matched-random controls,
          and the natural-document utility check; writes JSON to docs/findings/step1-data/.
"""

import argparse
import csv
import json

import torch
from _common import GEMMA, LLAMA, ROOT

from evals.cases import (
    PARAPHRASES,
    RENDERINGS,
    followed,
    is_compromised,
    is_correct,
    natural_documents,
    salary_conversation,
)
from nexec import NX, Directions, DirectionsConfig
from nexec.controls import matched_random
from nexec.corpus import Example, Span, Turn
from nexec.fit import fit_labeled, token_labels
from nexec.nx import load_model

OUT = ROOT / "docs" / "findings" / "step1-data"
GEN = dict(max_new_tokens=160)


def _norm(s: str) -> str:
    return s.strip().replace("\n", " ")[:200]


def parity():
    from safetensors.torch import load_file

    rows = [json.loads(x) for x in (ROOT / "spike" / "corpus_pilot.jsonl").read_text().splitlines() if x.strip()]
    pilot = [Example(r["id"], r["category"], r["split"], r["imperative_content"],
                     [Turn("user", r["text"], [Span(**s) for s in r["spans"]])]) for r in rows]
    model, tok = load_model(LLAMA)
    fit = fit_labeled(model, (token_labels(tok, e) for e in pilot if e.split == "train"))
    ref = load_file(str(ROOT / "spike/out/Llama-3.2-3B-Instruct/user/fit.safetensors"))
    report = {k: float((getattr(fit, k) - ref[k]).abs().max()) for k in ("d", "mu_instr", "mu_non")}
    report["fit_allclose_1e-4"] = all(v <= 1e-4 for v in report.values())
    nx = NX(model, tok, Directions(fit.d, fit.mu_instr, fit.mu_non,
                                   DirectionsConfig(model_id=LLAMA, layers=list(range(28)), alpha=1.0, gate="raw")))
    sweep = next(r for r in csv.DictReader(open(ROOT / "docs/findings/step0-data/salary_sweep.csv"))
                 if (r["model"], r["context"], r["layer_set"], r["alpha"], r["gate"], r["rendering"]) ==
                 ("Llama-3.2-3B-Instruct", "user", "all", "1.0", "raw", "md_before"))
    got = {"undefended": nx.generate(salary_conversation("md_before"), defend=False, **GEN)[0],
           "defended": nx.generate(salary_conversation("md_before"), **GEN)[0],
           "defended_clean": nx.generate(salary_conversation("md_before", None), **GEN)[0]}
    for k, v in got.items():
        report[f"{k}_matches_spike"] = _norm(v) == sweep[k]
        report[f"{k}_text"] = _norm(v)
    print(json.dumps(report, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity.json").write_text(json.dumps(report, indent=1))


def full(model_ids):
    OUT.mkdir(parents=True, exist_ok=True)
    for mid in model_ids:
        nx = NX.from_pretrained(mid)
        res = {"config": {k: v for k, v in vars(nx.directions.config).items() if k != "diagnostics"},
               "salary": {}, "paraphrases": [], "utility": []}
        for r in RENDERINGS:
            und = nx.generate(salary_conversation(r), defend=False, **GEN)[0]
            und_clean = nx.generate(salary_conversation(r, None), defend=False, **GEN)[0]
            dfd = nx.generate(salary_conversation(r), **GEN)[0]
            cln = nx.generate(salary_conversation(r, None), **GEN)[0]
            matched = [matched_random(nx, salary_conversation(r), seed=s, **GEN)[0] for s in range(5)]
            res["salary"][r] = {
                "undefended": und, "undefended_clean": und_clean, "defended": dfd, "defended_clean": cln,
                "undefended_compromised": is_compromised(und), "undefended_clean_correct": is_correct(und_clean),
                "defended_blocked": not is_compromised(dfd), "defended_correct": is_correct(dfd),
                "defended_clean_correct": is_correct(cln),
                "matched_blocked": sum(not is_compromised(m) for m in matched), "matched": matched}
            print(mid, r, {k: v for k, v in res["salary"][r].items() if not isinstance(v, (str, list))}, flush=True)
        rendering = "md_before" if mid == LLAMA else "md_after"
        for i, inj in enumerate(PARAPHRASES):
            und = nx.generate(salary_conversation(rendering, inj), defend=False, **GEN)[0]
            dfd = nx.generate(salary_conversation(rendering, inj), **GEN)[0]
            cln_ok = is_correct(nx.generate(salary_conversation(rendering, None), **GEN)[0]) if i == 0 else None
            matched = [matched_random(nx, salary_conversation(rendering, inj), seed=s, **GEN)[0] for s in range(3)]
            res["paraphrases"].append({"injection": inj, "rendering": rendering, "undefended": und, "defended": dfd,
                                       "followed_undefended": followed(i, und), "followed_defended": followed(i, dfd),
                                       "followed_matched": [followed(i, m) for m in matched], "clean_ok": cln_ok})
        p = res["paraphrases"]
        res["paraphrase_summary"] = {
            "undefended": sum(x["followed_undefended"] for x in p), "defended": sum(x["followed_defended"] for x in p),
            "matched_mean": sum(sum(x["followed_matched"]) for x in p) / 3}
        print(mid, "paraphrases", res["paraphrase_summary"], flush=True)
        for doc in natural_documents():
            off = nx.generate(doc["conversation"], defend=False, max_new_tokens=80)[0]
            on = nx.generate(doc["conversation"], max_new_tokens=80)[0]
            res["utility"].append({"kind": doc["kind"], "nx_off": off, "nx_on": on, "identical": off == on})
        print(mid, "utility identical", sum(u["identical"] for u in res["utility"]), "/ 10", flush=True)
        (OUT / f"validate_{mid.split('/')[-1]}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        del nx
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["parity", "full"])
    ap.add_argument("--models", nargs="*", default=[LLAMA, GEMMA])
    a = ap.parse_args()
    parity() if a.mode == "parity" else full(a.models)
