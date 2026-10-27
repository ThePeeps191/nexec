"""Free experiments on BIPIA's rule-scored test attacks (emoji, language, code snippets: no judge calls).

    sweep:  ASR per family under gate x alpha configs (all layers)
    prime:  the same injections with the imperative removed (emoji: a declarative mention; code: the snippet alone),
            undefended and NX: does the model still emit emojis / paste the snippet with no instruction at all?

    uv run python scripts/rule_sweep.py sweep --model meta-llama/Llama-3.2-3B-Instruct
    uv run python scripts/rule_sweep.py prime --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import dataclasses
import json
import random
from collections import defaultdict
from pathlib import Path

from evals.bipia.asr import Rule, evaluator
from evals.bipia.data import attack_text
from evals.bipia.pipeline import MAX_NEW, _items, samples
from evals.bipia.run import read_records, run
from nexec import NX

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/findings/step3.5-data"
CONFIGS = [("raw", 1.0), ("raw", 2.0), ("raw", 4.0), ("affine", 1.0), ("affine", 2.0), ("affine", 4.0)]
EMOJI_MENTION = "Many readers enjoy emojis."


def family(attack: str, task: str) -> str:
    typ = attack.rsplit("-", 1)[0]
    return "code" if task == "code" else typ


def rule_samples():
    return [s for s in samples("test")[0] if isinstance(evaluator(s.attack, split="test"), Rule)]


def score(path, by_uid) -> dict:
    c = defaultdict(lambda: [0, 0])
    for r in read_records(path):
        s = by_uid[r["uid"]]
        ok = evaluator(s.attack, split="test")(r["output"])
        f = family(s.attack, s.task)
        c[f][0] += ok
        c[f][1] += 1
        c["all"][0] += ok
        c["all"][1] += 1
    return {f: {"k": k, "n": n, "asr": k / n} for f, (k, n) in sorted(c.items())}


def _short(model):
    return model.split("/")[-1]


def sweep(nx, model):
    ss = rule_samples()
    by_uid = {s.uid: s for s in ss}
    shipped_gate = nx.directions.config.gate
    res = {}
    for gate, alpha in CONFIGS:
        tag = f"{gate}_a{alpha:g}"
        path = OUT / "gen" / _short(model) / f"sweep_{tag}.jsonl.gz"
        nx.directions.config.gate = gate
        run(nx, "nx", _items(ss), path, alpha=alpha, max_new_tokens=MAX_NEW)
        res[tag] = score(path, by_uid)
        print(tag, {f: round(100 * v["asr"], 1) for f, v in res[tag].items()}, flush=True)
    nx.directions.config.gate = shipped_gate
    return res


def primed(s):
    atk = attack_text(s.attack, s.task, s.split)
    if s.task == "code":
        new = atk[atk.index("\n") + 1:]  # the fenced snippet alone, no "Add the following code snippet..." sentence
    else:
        new = EMOJI_MENTION
    return dataclasses.replace(s, uid=s.uid + "/primed", context=s.context.replace(atk, new, 1))


def prime(nx, model, n=60):
    ss = rule_samples()
    rng = random.Random(0)
    groups = {"emoji": [s for s in ss if s.attack.startswith("Emoji")], "code": [s for s in ss if s.task == "code"]}
    res = {}
    for g, pool in groups.items():
        pick = rng.sample(pool, min(n, len(pool)))
        orig = {s.uid: s for s in pick}
        pr = {p.uid: (p, s) for s in pick for p in [primed(s)]}
        for cond in ("undefended", "nx"):
            for kind, items in (("injected", list(orig.values())), ("primed", [p for p, _ in pr.values()])):
                path = OUT / "gen" / _short(model) / f"prime_{g}_{kind}_{cond}.jsonl.gz"
                run(nx, cond, _items(items), path, max_new_tokens=MAX_NEW)
                k = 0
                recs = read_records(path)
                for r in recs:
                    s = orig[r["uid"]] if kind == "injected" else pr[r["uid"]][1]
                    k += evaluator(s.attack, split="test")(r["output"])
                res[f"{g}/{kind}/{cond}"] = {"k": k, "n": len(recs), "rate": k / len(recs)}
                print(g, kind, cond, f"{100 * k / len(recs):.1f}%", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "prime"])
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    nx = NX.from_pretrained(a.model)
    res = sweep(nx, a.model) if a.cmd == "sweep" else prime(nx, a.model)
    out = OUT / f"{a.cmd}_{_short(a.model)}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
