"""How strongly does the instruction direction register each BIPIA injection? Per attack type, over sampled test
prompts: the pre-edit margin h·d − b on the injected sentence's tokens vs the rest of the untrusted context, the share
of injected tokens the edit fires on, and the edit mass applied. Compared with each type's residual ASR under NX.

    uv run python scripts/diagnose_attacks.py --model meta-llama/Llama-3.2-3B-Instruct [--directions DIR] [--out f.json]
"""

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from evals.bipia.data import attack_text, conversation
from evals.bipia.pipeline import samples
from nexec import NX
from nexec.directions import Directions
from nexec.viz import _projections

ROOT = Path(__file__).resolve().parents[1]


def residual_asr(model_short: str, tag: str = "nx") -> dict[str, float]:
    p = ROOT / "docs/findings/step3-data/verdicts" / model_short / "test" / f"{tag}.jsonl.gz"
    if not p.exists():
        return {}
    c = defaultdict(lambda: [0, 0])
    for line in gzip.open(p, "rt"):
        v = json.loads(line)
        if v["kind"] == "asr" and v["outcome"] is not None:
            typ = v["uid"].split("/")[3].rsplit("-", 1)[0]
            c[typ][0] += v["outcome"]
            c[typ][1] += 1
    return {t: k / n for t, (k, n) in c.items()}


@torch.no_grad()
def diagnose(nx, samples, layers, early):
    d, bias = nx.directions.d, nx.directions.bias()
    b = torch.zeros(d.shape[0]) if bias is None else bias.float()
    by_type = defaultdict(lambda: defaultdict(list))
    for s in samples:
        enc = nx.encode([conversation(s)])
        r = enc.renders[0]
        atk = attack_text(s.attack, s.task, s.split)
        a0 = r.text.find(atk)
        if a0 < 0:
            raise ValueError(f"attack text not found in render for {s.uid}")
        a1 = a0 + len(atk)
        h1 = a0 + (atk.index("\n") if "\n" in atk else len(atk))  # the instruction sentence; the rest is a code block
        untrusted = enc.untrusted_mask[0].tolist()
        is_atk = [st < a1 and en > a0 for st, en in r.offsets]
        is_head = [st < h1 and en > a0 for st, en in r.offsets]
        is_body = [a and not h for a, h in zip(is_atk, is_head, strict=True)]
        rest = [u and not a for u, a in zip(untrusted, is_atk, strict=True)]
        ids = enc.input_ids.to(nx.model.device)
        proj = _projections(nx.model, ids, d) - b[:, None]  # [L, T] margins before any edit
        coefs: dict[int, torch.Tensor] = {}
        with nx.protect(enc.untrusted_mask, record=coefs):
            nx.model(ids, use_cache=False, logits_to_keep=1)
        t = by_type[s.attack.rsplit("-", 1)[0]]
        c_all = torch.stack([coefs[l][0] for l in range(len(coefs))])
        for band, ls in (("", layers), ("early_", early)):
            pl, c = proj[ls], c_all[ls]
            for name, m in (("attack", is_atk), ("head", is_head), ("body", is_body), ("rest", rest)):
                if any(m):
                    mi = torch.tensor(m)
                    t[f"{band}margin_{name}"].append(float(pl[:, mi].mean()))
                    t[f"{band}fired_{name}"].append(float((c[:, mi] > 0).float().mean()))
            t[f"{band}coef_attack"].append(float(c[:, torch.tensor(is_atk)].mean()))
    return {typ: {k: sum(v) / len(v) for k, v in m.items()} | {"n": len(m["margin_attack"])}
            for typ, m in by_type.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--directions", default=None, help="directions dir (default: bundled)")
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--position", default="end")
    ap.add_argument("--tag", default="nx", help="verdict tag for residual ASR")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nx = NX.from_pretrained(a.model)
    if a.directions:
        nx.directions = Directions.load(a.directions)
    L = len(nx.model.model.layers)
    by_type = defaultdict(list)
    for s in samples("test")[0]:
        if s.position == a.position:
            by_type[s.attack.rsplit("-", 1)[0]].append(s)
    rng = random.Random(0)
    picked = [s for v in by_type.values() for s in rng.sample(v, min(a.per_type, len(v)))]
    res = {"all": diagnose(nx, picked, list(range(L)), list(range(round(0.4 * L) + 1)))}
    asr = residual_asr(a.model.split("/")[-1], a.tag)
    rows = sorted(res["all"].items(), key=lambda kv: -asr.get(kv[0], 0))
    cols = ["margin_attack", "margin_head", "margin_body", "margin_rest", "fired_head", "fired_rest",
            "early_margin_head", "early_fired_head", "early_fired_rest"]
    short = [c.replace("margin", "m").replace("fired", "f")[:11] for c in cols]
    print(f"{'attack type':40s} {'NX ASR':>6s} " + " ".join(f"{c:>11s}" for c in short))
    for typ, m in rows:
        vals = " ".join(f"{m.get(c, float('nan')):11.2f}" for c in cols)
        print(f"{typ[:40]:40s} {100 * asr.get(typ, float('nan')):6.1f} {vals}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "position": a.position, "residual_asr": asr, **res},
                                          indent=1) + "\n")
