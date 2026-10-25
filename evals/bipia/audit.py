"""Judge quality checks: (1) stability = re-judge a random sample in a fresh batch and measure agreement;
(2) export a stratified sample (ASR / utility) for a hand audit in Claude Code."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from evals.bipia.judge import judge_many, load_cache
from evals.bipia.pipeline import CACHE

STABILITY_CACHE = CACHE.with_name("stability_cache.jsonl")


def _kind(prompt: str) -> str:
    return "asr" if "Concisely reply in YES, NO, or UNKNOWN." in prompt else "utility"


def stability(n: int = 200, seed: int = 0) -> dict:
    recs = [r for r in load_cache(CACHE).values() if r.verdict in ("yes", "no", "unknown")]
    sample = random.Random(seed).sample(recs, min(n, len(recs)))
    again = judge_many({r.key: r.prompt for r in sample}, cache_path=STABILITY_CACHE)
    agree = sum(again[r.key].verdict == r.verdict for r in sample)
    by_kind = {}
    for r in sample:
        k = _kind(r.prompt)
        a, t = by_kind.get(k, (0, 0))
        by_kind[k] = (a + (again[r.key].verdict == r.verdict), t + 1)
    return {"n": len(sample), "agreement": agree / len(sample),
            "by_kind": {k: {"agree": a, "n": t, "rate": a / t} for k, (a, t) in by_kind.items()}}


def export_audit(path, n_per_kind: int = 50, seed: int = 0) -> int:
    recs = [r for r in load_cache(CACHE).values() if r.verdict in ("yes", "no", "unknown")]
    rng = random.Random(seed)
    out = []
    for kind in ("asr", "utility"):
        pool = [r for r in recs if _kind(r.prompt) == kind]
        out += [{"id": r.key, "kind": kind, "prompt": r.prompt, "judge_verdict": r.verdict}
                for r in rng.sample(pool, min(n_per_kind, len(pool)))]
    Path(path).write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    return len(out)


if __name__ == "__main__":
    if sys.argv[1] == "stability":
        print(json.dumps(stability(), indent=1))
    else:
        print(export_audit(sys.argv[2]))
