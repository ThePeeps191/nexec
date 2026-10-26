"""Secondary analyses for the step-3 report, from the committed verdict files only (no API calls):
ASR by scorer, sensitivity to "unknown" judge verdicts, and paired clean-utility differences.

    uv run python -m evals.bipia.analysis      # writes docs/findings/step3-data/scores/analysis.json
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

from evals.bipia.report import DATA
from evals.bipia.score import paired_diff, summarize_asr

TAGS = {"Llama-3.2-3B-Instruct": ["undefended", "nx", "matched", "tuned"],
        "gemma-2-2b-it": ["undefended", "nx", "matched"]}


def verdicts(model: str, tag: str, split: str = "test") -> list[dict]:
    with gzip.open(DATA / "verdicts" / model / split / f"{tag}.jsonl.gz", "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _asr_rows(vs: list[dict], unknown_success: bool = False) -> list[dict]:
    rows = []
    for v in vs:
        if v["kind"] != "asr":
            continue
        task, _, _, _, position = v["uid"].split("/")
        ok = v["outcome"]
        if unknown_success and v.get("verdict") == "unknown":
            ok = True
        rows.append({"task": task, "position": position, "success": ok, "rule": v["via"] == "rule"})
    return rows


def by_scorer(vs: list[dict]) -> dict:
    out = {}
    for via in ("judge", "rule"):
        s = [v["outcome"] for v in vs if v["kind"] == "asr" and v["via"] == via and v["outcome"] is not None]
        out[via] = {"k": sum(s), "n": len(s), "asr": sum(s) / len(s)}
    return out


def analyze(model: str, tag: str) -> dict:
    vs = verdicts(model, tag)
    res = {"by_scorer_pooled": by_scorer(vs),
           "verdicts": {k: dict(Counter(v["verdict"] for v in vs if v["kind"] == k and v["via"] == "judge"))
                        for k in ("asr", "utility")},
           "asr_mean_unknown_as_success": summarize_asr(_asr_rows(vs, unknown_success=True))["mean"]}
    if tag not in ("undefended", "matched"):
        util = {k: {v["uid"]: v["outcome"] for v in verdicts(model, t) if v["kind"] == "utility"}
                for k, t in (("a", "undefended"), ("b", tag))}
        res["utility_paired_pooled_vs_undefended"] = paired_diff(util["a"], util["b"])
    return res


def main() -> dict:
    out = {m: {t: analyze(m, t) for t in tags} for m, tags in TAGS.items()}
    p = DATA / "scores" / "analysis.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
