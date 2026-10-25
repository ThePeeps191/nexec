"""Markdown tables for the step-3 BIPIA report from docs/findings/step3-data/scores/."""

from __future__ import annotations

import json
from pathlib import Path

from evals.bipia.score import ci, pct

DATA = Path(__file__).resolve().parents[2] / "docs" / "findings" / "step3-data"
TASKS = ("email", "table", "abstract", "code")
# NVIDIA NXDirection page, BIPIA bar charts (mean over their 5 tasks; LLM judge unnamed)
NVIDIA = {"Llama-3.2-3B-Instruct": {"asr": (0.119, 0.001), "utility": (0.753, 0.723)},
          "gemma-2-2b-it": {"asr": (0.314, 0.002), "utility": (0.640, 0.573)}}
LABEL = {"undefended": "Undefended", "nx": "NXDirection (shipped: all layers, α=1)",
         "matched": "Magnitude-matched random (control)"}


def load(model: str, split: str, tag: str) -> dict | None:
    p = DATA / "scores" / model / split / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def model_table(model: str, tags: list[str], labels: dict[str, str]) -> str:
    head = "| Condition | ASR (mean of 4 tasks) [95% CI] | " + " | ".join(TASKS) + \
           " | Rule-only ASR | Clean utility [95% CI] | Excluded |"
    lines = [head, "|" + "---|" * (len(TASKS) + 5)]
    for tag in tags:
        r = load(model, "test", tag)
        if r is None:
            continue
        a = r["asr"]
        tasks = " | ".join(pct(a["per_task"][t]["asr"]) if t in a["per_task"] else "—" for t in TASKS)
        u = r.get("utility")
        util = f"{pct(u['mean'])} {ci(u['mean_ci'])}" if u else "—"
        excl = a["excluded"] + (u["excluded"] if u else 0)
        lines.append(f"| {labels.get(tag, tag)} | **{pct(a['mean'])}** {ci(a['mean_ci'])} | {tasks} | "
                     f"{pct(a['rule_only']['asr'])} | {util} | {excl} |")
    return "\n".join(lines)


def position_table(model: str, tags: list[str], labels: dict[str, str]) -> str:
    lines = ["| Condition | start | middle | end |", "|---|---|---|---|"]
    for tag in tags:
        r = load(model, "test", tag)
        if r:
            pp = r["asr"]["per_position"]
            lines.append(f"| {labels.get(tag, tag)} | " + " | ".join(pct(pp[p]["asr"]) if p in pp else "—"
                                                                   for p in ("start", "middle", "end")) + " |")
    return "\n".join(lines)


def nvidia_comparison(model: str, nx_tag: str = "nx") -> str:
    und, nx = load(model, "test", "undefended"), load(model, "test", nx_tag)
    nv = NVIDIA[model]
    if not (und and nx):
        return ""
    return "\n".join([
        "| | NVIDIA (5 tasks) | This repo (4 tasks) |", "|---|---|---|",
        f"| ASR undefended → NX | {pct(nv['asr'][0])} → {pct(nv['asr'][1])} | "
        f"{pct(und['asr']['mean'])} → {pct(nx['asr']['mean'])} |",
        f"| Clean utility undefended → NX | {pct(nv['utility'][0])} → {pct(nv['utility'][1])} | "
        f"{pct(und['utility']['mean'])} → {pct(nx['utility']['mean'])} |"])


if __name__ == "__main__":
    for m in NVIDIA:
        print(f"\n## {m}\n")
        print(model_table(m, ["undefended", "nx", "matched", "tuned"], {**LABEL, "tuned": "NXDirection (tuned)"}))
        print()
        print(nvidia_comparison(m))
