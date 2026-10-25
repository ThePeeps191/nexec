"""Resumable, batched generation for BIPIA samples under one condition (undefended | nx | matched).

Records are appended to a gzipped JSONL file after every batch; already-present uids are skipped, so a killed run
continues where it stopped. Prompts are sorted by length to minimize padding.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from nexec.controls import matched_random

CONDITIONS = ("undefended", "nx", "matched")


def read_records(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with gzip.open(path, "rt") as fh:  # reads concatenated gzip members (one per appended batch)
        return [json.loads(line) for line in fh if line.strip()]


def _append(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _length(conv) -> int:
    return sum(len(p if isinstance(p, str) else p.text) for m in conv
               for p in (m["content"] if isinstance(m["content"], list) else [m["content"]]))


def run(nx, condition: str, items: list[dict], out_path, *, layers=None, alpha=None, batch_size: int = 16,
        max_new_tokens: int | dict = 160, log=print) -> int:
    """items: [{"uid", "conversation", "meta"}]. ``max_new_tokens`` may be a dict keyed by meta["task"].
    Returns the number of new records written."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    done = {r["uid"] for r in read_records(out_path)}
    todo = sorted((it for it in items if it["uid"] not in done), key=lambda it: _length(it["conversation"]))
    cfg = nx.directions.config
    eff_layers = cfg.layers if layers is None else layers
    eff_alpha = cfg.alpha if alpha is None else alpha

    def mnt(it):
        return max_new_tokens.get(it["meta"].get("task"), 160) if isinstance(max_new_tokens, dict) else max_new_tokens

    groups: dict[int, list[dict]] = {}
    for it in todo:
        groups.setdefault(mnt(it), []).append(it)
    written = 0
    for n_new, group in groups.items():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            convs = [it["conversation"] for it in batch]
            gen = dict(max_new_tokens=n_new)
            if condition == "undefended":
                outs = nx.generate(convs, defend=False, **gen)
            elif condition == "nx":
                outs = nx.generate(convs, layers=layers, alpha=alpha, **gen)
            else:
                outs = matched_random(nx, convs, seed=0, layers=layers, alpha=alpha, **gen)
            _append(out_path, [{"uid": it["uid"], **it["meta"], "condition": condition, "model": cfg.model_id,
                                "layers": f"{eff_layers[0]}-{eff_layers[-1]}" if eff_layers else "",
                                "alpha": eff_alpha, "gate": cfg.gate, "output": o}
                               for it, o in zip(batch, outs, strict=True)])
            written += len(batch)
            log(f"{condition}: {len(done) + written}/{len(done) + len(todo)}")
    return written
