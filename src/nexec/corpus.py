"""Labeled instruction/content corpus used to fit instruction directions.

Each example is a short conversation. User turns carry character spans labeled by ROLE, not mood:
``instruction`` (the user's actual request), ``non_instruction`` (the content it operates on, even when that content is
phrased as a command), or ``boundary`` (connectors and framing such as ":" or "Here is my essay:").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

CORPUS_VERSION = "v1"
LABELS = ("instruction", "non_instruction", "boundary")


@dataclass
class Span:
    start: int
    end: int
    label: str


@dataclass
class Turn:
    role: str
    text: str
    spans: list[Span] = field(default_factory=list)


@dataclass
class Example:
    id: str
    category: str
    split: str
    imperative_content: bool
    turns: list[Turn]


def default_path() -> Path:
    return Path(str(resources.files("nexec") / "data" / f"corpus_{CORPUS_VERSION}.jsonl"))


def load_corpus(path=None) -> list[Example]:
    out = []
    for line in Path(path or default_path()).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        turns = [Turn(t["role"], t["text"], [Span(**s) for s in t["spans"]]) for t in row["turns"]]
        out.append(Example(row["id"], row["category"], row["split"], row["imperative_content"], turns))
    return out


def corpus_sha256(path=None) -> str:
    return hashlib.sha256(Path(path or default_path()).read_bytes()).hexdigest()


def validate(ex: Example) -> list[str]:
    errors = []
    if not ex.turns or ex.turns[0].role != "user":
        errors.append("first turn must be a user turn")
    for i, t in enumerate(ex.turns):
        if t.role not in ("user", "assistant"):
            errors.append(f"turn {i}: bad role {t.role!r}")
        if i and t.role == ex.turns[i - 1].role:
            errors.append(f"turn {i}: roles must alternate")
        if t.role == "assistant":
            if t.spans:
                errors.append(f"turn {i}: assistant turns carry no spans")
            continue
        prev, covered = 0, [False] * len(t.text)
        for s in t.spans:
            if s.label not in LABELS:
                errors.append(f"turn {i}: bad label {s.label!r}")
            if not 0 <= s.start < s.end <= len(t.text):
                errors.append(f"turn {i}: span out of range {s}")
                continue
            if s.start < prev:
                errors.append(f"turn {i}: spans unsorted or overlapping at {s.start}")
            prev = max(prev, s.end)
            covered[s.start:s.end] = [True] * (s.end - s.start)
        gap = next((j for j, ch in enumerate(t.text) if not ch.isspace() and not covered[j]), None)
        if gap is not None:
            errors.append(f"turn {i}: uncovered char {t.text[gap]!r} at {gap}")
    labels = {s.label for t in ex.turns for s in t.spans}
    for need in ("instruction", "non_instruction"):
        if need not in labels:
            errors.append(f"no {need} span")
    return errors


def build(rows) -> list[dict]:
    """Turn ``(id, category, split, imperative, turns)`` tuples written as labeled segments into JSON rows."""
    out = []
    for ex_id, category, split, imperative, turns in rows:
        jt = []
        for role, body in turns:
            if role == "assistant":
                jt.append({"role": role, "text": body, "spans": []})
                continue
            text, spans = "", []
            for label, seg in body:
                spans.append({"start": len(text), "end": len(text) + len(seg), "label": label})
                text += seg
            jt.append({"role": role, "text": text, "spans": spans})
        out.append({"id": ex_id, "category": category, "split": split, "imperative_content": imperative,
                    "turns": jt})
    return out
