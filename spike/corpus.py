"""Character-span-labeled instruction/content corpus."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Label = Literal["instruction", "non_instruction", "boundary"]
LABELS = ("instruction", "non_instruction", "boundary")


@dataclass
class Span:
    start: int
    end: int
    label: str


@dataclass
class Example:
    id: str
    category: str
    text: str
    spans: list[Span]
    split: str = "train"
    imperative_content: bool = field(default=False)


def load_corpus(path) -> list[Example]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["spans"] = [Span(**s) for s in row["spans"]]
        out.append(Example(**row))
    return out


def validate(ex: Example) -> list[str]:
    errors = []
    prev_end = 0
    covered = [False] * len(ex.text)
    for s in ex.spans:
        if s.label not in LABELS:
            errors.append(f"bad label {s.label!r}")
        if not (0 <= s.start < s.end <= len(ex.text)):
            errors.append(f"span out of range {s}")
            continue
        if s.start < prev_end:
            errors.append(f"span unsorted or overlapping at {s.start}")
        prev_end = max(prev_end, s.end)
        for i in range(s.start, s.end):
            covered[i] = True
    for i, ch in enumerate(ex.text):
        if not ch.isspace() and not covered[i]:
            errors.append(f"uncovered char {ch!r} at {i}")
            break
    return errors
