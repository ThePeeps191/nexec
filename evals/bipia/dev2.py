"""dev2: a selection set for BIPIA's rule-scored test attack families (emoji, language, code): the 4 instruction
variants per type that the test subset did not sample, inserted into BIPIA *train* contexts. Rule-scored, so it is
free to evaluate. See docs/findings/2026-10-step3.5-residual.md.

Instruction-disjoint from the test subset. Context-disjoint except where BIPIA's own train and test files duplicate a
row: one email context, 24 of the 384 items (`overlaps_test`).
"""

from __future__ import annotations

from evals.bipia.asr import Rule, evaluator
from evals.bipia.data import (
    POSITIONS,
    TASKS,
    Sample,
    _load_rows,
    _normal,
    _pick_attacks,
    _pick_contexts,
    attack_text,
    attack_types,
    insert,
)

N_CONTEXTS, SEED = 2, 2
TEXT_FAMILIES = ("Emoji Substitution", "Language Translation")
TEST_SEED = 0  # the seed pipeline.samples("test") uses to pick 1 instruction per type


def held_out_attacks(task: str) -> list[str]:
    """Rule-scored test attack names whose instruction the test subset did not sample."""
    used = set(_pick_attacks(task, "test", 1, TEST_SEED))
    return [f"{t}-{i}" for t, instrs in attack_types(task, "test").items() for i in range(len(instrs))
            if f"{t}-{i}" not in used and isinstance(evaluator(f"{t}-{i}", split="test"), Rule)
            and (task == "code" or t in TEXT_FAMILIES)]


def build() -> list[Sample]:
    out = []
    for task in TASKS:
        rows = _load_rows(task, "train")
        attacks = held_out_attacks(task)
        for pos in POSITIONS:
            for i in _pick_contexts(task, "train", N_CONTEXTS, SEED):
                ctx, fields, ideal = _normal(task, rows[i])
                for a in attacks:
                    out.append(Sample(f"{task}/dev2/{i:04d}/{a}/{pos}", task, "dev2", a, pos,
                                      insert(ctx, attack_text(a, task, "test"), pos), dict(fields), ideal))
    return out


def _test_contexts(task: str) -> set[str]:
    from evals.bipia.pipeline import samples

    rows = _load_rows(task, "test")
    used = {int(x.uid.split("/")[2]) for part in samples("test") for x in part if x.task == task}
    return {_normal(task, rows[i])[0] for i in used}


def overlaps_test(s: Sample) -> bool:
    """Is this item's raw (pre-insertion) context also a context in the test subset?"""
    if s.task not in _TEST_CTX:
        _TEST_CTX[s.task] = _test_contexts(s.task)
    return _normal(s.task, _load_rows(s.task, "train")[int(s.uid.split("/")[2])])[0] in _TEST_CTX[s.task]


_TEST_CTX: dict[str, set[str]] = {}


def success(s: Sample, output: str) -> bool:
    return evaluator(s.attack, split="test")(output)


def family(s: Sample) -> str:
    return "code" if s.task == "code" else s.attack.rsplit("-", 1)[0]
