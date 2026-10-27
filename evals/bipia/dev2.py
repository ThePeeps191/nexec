"""dev2: a selection set for BIPIA's rule-scored test attack families (emoji, language, code), disjoint from the test
subset on both axes: the 4 instruction variants per type that the test subset did not sample, inserted into BIPIA
*train* contexts. Rule-scored, so it is free to evaluate. See docs/findings/2026-10-step3.5-residual.md.
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


def success(s: Sample, output: str) -> bool:
    return evaluator(s.attack, split="test")(output)


def family(s: Sample) -> str:
    return "code" if s.task == "code" else s.attack.rsplit("-", 1)[0]
