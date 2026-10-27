import collections

import pytest

pytest.importorskip("evals.bipia.pipeline")

from evals.bipia import dev2  # noqa: E402
from evals.bipia.asr import Rule, evaluator  # noqa: E402
from evals.bipia.data import attack_text, context_index  # noqa: E402
from evals.bipia.pipeline import samples  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return dev2.build()


def test_dev2_is_disjoint_from_test_on_instructions_and_contexts(built):
    test = [s for s in samples("test")[0] if isinstance(evaluator(s.attack, split="test"), Rule)]
    test_attacks = {(s.task, s.attack) for s in test}
    assert not {(s.task, s.attack) for s in built} & test_attacks
    assert all(s.uid.split("/")[1] == "dev2" for s in built)  # contexts come from BIPIA's train rows
    test_ctx = {s.context for s in samples("test")[0]} | {s.context for s in samples("test")[1]}
    assert not any(s.context in test_ctx for s in built)


def test_dev2_covers_rule_families_with_held_out_variants(built):
    fam = collections.Counter(dev2.family(s) for s in built)
    assert set(fam) == {"Emoji Substitution", "Language Translation", "code"}
    assert len(built) == 384 and len({s.uid for s in built}) == 384
    s = built[0]
    assert attack_text(s.attack, s.task, "test") in s.context and context_index(s) >= 0
