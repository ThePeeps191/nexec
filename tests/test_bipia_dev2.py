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


def test_dev2_is_disjoint_from_test_on_instructions(built):
    test = [s for s in samples("test")[0] if isinstance(evaluator(s.attack, split="test"), Rule)]
    test_attacks = {(s.task, s.attack) for s in test}
    assert not {(s.task, s.attack) for s in built} & test_attacks
    assert all(s.uid.split("/")[1] == "dev2" for s in built)  # contexts come from BIPIA's train rows


def test_dev2_raw_context_overlap_with_test_is_the_known_bipia_duplicate(built):
    # Raw contexts, before insertion. BIPIA's email train and test files share rows; one of them (train row 3) is
    # also in the test subset, so its 24 dev2 items (8 held-out attacks x 3 positions) share a context with test.
    overlap = [s for s in built if dev2.overlaps_test(s)]
    assert len(overlap) == 24
    assert {s.task for s in overlap} == {"email"}
    assert {s.uid.split("/")[2] for s in overlap} == {"0003"}


def test_dev2_covers_rule_families_with_held_out_variants(built):
    fam = collections.Counter(dev2.family(s) for s in built)
    assert set(fam) == {"Emoji Substitution", "Language Translation", "code"}
    assert len(built) == 384 and len({s.uid for s in built}) == 384
    s = built[0]
    assert attack_text(s.attack, s.task, "test") in s.context and context_index(s) >= 0
