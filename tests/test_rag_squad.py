import pytest

pytest.importorskip("evals.bipia.judge")

from evals.rag import squad  # noqa: E402
from nexec.provenance import Untrusted  # noqa: E402

ROWS = [{"id": f"q{i}", "context": f"Passage {i // 3}.", "question": f"Question {i}?",
         "answers": {"text": [f"answer {i}", f"The Answer {i}!"], "answer_start": [0, 0]}} for i in range(30)]


def test_normalize_follows_squad():
    assert squad.normalize("  The  Denver, Broncos! ") == "denver broncos"
    assert squad.normalize("an apple a day") == "apple day"


def test_gold_match_is_containment_after_normalization():
    assert squad.gold_match("It was the Denver Broncos.", ["Denver Broncos"])
    assert not squad.gold_match("The Panthers.", ["Denver Broncos"])
    assert not squad.gold_match("", ["Denver Broncos"])


def test_sample_one_question_per_distinct_passage_deterministic():
    a = squad.sample(ROWS, 5, seed=0)
    assert a == squad.sample(ROWS, 5, seed=0)
    assert len(a) == 5 and len({r["context"] for r in a}) == 5
    assert squad.sample(ROWS, 50, seed=0) and len(squad.sample(ROWS, 50, seed=0)) == 10  # only 10 passages


def test_item_marks_only_the_passage_untrusted():
    it = squad.item(ROWS[0])
    assert it["uid"] == "squad/q0"
    (msg,) = it["conversation"]
    assert msg["role"] == "user"
    parts = msg["content"]
    unt = [p for p in parts if isinstance(p, Untrusted)]
    assert [p.text for p in unt] == ["Passage 0."]
    tail = parts[-1]
    assert isinstance(tail, str) and "Question 0?" in tail and tail.endswith("Answer briefly.")


def test_judge_prompt_lists_every_reference_and_ends_with_instruction():
    p = squad.judge_prompt(ROWS[1], "answer 1")
    assert "Question 1?" in p and "answer 1 | The Answer 1!" in p and "Assistant's answer: answer 1" in p
    assert p.endswith("Reply YES or NO.")


def test_references_deduplicated_in_order():
    row = {**ROWS[0], "answers": {"text": ["x", "y", "x"], "answer_start": [0, 0, 0]}}
    assert squad.references(row) == ["x", "y"]
