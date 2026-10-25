from pathlib import Path

from corpus import Example, Span, load_corpus, validate

PILOT = Path(__file__).resolve().parents[1] / "corpus_pilot.jsonl"


def _first_instruction_after_content(ex: Example) -> bool:
    for s in ex.spans:
        if s.label == "instruction":
            return False
        if s.label == "non_instruction":
            return True
    return False


def test_pilot_is_valid():
    for ex in load_corpus(PILOT):
        assert validate(ex) == [], ex.id


def test_pilot_composition():
    examples = load_corpus(PILOT)
    assert len(examples) >= 40
    assert len({ex.category for ex in examples}) >= 12
    assert sum(_first_instruction_after_content(ex) for ex in examples) >= 10
    assert sum(ex.imperative_content for ex in examples) >= 6
    heldout = [ex for ex in examples if ex.split == "heldout"]
    assert len(heldout) == 8
    assert len({ex.category for ex in heldout}) >= 6


def test_validate_catches_overlap():
    ex = Example("x", "c", "Translate this: hello", [Span(0, 9, "instruction"), Span(5, 21, "non_instruction")], "train")
    assert validate(ex)


def test_validate_catches_uncovered_text():
    ex = Example("x", "c", "Translate this: hello", [Span(0, 14, "instruction"), Span(16, 21, "non_instruction")], "train")
    assert validate(ex)  # ':' at 14 is uncovered


def test_validate_catches_bad_label():
    ex = Example("x", "c", "hi", [Span(0, 2, "order")], "train")
    assert validate(ex)
