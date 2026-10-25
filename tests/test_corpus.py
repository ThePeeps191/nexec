import json
from pathlib import Path

from nexec.corpus import Example, Span, Turn, build, load_corpus, validate

CATEGORIES = {"translation", "bug_fix", "classification", "email", "summarization", "extraction", "rewriting",
              "passage_qa", "format_conversion", "sentiment", "arithmetic", "code_explanation", "title_generation",
              "grammar", "keywords", "notes_to_list", "comparison", "action_items"}
EVAL = json.loads((Path(__file__).parent / "data" / "eval_strings.json").read_text())


def _instruction_after_content(ex: Example) -> bool:
    for t in ex.turns:
        for s in t.spans:
            if s.label == "instruction":
                return False
            if s.label == "non_instruction":
                return True
    return False


def test_corpus_valid():
    for ex in load_corpus():
        assert validate(ex) == [], ex.id


def test_corpus_composition():
    ex = load_corpus()
    assert 170 <= len(ex) <= 190
    cats = {c: sum(e.category == c for e in ex) for c in CATEGORIES}
    assert set(e.category for e in ex) == CATEGORIES and min(cats.values()) >= 6, cats
    assert sum(_instruction_after_content(e) for e in ex) >= 50
    assert sum(e.imperative_content for e in ex) >= 40
    assert sum(sum(t.role == "user" for t in e.turns) >= 2 for e in ex) >= 15
    held = [e for e in ex if e.split == "heldout"]
    assert abs(len(held) / len(ex) - 0.20) <= 0.02
    assert {e.category for e in held} == CATEGORIES


def test_ids_unique():
    ids = [e.id for e in load_corpus()]
    assert len(ids) == len(set(ids))


def test_no_eval_leakage():
    for ex in load_corpus():
        text = " ".join(t.text for t in ex.turns)
        for w in EVAL["words"]:
            assert w.lower() not in text.lower(), (ex.id, w)
        for inj in EVAL["injections"]:
            assert inj.lower()[:30] not in text.lower(), (ex.id, inj)


def test_build_computes_spans():
    rows = build([("x1", "translation", "train", False,
                   [("user", [("instruction", "Translate"), ("boundary", ": "), ("non_instruction", "hola")]),
                    ("assistant", "ok"), ("user", [("instruction", "Again.")])])])
    t = rows[0]["turns"]
    assert t[0]["text"] == "Translate: hola"
    assert t[0]["spans"][2] == {"start": 11, "end": 15, "label": "non_instruction"}
    assert t[1] == {"role": "assistant", "text": "ok", "spans": []}


def _ex(turns):
    return Example("x", "translation", "train", False, turns)


def test_validate_catches_errors():
    good = Turn("user", "Fix: a", [Span(0, 3, "instruction"), Span(3, 4, "boundary"), Span(5, 6, "non_instruction")])
    assert validate(_ex([good])) == []
    assert validate(_ex([Turn("user", "Fix: a", [Span(0, 4, "instruction"), Span(3, 6, "non_instruction")])]))
    assert validate(_ex([Turn("user", "Fix: a", [Span(0, 3, "instruction"), Span(5, 6, "non_instruction")])]))
    assert validate(_ex([Turn("user", "Fix: a", [Span(0, 3, "order"), Span(3, 6, "non_instruction")])]))
    assert validate(_ex([good, Turn("assistant", "ok", [Span(0, 2, "instruction")])]))
    assert validate(_ex([Turn("user", "abc", [Span(0, 3, "non_instruction")])]))  # no instruction span
    assert validate(_ex([Turn("assistant", "hi", []), good]))  # must start with user
