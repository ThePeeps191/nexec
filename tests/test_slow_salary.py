"""End-to-end: NVIDIA's salary-table demo through the public API with the bundled directions (needs cached weights)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.cases import is_compromised, is_correct, salary_conversation  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nx():
    from nexec import NX

    return NX.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


def test_salary_flip_llama(nx):
    conv = salary_conversation("md_before")
    assert is_compromised(nx.generate(conv, defend=False, max_new_tokens=160)[0])
    defended = nx.generate(conv, max_new_tokens=160)[0]
    assert is_correct(defended), defended
    clean = nx.generate(salary_conversation("md_before", None), max_new_tokens=160)[0]
    assert is_correct(clean), clean


def test_batched_salary_matches_flags(nx):
    convs = [salary_conversation("md_before"), salary_conversation("md_before", None)]
    defended, clean = nx.generate(convs, max_new_tokens=160)
    assert is_correct(defended) and is_correct(clean)
