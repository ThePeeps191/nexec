"""End-to-end: the salary-table demo script on the real Llama-3.2-3B (needs cached weights)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "salary_table.py"
sys.path.insert(0, str(SCRIPT.parents[1]))
from evals.cases import is_compromised, is_correct  # noqa: E402


def test_demo_llama(tmp_path):
    spec = importlib.util.spec_from_file_location("salary_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main(["--out", str(tmp_path)])
    assert is_compromised(out["undefended"])
    assert is_correct(out["defended"]) and is_correct(out["clean_defended"])
    assert Path(out["html"]).exists() and Path(out["png"]).exists()
