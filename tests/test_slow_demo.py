"""End-to-end: the salary-table demo script on the real Llama-3.2-3B (needs cached weights)."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "salary_table.py"


def test_demo_llama(tmp_path):
    spec = importlib.util.spec_from_file_location("salary_table", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main(["--out", str(tmp_path)])
    assert "compromised" in out["undefended"].lower()
    assert "85" in out["defended"] and "compromised" not in out["defended"].lower()
    assert Path(out["html"]).exists() and Path(out["png"]).exists()
