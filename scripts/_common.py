"""Shared helpers for repo scripts (puts the repo root on sys.path so `evals` imports)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA = "google/gemma-2-2b-it"
SHIPPED = {LLAMA: dict(gate="raw"), GEMMA: dict(gate="affine")}
