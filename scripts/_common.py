"""Shared helpers for repo scripts (puts the repo root on sys.path so `evals` imports)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA = "google/gemma-2-2b-it"
# Gemma alpha 2: step-3.5 retune (docs/findings/2026-10-step3.5-residual.md)
SHIPPED = {LLAMA: dict(gate="raw", alpha=1.0), GEMMA: dict(gate="affine", alpha=2.0)}
