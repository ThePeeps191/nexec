"""Per-model instruction directions as versioned artifacts (like tokenizer files).

On disk a directions directory holds ``directions.safetensors`` (fp32 ``d``, ``mu_instr``, ``mu_non``, each
``[n_layers, d_model]``) and ``nexec.json`` (how they were fit and how to apply them).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from nexec._version import __version__

BUNDLED_DIR = Path(__file__).parent / "directions"
TENSORS, CONFIG = "directions.safetensors", "nexec.json"


class DirectionsMismatchError(ValueError):
    """Directions were fit for a different model, revision, or chat template than the one in use."""


@dataclass
class DirectionsConfig:
    model_id: str
    model_revision: str | None = None
    chat_template_sha256: str | None = None
    hook_point: str = "resid_post"
    layers: list[int] = field(default_factory=list)
    alpha: float = 1.0
    gate: Literal["raw", "affine", "mid"] = "raw"
    fit_context: str = "user"
    corpus_version: str = ""
    corpus_sha256: str = ""
    nexec_version: str = __version__
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    diagnostics: dict = field(default_factory=dict)
    format_version: int = 1


@dataclass
class Directions:
    d: torch.Tensor          # [L, D] unit rows
    mu_instr: torch.Tensor   # [L, D]
    mu_non: torch.Tensor     # [L, D]
    config: DirectionsConfig

    def bias(self, gate: str | None = None) -> torch.Tensor | None:
        """Per-layer ReLU threshold b for ``gate`` (default: the configured gate). None for the raw gate (b = 0)."""
        gate = gate or self.config.gate
        if gate == "raw":
            return None
        if gate == "affine":
            ref = self.mu_non
        elif gate == "mid":
            ref = (self.mu_instr + self.mu_non) / 2
        else:
            raise ValueError(f"unknown gate {gate!r}")
        return (ref * self.d).sum(-1)

    def save(self, path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({"d": self.d.float().contiguous(), "mu_instr": self.mu_instr.float().contiguous(),
                   "mu_non": self.mu_non.float().contiguous()}, str(path / TENSORS))
        (path / CONFIG).write_text(json.dumps(dataclasses.asdict(self.config), indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path) -> Directions:
        path = Path(path)
        t = load_file(str(path / TENSORS))
        raw = json.loads((path / CONFIG).read_text())
        if raw.get("format_version", 1) != 1:
            raise ValueError(f"unsupported directions format_version {raw['format_version']} in {path}")
        return cls(t["d"], t["mu_instr"], t["mu_non"], DirectionsConfig(**raw))

    def check_compatible(self, model, tok, strict: bool = True) -> None:
        problems = []
        n_layers, d_model = self.d.shape
        cfg = getattr(model.config, "text_config", None) or model.config
        if (cfg.num_hidden_layers, cfg.hidden_size) != (n_layers, d_model):
            problems.append(f"model has {cfg.num_hidden_layers} layers x {cfg.hidden_size} dims; "
                            f"directions have {n_layers} layers x {d_model} dims")
        rev = getattr(model.config, "_commit_hash", None)
        if rev and self.config.model_revision and rev != self.config.model_revision:
            problems.append(f"model revision {rev} != directions revision {self.config.model_revision}")
        sha = template_sha256(tok)
        if sha and self.config.chat_template_sha256 and sha != self.config.chat_template_sha256:
            problems.append("chat template differs from the one the directions were fit with")
        if not problems:
            return
        msg = f"directions for {self.config.model_id} do not match: " + "; ".join(problems)
        if strict:
            raise DirectionsMismatchError(msg + " (pass strict=False to apply anyway)")
        warnings.warn(msg, UserWarning, stacklevel=2)


def template_sha256(tok) -> str | None:
    tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, dict):
        tpl = json.dumps(tpl, sort_keys=True)
    return hashlib.sha256(tpl.encode()).hexdigest() if tpl else None


def slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def user_cache_dir(model_id: str) -> Path:
    root = Path(os.environ.get("NEXEC_CACHE", Path.home() / ".cache" / "nexec"))
    return root / slug(model_id)


def _is_artifact(p: Path) -> bool:
    return (p / TENSORS).exists() and (p / CONFIG).exists()


def resolve(model_id: str, source: str | os.PathLike | None = None) -> Path | None:
    """Find a directions directory: explicit dir or ``hf://repo[/subdir]``, else user cache, else bundled."""
    if source is not None:
        s = str(source)
        if s.startswith("hf://"):
            parts = s[len("hf://"):].split("/")
            repo_id, subdir = "/".join(parts[:2]), "/".join(parts[2:])
            p = Path(snapshot_download(repo_id=repo_id))
            p = p / subdir if subdir else p
        else:
            p = Path(s)
        if not _is_artifact(p):
            raise FileNotFoundError(f"no {TENSORS} + {CONFIG} in {p}")
        return p
    for p in (user_cache_dir(model_id), BUNDLED_DIR / slug(model_id)):
        if _is_artifact(p):
            return p
    return None
