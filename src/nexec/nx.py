"""The NX facade: load a model with its directions, render conversations with provenance, generate defended."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import torch

from nexec.directions import Directions, resolve, user_cache_dir
from nexec.hook import protect
from nexec.provenance import Encoded, encode

log = logging.getLogger("nexec")


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_gate(diagnostics: dict) -> str:
    """Spec §5.3: if most heldout content tokens already project positively on d, use the affine gate."""
    frac = sorted(diagnostics.get("heldout_frac_non_pos", [0.0]))
    return "affine" if frac[len(frac) // 2] > 0.5 else "raw"


class NX:
    def __init__(self, model, tok, directions: Directions, *, strict: bool = True):
        directions.check_compatible(model, tok, strict=strict)
        self.model, self.tok, self.directions = model, tok, directions

    @property
    def pad_token_id(self) -> int:
        return self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    @classmethod
    def from_pretrained(cls, model_id: str, *, directions: Directions | str | Path | None = None, model=None,
                        tokenizer=None, device: str | None = None, dtype: torch.dtype = torch.bfloat16,
                        fit_if_missing: bool = False, strict: bool = True) -> NX:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
        if model is None:
            from transformers import AutoModelForCausalLM

            attn = "eager" if "gemma" in model_id.lower() else "sdpa"  # Gemma-2 soft-capping needs eager attention
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
            model = model.to(device or default_device()).eval()
        if not isinstance(directions, Directions):
            path = resolve(model_id, directions)
            if path is not None:
                directions = Directions.load(path)
            elif fit_if_missing:
                from nexec.fit import fit_directions

                log.warning("no directions for %s; fitting them now (one-time)", model_id)
                directions = fit_directions(model, tokenizer, model_id)
                directions.config.gate = choose_gate(directions.config.diagnostics)
                directions.save(user_cache_dir(model_id))
            else:
                raise FileNotFoundError(
                    f"no nexec directions for {model_id}. Pass fit_if_missing=True, or run "
                    f"`uv run python scripts/fit_directions.py --model {model_id}`.")
        return cls(model, tokenizer, directions, strict=strict)

    def encode(self, conversations) -> Encoded:
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]
        return encode(self.tok, conversations)

    @contextmanager
    def protect(self, untrusted_mask: torch.Tensor, *, layers: list[int] | None = None, alpha: float | None = None,
                gate: str | None = None, record: dict | None = None, replay=None):
        cfg = self.directions.config
        with protect(self.model, d=self.directions.d, layers=cfg.layers if layers is None else layers,
                     alpha=cfg.alpha if alpha is None else alpha, untrusted_mask=untrusted_mask,
                     bias=self.directions.bias(gate), record=record, replay=replay) as state:
            yield state

    def _generate(self, enc: Encoded, **kwargs) -> list[str]:
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("max_new_tokens", 256)
        kwargs.setdefault("pad_token_id", self.pad_token_id)
        dev = self.model.device
        out = self.model.generate(enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev), **kwargs)
        n = enc.input_ids.shape[1]
        return [self.tok.decode(row[n:], skip_special_tokens=True) for row in out]

    @torch.no_grad()
    def generate(self, conversations, *, defend: bool = True, layers: list[int] | None = None,
                 alpha: float | None = None, gate: str | None = None, **generate_kwargs) -> list[str]:
        """Generate for one conversation (list of messages) or a batch of them; returns one string per row."""
        enc = self.encode(conversations)
        if not defend:
            return self._generate(enc, **generate_kwargs)
        with self.protect(enc.untrusted_mask, layers=layers, alpha=alpha, gate=gate):
            return self._generate(enc, **generate_kwargs)
