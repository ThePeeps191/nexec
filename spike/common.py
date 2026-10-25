"""Shared model loading for the step-0 spike."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA = "google/gemma-2-2b-it"
MODELS = [LLAMA, GEMMA]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def load(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    # Gemma-2 needs eager attention for its logit soft-capping; bf16 avoids its fp16 overflow.
    attn = "eager" if "gemma" in model_id.lower() else "sdpa"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation=attn)
    return model.to(DEVICE).eval(), tok
