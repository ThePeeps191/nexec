import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, vocab_size=256, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def stub_item():
    ids = list(range(5, 29))  # 24 tokens
    labels = [None] + ["instruction"] * 7 + ["boundary"] + ["non_instruction"] * 15
    return ids, labels
