import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA = "google/gemma-2-2b-it"


def _tiny_llama(vocab_size: int):
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, vocab_size=vocab_size, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def tiny_llama():
    return _tiny_llama(256)


@pytest.fixture(scope="session")
def tiny_gemma2():
    torch.manual_seed(0)
    cfg = Gemma2Config(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=16, intermediate_size=128, vocab_size=256, max_position_embeddings=256,
                       sliding_window=8, query_pre_attn_scalar=16, attn_implementation="eager")
    return Gemma2ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def llama_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LLAMA)


@pytest.fixture(scope="session")
def gemma_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GEMMA)


@pytest.fixture(scope="session")
def tiny_llama_fullvocab(llama_tok):
    return _tiny_llama(len(llama_tok))
