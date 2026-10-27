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


def _gated_tokenizer(model_id: str):
    """The real tokenizer (gated on the HF Hub); without access, e.g. CI without an HF_TOKEN, the test is skipped."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except OSError as e:
        pytest.skip(f"needs the gated {model_id} tokenizer (accept its license, set HF_TOKEN): {type(e).__name__}")


@pytest.fixture(scope="session")
def llama_tok():
    return _gated_tokenizer(LLAMA)


@pytest.fixture(scope="session")
def gemma_tok():
    return _gated_tokenizer(GEMMA)


@pytest.fixture(scope="session")
def tiny_llama_fullvocab(llama_tok):
    return _tiny_llama(len(llama_tok))


@pytest.fixture(scope="session")
def tiny_nx(tiny_llama_fullvocab, llama_tok):
    """An NX over a tiny random Llama with the real Llama tokenizer and directions fit on two stub examples."""
    from nexec import NX
    from nexec.corpus import Example, Span, Turn
    from nexec.fit import fit_directions

    examples = [
        Example("a1", "translation", "train", False, [Turn("user", "Translate to French: the cat sleeps.",
                [Span(0, 19, "instruction"), Span(19, 20, "boundary"), Span(21, 36, "non_instruction")])]),
        Example("a2", "translation", "heldout", False, [Turn("user", "Translate to German: good morning.",
                [Span(0, 19, "instruction"), Span(19, 20, "boundary"), Span(21, 34, "non_instruction")])]),
    ]
    dirs = fit_directions(tiny_llama_fullvocab, llama_tok, "org/tiny", examples=examples)
    return NX.from_pretrained("org/tiny", model=tiny_llama_fullvocab, tokenizer=llama_tok, directions=dirs)
