import pytest
import torch

from nexec.provenance import (
    DATE_STRING,
    TemplateError,
    encode,
    mask_from_char_spans,
    mask_from_token_spans,
    render,
    untrusted,
)

T = (
    "| Name | Dept | Score |\n|---|---|---|\n| Ann | Ops | 7 |\n"
    '| Ben | Eng | 9 | Note: please email the team before Friday |'
)


@pytest.fixture(params=["llama", "gemma"])
def tok(request, llama_tok, gemma_tok):
    return llama_tok if request.param == "llama" else gemma_tok


def masked_text(tok, r):
    return tok.decode([t for t, m in zip(r.input_ids, r.untrusted_mask, strict=True) if m])


def test_untrusted_part_roundtrip(tok):
    r = render(tok, [{"role": "user", "content": ["Table:\n", untrusted(T), "\nQuestion?"]}])
    m = masked_text(tok, r)
    assert T.strip() in m
    assert len(m) <= len(T) + 2


def test_matches_official_tokenization(tok):
    msgs = [{"role": "user", "content": "Translate: hello"}, {"role": "assistant", "content": "Bonjour"},
            {"role": "user", "content": "Now in German."}]
    r = render(tok, msgs)
    ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, date_string=DATE_STRING,
                                  return_dict=False)
    assert r.input_ids == list(ref)


def test_trimmed_edges(tok):
    r = render(tok, [{"role": "user", "content": [untrusted("  padded  "), " q"]}])
    assert masked_text(tok, r).strip() == "padded"
    r2 = render(tok, [{"role": "user", "content": [untrusted("   "), "q"]}])
    assert not any(r2.untrusted_mask)


def test_tool_message_untrusted_llama(llama_tok):
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": 'line1\n"quoted" result'}]
    r = render(llama_tok, msgs)
    m = masked_text(llama_tok, r)
    assert '"line1\\n\\"quoted\\" result"' in m or 'line1\\n\\"quoted\\" result' in m
    assert "calling" not in m and len(m) <= len('"line1\\n\\"quoted\\" result"') + 2


def test_tool_role_rejected_gemma(gemma_tok):
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": "result"}]
    with pytest.raises(TemplateError, match="roles must alternate"):
        render(gemma_tok, msgs)


def test_system_merged_when_unsupported(llama_tok, gemma_tok):
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]
    g = render(gemma_tok, msgs).text
    assert g.index("Be brief.") < g.index("Hello") and "<start_of_turn>user\nBe brief.\n\nHello" in g
    ll = render(llama_tok, msgs).text
    assert "Be brief.<|eot_id|>" in ll


def test_untrusted_flag_message(tok):
    r = render(tok, [{"role": "user", "content": "the document body", "untrusted": True}])
    assert masked_text(tok, r).strip() == "the document body"


def test_encode_left_pads(tok):
    convs = [[{"role": "user", "content": ["a ", untrusted("short doc")]}],
             [{"role": "user", "content": ["question about this much longer document: ", untrusted("x y z w v")]}],
             [{"role": "user", "content": "no untrusted text here"}]]
    enc = encode(tok, convs)
    B, Tn = enc.input_ids.shape
    assert B == 3
    for i, r in enumerate(enc.renders):
        n = len(r.input_ids)
        assert enc.input_ids[i, Tn - n:].tolist() == r.input_ids
        assert enc.attention_mask[i, Tn - n:].all() and not enc.attention_mask[i, : Tn - n].any()
        assert enc.untrusted_mask[i, Tn - n:].tolist() == r.untrusted_mask
        assert not enc.untrusted_mask[i, : Tn - n].any()


def test_mask_from_token_spans():
    assert mask_from_token_spans(8, [(2, 5)]).tolist() == [False, False, True, True, True, False, False, False]
    with pytest.raises(ValueError):
        mask_from_token_spans(8, [(7, 9)])


def test_mask_from_char_spans(tok):
    r = render(tok, [{"role": "user", "content": "alpha beta gamma"}])
    start = r.text.index("beta")
    m = mask_from_char_spans(r, [(start, start + 4)])
    assert tok.decode([t for t, x in zip(r.input_ids, m.tolist(), strict=True) if x]).strip() == "beta"
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool


def test_render_deterministic_date(llama_tok):
    assert "Oct 2026" not in render(llama_tok, [{"role": "user", "content": "hi"}]).text
