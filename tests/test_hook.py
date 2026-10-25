import copy

import pytest
import torch

from nexec.hook import edit, protect

T, LO, HI = 24, 10, 16
LAYERS = [0, 1, 2]
ALWAYS = torch.full((3,), -1e3)  # bias so negative the ReLU fires on every untrusted row


def layer_outputs(model, ids, **kw):
    outs = {}
    hs = [layer.register_forward_hook(lambda m, a, o, l=l: outs.__setitem__(l, o.detach().clone()))
          for l, layer in enumerate(model.model.layers)]
    try:
        model(ids, **kw)
    finally:
        for h in hs:
            h.remove()
    return outs


def span_direction(model, ids, lo=LO, hi=HI):
    outs = layer_outputs(model, ids)
    v = torch.stack([outs[l][0, lo:hi].float().mean(0) for l in sorted(outs)])
    return v / v.norm(dim=-1, keepdim=True)


@pytest.fixture
def ids():
    return torch.arange(5, 5 + T).reshape(1, -1)


@pytest.fixture
def mask():
    m = torch.zeros(T, dtype=torch.bool)
    m[LO:HI] = True
    return m


@pytest.fixture
def d(tiny_llama, ids):
    return span_direction(tiny_llama, ids)


def test_relu_math():
    dv = torch.tensor([1.0, 0.0])
    assert torch.allclose(edit(torch.tensor([[3.0, 5.0]]), dv, 1.0, 0.0), torch.tensor([[0.0, 5.0]]))
    assert torch.equal(edit(torch.tensor([[-2.0, 5.0]]), dv, 1.0, 0.0), torch.tensor([[-2.0, 5.0]]))
    assert torch.allclose(edit(torch.tensor([[3.0, 5.0]]), dv, 2.0, 0.0), torch.tensor([[-3.0, 5.0]]))
    assert torch.allclose(edit(torch.tensor([[3.0, 5.0]]), dv, 1.0, 1.0), torch.tensor([[1.0, 5.0]]))


def test_masked_rows_untouched(tiny_llama, ids, mask, d):
    before, after = {}, {}
    pre = [lay.register_forward_hook(lambda m, a, o, l=l: before.__setitem__(l, o.clone()))
           for l, lay in enumerate(tiny_llama.model.layers)]
    try:
        with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask):
            post = [lay.register_forward_hook(lambda m, a, o, l=l: after.__setitem__(l, o.clone()))
                    for l, lay in enumerate(tiny_llama.model.layers)]
            tiny_llama(ids)
            for h in post:
                h.remove()
    finally:
        for h in pre:
            h.remove()
    for l in LAYERS:
        assert torch.equal(after[l][0, ~mask], before[l][0, ~mask])
        assert not torch.equal(after[l][0, mask], before[l][0, mask])


def test_all_trusted_identity(tiny_llama, ids, d):
    base = tiny_llama(ids).logits
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=torch.zeros(T, dtype=torch.bool)):
        assert torch.equal(base, tiny_llama(ids).logits)


def test_prefix_invariance(tiny_llama, ids, mask, d):
    base = tiny_llama(ids).logits
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask):
        out = tiny_llama(ids).logits
    assert torch.equal(base[0, :LO], out[0, :LO])
    assert not torch.allclose(base[0, HI:], out[0, HI:])


def test_decode_no_edit(tiny_llama, ids, mask, d):
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask) as st:
        tiny_llama.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False)
    assert st.edits == int(mask.sum()) * len(LAYERS)


def test_chunked_prefill_matches(tiny_llama, ids, mask, d):
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask):
        full = tiny_llama(ids).logits
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask) as st:
        first = tiny_llama(ids[:, :12], use_cache=True)
        second = tiny_llama(ids[:, 12:], past_key_values=first.past_key_values, use_cache=True)
    assert st.edits == int(mask.sum()) * len(LAYERS)
    assert torch.allclose(full[0, 12:], second.logits[0], atol=1e-5)


def test_generate_reuse_resets(tiny_llama, ids, mask, d):
    kw = dict(attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False)
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask) as st:
        a = tiny_llama.generate(ids, **kw)
        first = st.edits
        b = tiny_llama.generate(ids, **kw)
    assert first == int(mask.sum()) * len(LAYERS) and st.edits == 2 * first
    assert torch.equal(a, b)


def test_prefilled_cache_at_entry(tiny_llama, ids, mask, d):
    first = tiny_llama(ids[:, :12], use_cache=True)
    rec = {}
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask, bias=ALWAYS, record=rec):
        tiny_llama(ids[:, 12:], past_key_values=first.past_key_values, use_cache=True)
    for l in LAYERS:
        assert rec[l][0].nonzero().squeeze(1).tolist() == [12, 13, 14, 15]


def test_batch_left_padded_equals_unbatched(tiny_llama):
    lengths, spans = [20, 24, 16], [(4, 9), (10, 16), (2, 6)]
    Tm = max(lengths)
    prompts = [torch.arange(7 + i, 7 + i + n) for i, n in enumerate(lengths)]
    ids = torch.zeros(3, Tm, dtype=torch.long)
    att = torch.zeros(3, Tm, dtype=torch.long)
    msk = torch.zeros(3, Tm, dtype=torch.bool)
    for i, (p, (a, b)) in enumerate(zip(prompts, spans, strict=True)):
        ids[i, Tm - len(p):], att[i, Tm - len(p):] = p, 1
        msk[i, Tm - len(p) + a:Tm - len(p) + b] = True
    d = span_direction(tiny_llama, prompts[1][None], 10, 16)
    kw = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
    rec = {}
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=msk, bias=ALWAYS, record=rec) as st:
        batched = tiny_llama.generate(ids, attention_mask=att, **kw)
    assert st.edits == int(msk.sum()) * len(LAYERS)
    for l in LAYERS:
        assert torch.equal(rec[l] != 0, msk)  # exactly the untrusted rows, never pads
    for i, (p, (a, b)) in enumerate(zip(prompts, spans, strict=True)):
        m = torch.zeros(len(p), dtype=torch.bool)
        m[a:b] = True
        with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=m, bias=ALWAYS):
            alone = tiny_llama.generate(p[None], attention_mask=torch.ones(1, len(p), dtype=torch.long), **kw)
        assert batched[i, Tm:].tolist() == alone[0, len(p):].tolist()


def test_sliding_window_offsets(tiny_gemma2):
    ids = torch.arange(5, 29).reshape(1, -1)
    m = torch.zeros(24, dtype=torch.bool)
    m[18:22] = True
    d = span_direction(tiny_gemma2, ids, 18, 22)
    with protect(tiny_gemma2, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=m, bias=ALWAYS):
        full = tiny_gemma2(ids, use_cache=True).logits
    rec = {}
    with protect(tiny_gemma2, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=m, bias=ALWAYS, record=rec):
        first = tiny_gemma2(ids[:, :16], use_cache=True)
        second = tiny_gemma2(ids[:, 16:], past_key_values=first.past_key_values, use_cache=True)
    for l in LAYERS:
        assert rec[l][0].nonzero().squeeze(1).tolist() == [18, 19, 20, 21]
    assert torch.allclose(full[0, 16:], second.logits[0], atol=1e-4)


def test_bf16_dtype_preserved(tiny_llama, ids, mask, d):
    m16 = copy.deepcopy(tiny_llama).to(torch.bfloat16)
    before, after = {}, {}
    pre = m16.model.layers[1].register_forward_hook(lambda m, a, o: before.__setitem__(1, o.clone()))
    with protect(m16, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask):
        post = m16.model.layers[1].register_forward_hook(lambda m, a, o: after.__setitem__(1, o.clone()))
        m16(ids)
        post.remove()
    pre.remove()
    assert after[1].dtype == torch.bfloat16
    assert torch.equal(after[1][0, ~mask], before[1][0, ~mask])


def test_record_then_replay_matches_sizes(tiny_llama, ids, mask, d):
    rec = {}
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask, record=rec):
        tiny_llama(ids)
    R = torch.randn(d.shape, generator=torch.Generator().manual_seed(0))
    R = R / R.norm(dim=-1, keepdim=True)
    before, after = {}, {}
    pre = [tiny_llama.model.layers[l].register_forward_hook(lambda m, a, o, l=l: before.__setitem__(l, o.clone()))
           for l in LAYERS]
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask, replay=(rec, R)):
        post = [tiny_llama.model.layers[l].register_forward_hook(lambda m, a, o, l=l: after.__setitem__(l, o.clone()))
                for l in LAYERS]
        tiny_llama(ids)
        for h in post:
            h.remove()
    for h in pre:
        h.remove()
    for l in LAYERS:
        step = after[l][0, LO:HI] - before[l][0, LO:HI]
        assert torch.allclose(step.norm(dim=-1), rec[l][0, LO:HI].abs(), atol=1e-5)
        nz = rec[l][0, LO:HI] > 0
        cos = torch.nn.functional.cosine_similarity(step[nz], -R[l].expand_as(step[nz]), dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)


def test_mask_batch_mismatch_raises(tiny_llama, d):
    ids = torch.arange(5, 5 + T).reshape(1, -1).repeat(3, 1)
    with pytest.raises(ValueError, match="batch"):
        with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=torch.ones(2, T, dtype=torch.bool)):
            tiny_llama(ids)


def test_bad_layer_raises(tiny_llama, d, mask):
    with pytest.raises(ValueError, match="layer"):
        with protect(tiny_llama, d=d, layers=[7], alpha=1.0, untrusted_mask=mask):
            pass


def test_hooks_removed_after_exit(tiny_llama, ids, mask, d):
    base = tiny_llama(ids).logits
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask):
        pass
    assert torch.equal(base, tiny_llama(ids).logits)


def test_reused_context_with_shorter_prefix_cache(tiny_llama, ids, mask, d):
    """Review #1: a reused context continuing from a prefix cache must use the cache length, not the last sequence."""
    prefix = tiny_llama(ids[:, :12], use_cache=True)  # shared prefix computed earlier, outside nexec
    rec = {}
    with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=mask, bias=ALWAYS, record=rec):
        tiny_llama(ids)  # some other full-length sequence first
        rec.clear()
        tiny_llama(ids[:, 12:], past_key_values=prefix.past_key_values, use_cache=True)
    for l in LAYERS:
        assert rec[l][0].nonzero().squeeze(1).tolist() == [12, 13, 14, 15]


def test_mask_shorter_than_prompt_raises(tiny_llama, ids, d):
    """Review #6: a mask built for a different (shorter) prompt must not be applied silently."""
    short = torch.zeros(12, dtype=torch.bool)
    short[10:] = True
    with pytest.raises(ValueError, match="mask"):
        with protect(tiny_llama, d=d, layers=LAYERS, alpha=1.0, untrusted_mask=short):
            tiny_llama(ids)
