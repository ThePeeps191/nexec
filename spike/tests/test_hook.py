import copy

import pytest
import torch

from fit import capture
from hook import edit, nx_hooks

T, LO, HI = 24, 10, 16  # prompt length; untrusted span [LO, HI)
LAYERS = [0, 1, 2]


@pytest.fixture
def ids():
    return torch.arange(5, 5 + T).reshape(1, -1)


@pytest.fixture
def mask():
    m = torch.zeros(T, dtype=torch.bool)
    m[LO:HI] = True
    return m


@pytest.fixture
def d(tiny_model, ids):
    # Point d along the span's own mean so the ReLU is guaranteed to fire there.
    h = capture(tiny_model, ids[0])
    v = h[:, LO:HI].mean(1)
    return v / v.norm(dim=-1, keepdim=True)


def test_relu_math():
    dvec = torch.tensor([1.0, 0.0])
    assert torch.allclose(edit(torch.tensor([[3.0, 5.0]]), dvec, 1.0, 0.0), torch.tensor([[0.0, 5.0]]))
    assert torch.equal(edit(torch.tensor([[-2.0, 5.0]]), dvec, 1.0, 0.0), torch.tensor([[-2.0, 5.0]]))
    assert torch.allclose(edit(torch.tensor([[3.0, 5.0]]), dvec, 2.0, 0.0), torch.tensor([[-3.0, 5.0]]))
    # affine gate: only the excess over b is removed
    assert torch.allclose(edit(torch.tensor([[3.0, 5.0]]), dvec, 1.0, 1.0), torch.tensor([[1.0, 5.0]]))


def test_masked_rows_untouched(tiny_model, ids, mask, d):
    before, after = {}, {}
    pre = [layer.register_forward_hook(lambda m, a, o, l=l: before.__setitem__(l, o.clone()))
           for l, layer in enumerate(tiny_model.model.layers)]
    try:
        with nx_hooks(tiny_model, d, LAYERS, 1.0, mask):
            post = [layer.register_forward_hook(lambda m, a, o, l=l: after.__setitem__(l, o.clone()))
                    for l, layer in enumerate(tiny_model.model.layers)]
            tiny_model(ids)
            for h in post:
                h.remove()
    finally:
        for h in pre:
            h.remove()
    for l in LAYERS:
        assert torch.equal(after[l][0, ~mask], before[l][0, ~mask])
        assert not torch.equal(after[l][0, mask], before[l][0, mask])


def test_all_trusted_identity(tiny_model, ids, d):
    base = tiny_model(ids).logits
    with nx_hooks(tiny_model, d, LAYERS, 1.0, torch.zeros(T, dtype=torch.bool)):
        out = tiny_model(ids).logits
    assert torch.equal(base, out)


def test_prefix_invariance(tiny_model, ids, mask, d):
    base = tiny_model(ids).logits
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask):
        out = tiny_model(ids).logits
    assert torch.equal(base[0, :LO], out[0, :LO])
    assert not torch.allclose(base[0, HI:], out[0, HI:])


def test_decode_no_edit(tiny_model, ids, mask, d):
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask) as state:
        tiny_model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False)
    assert state.edits == int(mask.sum()) * len(LAYERS)


def test_chunked_prefill_matches(tiny_model, ids, mask, d):
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask):
        full = tiny_model(ids).logits
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask) as state:
        first = tiny_model(ids[:, :12], use_cache=True)
        second = tiny_model(ids[:, 12:], past_key_values=first.past_key_values, use_cache=True)
    assert state.edits == int(mask.sum()) * len(LAYERS)
    assert torch.allclose(full[0, 12:], second.logits[0], atol=1e-5)


def test_bf16_dtype_preserved(tiny_model, ids, mask, d):
    m16 = copy.deepcopy(tiny_model).to(torch.bfloat16)
    before, after = {}, {}
    pre = m16.model.layers[1].register_forward_hook(lambda m, a, o: before.__setitem__(1, o.clone()))
    with nx_hooks(m16, d, LAYERS, 1.0, mask):
        post = m16.model.layers[1].register_forward_hook(lambda m, a, o: after.__setitem__(1, o.clone()))
        m16(ids)
        post.remove()
    pre.remove()
    assert after[1].dtype == torch.bfloat16
    assert torch.equal(after[1][0, ~mask], before[1][0, ~mask])


def test_empty_cache_first_call(tiny_model, ids, mask, d):
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask) as state:
        tiny_model(ids, past_key_values=None, use_cache=False)
    assert state.edits == int(mask.sum()) * len(LAYERS)


def test_hooks_removed_after_exit(tiny_model, ids, mask, d):
    base = tiny_model(ids).logits
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask):
        pass
    assert torch.equal(base, tiny_model(ids).logits)
