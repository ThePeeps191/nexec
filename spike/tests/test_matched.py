import torch

from fit import capture
from hook import nx_hooks
from matched import record_coefs, replay_edit, patched_edit

T, LO, HI = 24, 10, 16
LAYERS = [0, 1, 2]


def _setup(tiny_model):
    ids = torch.arange(5, 5 + T).reshape(1, -1)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[LO:HI] = True
    h = capture(tiny_model, ids[0])
    v = h[:, LO:HI].mean(1)
    d = v / v.norm(dim=-1, keepdim=True)
    rnd = torch.randn(d.shape, generator=torch.Generator().manual_seed(0))
    return ids, mask, d, rnd / rnd.norm(dim=-1, keepdim=True)


def test_replay_applies_recorded_edit_sizes_per_layer_and_token(tiny_model):
    ids, mask, d, rnd = _setup(tiny_model)
    coefs = record_coefs(tiny_model, ids, d, LAYERS, 1.0, mask)
    assert set(coefs) == set(LAYERS) and all(c.shape == (HI - LO,) for c in coefs.values())

    before, after = {}, {}
    pre = [tiny_model.model.layers[l].register_forward_hook(lambda m, a, o, l=l: before.__setitem__(l, o.clone())) for l in LAYERS]
    with patched_edit(replay_edit(coefs, d, rnd)), nx_hooks(tiny_model, d, LAYERS, 1.0, mask):
        post = [tiny_model.model.layers[l].register_forward_hook(lambda m, a, o, l=l: after.__setitem__(l, o.clone())) for l in LAYERS]
        tiny_model(ids)
        for h in post:
            h.remove()
    for h in pre:
        h.remove()
    for l in LAYERS:
        delta = (after[l][0, LO:HI] - before[l][0, LO:HI]).norm(dim=-1)
        assert torch.allclose(delta, coefs[l].abs(), atol=1e-5), l
        # and the edit is along the random direction, not d
        step = after[l][0, LO:HI] - before[l][0, LO:HI]
        nz = coefs[l] > 0
        if nz.any():
            cos = torch.nn.functional.cosine_similarity(step[nz], -rnd[l].expand_as(step[nz]), dim=-1)
            assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)


def test_recorded_coefs_equal_real_hook_edit_sizes(tiny_model):
    ids, mask, d, _ = _setup(tiny_model)
    coefs = record_coefs(tiny_model, ids, d, LAYERS, 1.0, mask)
    before, after = {}, {}
    pre = [tiny_model.model.layers[l].register_forward_hook(lambda m, a, o, l=l: before.__setitem__(l, o.clone())) for l in LAYERS]
    with nx_hooks(tiny_model, d, LAYERS, 1.0, mask):
        post = [tiny_model.model.layers[l].register_forward_hook(lambda m, a, o, l=l: after.__setitem__(l, o.clone())) for l in LAYERS]
        tiny_model(ids)
        for h in post:
            h.remove()
    for h in pre:
        h.remove()
    for l in LAYERS:
        delta = (after[l][0, LO:HI] - before[l][0, LO:HI]).norm(dim=-1)
        assert torch.allclose(delta, coefs[l], atol=1e-5), l
