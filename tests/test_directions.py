import hashlib
from types import SimpleNamespace

import pytest
import torch

from nexec import directions as dirs
from nexec.directions import Directions, DirectionsConfig, DirectionsMismatchError, resolve

L, D = 3, 8


def make(model_revision="abc123", template_sha=None, gate="raw"):
    g = torch.Generator().manual_seed(0)
    d = torch.randn(L, D, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    cfg = DirectionsConfig(model_id="org/model", model_revision=model_revision, chat_template_sha256=template_sha,
                           layers=list(range(L)), alpha=1.0, gate=gate, diagnostics={"heldout_auc": [0.9] * L})
    return Directions(d, torch.randn(L, D, generator=g), torch.randn(L, D, generator=g), cfg)


def fake_model_tok(layers=L, hidden=D, rev="abc123", template="{{ messages }}"):
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden, _commit_hash=rev))
    return model, SimpleNamespace(chat_template=template)


def test_save_load_roundtrip(tmp_path):
    a = make()
    a.save(tmp_path / "x")
    b = Directions.load(tmp_path / "x")
    assert torch.equal(a.d, b.d) and torch.equal(a.mu_instr, b.mu_instr) and torch.equal(a.mu_non, b.mu_non)
    assert a.config == b.config
    assert (tmp_path / "x" / "directions.safetensors").exists() and (tmp_path / "x" / "nexec.json").exists()


def test_bias_by_gate():
    a = make(gate="raw")
    assert a.bias() is None
    a.config.gate = "affine"
    assert torch.allclose(a.bias(), (a.mu_non * a.d).sum(-1))
    a.config.gate = "mid"
    assert torch.allclose(a.bias(), ((a.mu_instr + a.mu_non) / 2 * a.d).sum(-1))


def test_compatible_ok():
    tpl = "{{ messages }}"
    a = make(template_sha=hashlib.sha256(tpl.encode()).hexdigest())
    a.check_compatible(*fake_model_tok(template=tpl))


def test_mismatch_layers_raises():
    with pytest.raises(DirectionsMismatchError, match="layers"):
        make().check_compatible(*fake_model_tok(layers=L + 1))
    with pytest.warns(UserWarning):
        make().check_compatible(*fake_model_tok(layers=L + 1), strict=False)


def test_mismatch_revision_raises():
    with pytest.raises(DirectionsMismatchError, match="revision"):
        make().check_compatible(*fake_model_tok(rev="zzz999"))
    with pytest.warns(UserWarning):
        make().check_compatible(*fake_model_tok(rev="zzz999"), strict=False)


def test_mismatch_template_raises():
    a = make(template_sha=hashlib.sha256(b"other").hexdigest())
    with pytest.raises(DirectionsMismatchError, match="template"):
        a.check_compatible(*fake_model_tok())


def test_resolve_order(tmp_path, monkeypatch):
    cache, bundle, explicit = tmp_path / "cache", tmp_path / "bundle", tmp_path / "explicit"
    monkeypatch.setenv("NEXEC_CACHE", str(cache))
    monkeypatch.setattr(dirs, "BUNDLED_DIR", bundle)
    assert resolve("org/model") is None
    make().save(bundle / "org--model")
    assert resolve("org/model") == bundle / "org--model"
    make().save(cache / "org--model")
    assert resolve("org/model") == cache / "org--model"
    make().save(explicit)
    assert resolve("org/model", explicit) == explicit


def test_resolve_hf_uri(tmp_path, monkeypatch):
    make().save(tmp_path / "snap" / "sub")
    monkeypatch.setattr(dirs, "snapshot_download", lambda repo_id, **kw: str(tmp_path / "snap"))
    assert resolve("org/model", "hf://org/repo/sub") == tmp_path / "snap" / "sub"
