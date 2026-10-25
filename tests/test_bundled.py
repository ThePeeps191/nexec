"""The shipped direction artifacts must load, match the packaged corpus, and carry the shipped settings."""

import pytest
import torch

from nexec.corpus import CORPUS_VERSION, corpus_sha256
from nexec.directions import Directions, resolve

SHIPPED = {"meta-llama/Llama-3.2-3B-Instruct": ("raw", 28, 3072), "google/gemma-2-2b-it": ("affine", 26, 2304)}


@pytest.mark.parametrize("model_id", sorted(SHIPPED))
def test_bundled_directions(model_id, monkeypatch, tmp_path):
    monkeypatch.setenv("NEXEC_CACHE", str(tmp_path))  # ignore any user-cache override
    gate, n_layers, d_model = SHIPPED[model_id]
    path = resolve(model_id)
    assert path is not None and "nexec/directions" in str(path)
    dirs = Directions.load(path)
    assert dirs.d.shape == (n_layers, d_model)
    assert torch.allclose(dirs.d.norm(dim=-1), torch.ones(n_layers), atol=1e-4)
    c = dirs.config
    assert (c.model_id, c.gate, c.alpha, c.fit_context) == (model_id, gate, 1.0, "user")
    assert c.layers == list(range(n_layers))
    assert c.corpus_version == CORPUS_VERSION
    assert c.corpus_sha256 == corpus_sha256(), "corpus changed since these directions were fit: refit them"
    assert c.model_revision and c.chat_template_sha256
