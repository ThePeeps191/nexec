"""The shipped direction artifacts must load, match the packaged corpus, and carry the shipped settings."""

import pytest
import torch

from nexec.corpus import CORPUS_VERSION, corpus_sha256
from nexec.directions import Directions, resolve

# (gate, alpha, n_layers, d_model). Step 3.5 kept alpha 1 for both (docs/findings/2026-10-step3.5-residual.md).
SHIPPED = {"meta-llama/Llama-3.2-3B-Instruct": ("raw", 1.0, 28, 3072),
           "google/gemma-2-2b-it": ("affine", 1.0, 26, 2304)}


@pytest.mark.parametrize("model_id", sorted(SHIPPED))
def test_bundled_directions(model_id, monkeypatch, tmp_path):
    monkeypatch.setenv("NEXEC_CACHE", str(tmp_path))  # ignore any user-cache override
    gate, alpha, n_layers, d_model = SHIPPED[model_id]
    path = resolve(model_id)
    assert path is not None and "nexec/directions" in str(path)
    dirs = Directions.load(path)
    assert dirs.d.shape == (n_layers, d_model)
    assert torch.allclose(dirs.d.norm(dim=-1), torch.ones(n_layers), atol=1e-4)
    c = dirs.config
    assert (c.model_id, c.gate, c.alpha, c.fit_context) == (model_id, gate, alpha, "user")
    assert c.layers == list(range(n_layers))
    assert c.corpus_version == CORPUS_VERSION
    assert c.corpus_sha256 == corpus_sha256(), "corpus changed since these directions were fit: refit them"
    assert c.model_revision and c.chat_template_sha256


LICENSE_FILES = {
    "meta-llama/Llama-3.2-3B-Instruct": (["NOTICE.md", "LICENSE.txt", "USE_POLICY.md"],
                                         ["Built with Llama", "Llama 3.2 is licensed under the Llama 3.2 Community "
                                          "License, Copyright © Meta Platforms, Inc. All Rights Reserved."]),
    "google/gemma-2-2b-it": (["NOTICE.md", "GEMMA_TERMS_OF_USE.txt", "GEMMA_PROHIBITED_USE_POLICY.txt"],
                             ["Gemma is provided under and subject to the Gemma Terms of Use found at "
                              "ai.google.dev/gemma/terms."]),
}


@pytest.mark.parametrize("model_id", sorted(LICENSE_FILES))
def test_bundled_directions_carry_their_license_terms(model_id):
    """Llama 3.2 license §1.b.i and Gemma Terms §3.1: ship a copy of the terms and the exact notice text."""
    from nexec.directions import BUNDLED_DIR, slug

    files, notices = LICENSE_FILES[model_id]
    d = BUNDLED_DIR / slug(model_id)
    for f in files:
        assert (d / f).stat().st_size > 500 or f == "NOTICE.md", f
    notice = (d / "NOTICE.md").read_text()
    for line in notices:
        assert line in notice
