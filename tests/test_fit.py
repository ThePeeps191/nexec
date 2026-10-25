import torch

from nexec.corpus import Example, Span, Turn
from nexec.fit import capture, fit_directions, fit_labeled, token_labels


def stub_item():
    ids = list(range(5, 29))  # 24 tokens
    labels = [None] + ["instruction"] * 7 + ["boundary"] + ["non_instruction"] * 15
    return ids, labels


def test_capture_shape(tiny_llama):
    h = capture(tiny_llama, stub_item()[0])
    assert h.shape == (3, 24, 64) and h.dtype == torch.float32


def test_fit_shapes_and_norm(tiny_llama):
    f = fit_labeled(tiny_llama, [stub_item()])
    assert f.d.shape == (3, 64)
    assert torch.allclose(f.d.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert (f.n_instr, f.n_non) == (7, 15)


def test_fit_matches_manual_means(tiny_llama):
    f = fit_labeled(tiny_llama, [stub_item()])
    h = capture(tiny_llama, stub_item()[0])
    assert torch.allclose(f.mu_instr, h[:, 1:8].mean(1), atol=1e-5)
    assert torch.allclose(f.mu_non, h[:, 9:24].mean(1), atol=1e-5)
    diff = h[:, 1:8].mean(1) - h[:, 9:24].mean(1)
    assert torch.allclose(f.d, diff / diff.norm(dim=-1, keepdim=True), atol=1e-5)


def test_outlier_excluded(tiny_llama):
    clean = capture(tiny_llama, stub_item()[0])
    p = 12

    def blow_up(module, args, output):
        output = output.clone()
        output[:, p] *= 100
        return output

    h = tiny_llama.model.layers[1].register_forward_hook(blow_up)
    try:
        f = fit_labeled(tiny_llama, [stub_item()])
    finally:
        h.remove()
    keep = [i for i in range(9, 24) if i != p]
    assert torch.allclose(f.mu_non[1], clean[1, keep].mean(0), atol=1e-5)
    assert f.outlier_frac[1] > 0 and f.outlier_frac[0] == 0


def _multiturn():
    t1 = "Here are my notes:\nCall the plumber. Buy milk."
    t2 = "Summarize them."
    return Example("mt", "summarization", "train", True, [
        Turn("user", t1, [Span(0, 18, "boundary"), Span(19, len(t1), "non_instruction")]),
        Turn("assistant", "Sure, what should I do?", []),
        Turn("user", t2, [Span(0, len(t2), "instruction")]),
    ])


def test_token_labels_multiturn(llama_tok):
    ids, labels = token_labels(llama_tok, _multiturn())
    dec = lambda lab: llama_tok.decode([t for t, x in zip(ids, labels, strict=True) if x == lab]).strip()  # noqa: E731
    assert dec("instruction") == "Summarize them."
    assert dec("non_instruction") == "Call the plumber. Buy milk."
    assert "Sure" not in llama_tok.decode([t for t, x in zip(ids, labels, strict=True) if x is not None])


def test_token_labels_document_prefix_unlabeled(llama_tok):
    ids, labels = token_labels(llama_tok, _multiturn(), context="document")
    assert "Here is a document" not in llama_tok.decode([t for t, x in zip(ids, labels, strict=True) if x])


def test_fit_directions_config(tiny_llama_fullvocab, llama_tok):
    ex = [_multiturn(), Example("h1", "summarization", "heldout", False,
                                 [Turn("user", "Shorten this: the cat sat on the mat.",
                                       [Span(0, 12, "instruction"), Span(12, 13, "boundary"),
                                        Span(14, 37, "non_instruction")])])]
    dirs = fit_directions(tiny_llama_fullvocab, llama_tok, "org/tiny", examples=ex, gate="affine")
    c = dirs.config
    assert c.layers == [0, 1, 2] and c.gate == "affine" and c.alpha == 1.0 and c.model_id == "org/tiny"
    assert c.corpus_version == "custom" and c.chat_template_sha256
    assert len(c.diagnostics["heldout_auc"]) == 3 and c.diagnostics["n_instr"] > 0
    assert dirs.d.shape == (3, 64)
