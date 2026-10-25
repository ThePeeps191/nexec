import torch

from fit import capture, fit_labeled, load_fit, save_fit


def test_capture_shape(tiny_model, stub_item):
    h = capture(tiny_model, stub_item[0])
    assert h.shape == (3, 24, 64) and h.dtype == torch.float32


def test_fit_shapes_and_norm(tiny_model, stub_item):
    f = fit_labeled(tiny_model, [stub_item])
    assert f.d.shape == (3, 64)
    assert torch.allclose(f.d.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert (f.n_instr, f.n_non) == (7, 15)


def test_fit_matches_manual_means(tiny_model, stub_item):
    f = fit_labeled(tiny_model, [stub_item])
    h = capture(tiny_model, stub_item[0])
    assert torch.allclose(f.mu_instr, h[:, 1:8].mean(1), atol=1e-5)
    assert torch.allclose(f.mu_non, h[:, 9:24].mean(1), atol=1e-5)
    manual = h[:, 1:8].mean(1) - h[:, 9:24].mean(1)
    assert torch.allclose(f.d, manual / manual.norm(dim=-1, keepdim=True), atol=1e-5)


def test_outlier_excluded(tiny_model, stub_item):
    clean = capture(tiny_model, stub_item[0])
    p = 12  # a non_instruction position

    def blow_up(module, args, output):
        output = output.clone()
        output[:, p] *= 100
        return output

    handle = tiny_model.model.layers[1].register_forward_hook(blow_up)
    try:
        f = fit_labeled(tiny_model, [stub_item])
    finally:
        handle.remove()
    keep = [i for i in range(9, 24) if i != p]
    assert torch.allclose(f.mu_non[1], clean[1, keep].mean(0), atol=1e-5)


def test_save_load_roundtrip(tiny_model, stub_item, tmp_path):
    f = fit_labeled(tiny_model, [stub_item])
    save_fit(f, tmp_path / "fit.safetensors")
    g = load_fit(tmp_path / "fit.safetensors")
    assert torch.equal(f.d, g.d) and torch.equal(f.mu_non, g.mu_non) and g.n_instr == f.n_instr
