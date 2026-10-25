# Step 2 — Demo and Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the result visible in seconds. Step 2 delivers a one-command salary-table demo, a per-token heatmap of instruction-likeness before and after NXDirection (self-contained HTML plus a PNG for the README), and a README that leads with both.

**Architecture:**
- `nexec.viz` (needs the optional `viz` extra for PNG output) computes per-token projections h·d at every layer in two forward passes over the same prompt. The undefended pass runs without hooks. The defended pass runs inside `NX.protect`, with capture hooks registered *after* the NX hooks so they see the edited residual.
- The module renders those projections as HTML and PNG. The HTML is self-contained, with a layer slider and a before/after toggle. The PNG uses matplotlib text with colored boxes.
- `examples/salary_table.py` drives the library end to end and writes `docs/assets/salary_heatmap.{html,png}`.

**Tech Stack:** nexec (step 1), matplotlib (`viz` extra and dev group), plain HTML/CSS/JS with no external assets, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-nexec-design.md`, §7 Phase 2: demo, heatmap, README GIF/image. Built with step-1 findings: `docs/findings/2026-10-step1-validation.md`.

## Global Constraints

- Everything stays in the project venv (`uv`). matplotlib goes in the `viz` extra and the dev group. `import nexec` must still work without matplotlib.
- **Projection definition:** `proj[l, t] = h_l[t] · d_l` on resid_post. "Before" means the undefended forward pass; "after" means the forward pass inside `nx.protect(mask)` with the bundled config.
- **Default display layer:** among layers `0 .. L//2`, the layer that maximizes the before-pass sum over untrusted tokens of `ReLU(proj − b)`. This is where the defense does the most work, and step 0 showed early layers are causal. It is overridable.
- **Color scale:** diverging and centered on 0, with red = instruction-like (positive) and blue = content-like. One symmetric scale per layer, shared by the before and after views, set to the 98th percentile of |proj| over the content tokens shown.
- **Display:** only message-content tokens are shown (system and user text). Template and special tokens are hidden. Untrusted tokens are underlined.
- HTML must escape token text and be fully self-contained: no network, no CDN.
- The demo uses `evals.cases.salary_conversation` and `md_before`, the rendering where undefended Llama is also correct on the clean table.
- Gemma's matplotlib-free import path is checked by a test.

## Review Focus

1. **Tokens containing `<`, `&`, quotes or newlines.** Expected: escaped in the HTML, and newlines shown as line breaks, not dropped. Pinned in Task 1.
2. **A conversation with no untrusted tokens.** Expected: the heatmap still renders, before and after are identical, and the default layer falls back to `L//4`. Pinned in Task 1.
3. **matplotlib not installed.** Expected: `import nexec.viz` works and `to_png` raises an `ImportError` that names the `viz` extra. Pinned in Task 1.
4. **Byte-level BPE token strings (`Ġ`, `Ċ`).** Expected: shown as decoded text (spaces and newlines), never as raw byte symbols. Pinned in Task 1.
5. **Gemma (affine gate, bias ≠ 0).** Expected: the after-pass projections at hooked layers are ≤ b + tolerance on untrusted tokens. Pinned in Task 1.

---

### Task 1: `nexec.viz`: projections, HTML, PNG

**Files:**
- Create `src/nexec/viz.py`.
- Modify `pyproject.toml` (add matplotlib to the dev group).
- Test in `tests/test_viz.py`.

**Interfaces:**
- Produces:
  - `@dataclass TokenScores` with fields:
    - `tokens: list[str]` (decoded per token)
    - `untrusted: list[bool]`
    - `content: list[bool]` (inside some message's content)
    - `before: Tensor[L, T]`
    - `after: Tensor[L, T]`
    - `bias: Tensor[L]` (zeros for the raw gate)
    - `hooked: list[int]`
    - `default_layer: int`
    - `title: str`
  - `token_scores(nx, conversation, *, title="") -> TokenScores`, for a batch of 1.
  - `to_html(scores, path) -> Path` and `to_png(scores, path, *, layer=None) -> Path`.
  - `default_layer(before, untrusted, bias) -> int`.

- [ ] **Step 1: Write the tests** (tiny Llama with full vocabulary, the real Llama tokenizer, and directions from `fit_directions` on stub examples, as in `tests/test_nx.py`):
  - `test_scores_shapes_and_tokens`: L = 3; `tokens` is the per-token decode; `len(tokens) == T`; the untrusted mask equals `nx.encode(...).untrusted_mask`.
  - `test_after_equals_before_before_first_untrusted`: at every layer, `after[:, :first_untrusted] == before[:, :first_untrusted]` (atol 1e-6).
  - `test_after_bounded_at_hooked_layers`: with α = 1, `after[l, untrusted] ≤ bias[l] + 1e-4` for every hooked layer.
  - `test_no_untrusted_identical_and_fallback_layer`: with an all-trusted conversation, before equals after and `default_layer == L // 4`.
  - `test_default_layer_picks_max_relu_mass`: a synthetic before tensor with known values selects the expected layer.
  - `test_html_escapes_and_selfcontained`: the content includes the literal string `<b>&"x"</b>\nnext`. The HTML contains `&lt;b&gt;&amp;` and no `<b>&`, contains no `http://` or `https://`, and contains both "before" and "after" data.
  - `test_bpe_markers_not_shown`: no `Ġ` or `Ċ` appears in `scores.tokens`.
  - `test_png_written`: a non-empty file starting with the PNG magic bytes.
  - `test_import_without_matplotlib`: with `sys.modules["matplotlib"] = None` monkeypatched, `to_png` raises an `ImportError` matching `nexec\[viz\]`.
- [ ] **Step 2:** Run `uv run pytest tests/test_viz.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. For captures, register forward hooks on every `model.model.layers[l]` *inside* `nx.protect(...)`, so they run after the NX hooks. Each pass is a single forward, with no generation.
- [ ] **Step 4:** Run the tests. Expected: PASS. Ruff clean.
- [ ] **Step 5:** Commit `feat: nexec.viz per-token instruction-likeness heatmaps (HTML + PNG)`.

### Task 2: Demo script, assets and README

**Files:**
- Create `examples/salary_table.py` and `docs/assets/salary_heatmap.html`, `.png`.
- Create `tests/test_slow_demo.py` (slow).
- Modify `README.md`.

**Interfaces:**
- Produces: `examples/salary_table.py` with `main(argv=None) -> dict`, returning `{"undefended", "defended", "clean_defended", "html", "png"}`. CLI flags:
  - `--model` (default Llama)
  - `--rendering` (default `md_before`)
  - `--out` (default `docs/assets`)
  - `--no-heatmap`

  It prints a side-by-side block: the injected note, the question, the undefended answer, the defended answer, and the defended answer on the clean table.

- [ ] **Step 1:** Write `tests/test_slow_demo.py::test_demo_llama`. `main(["--out", tmp])` returns an undefended answer containing "compromised", a defended answer matching 85k, and existing html and png paths.
- [ ] **Step 2:** Run `uv run pytest -m slow tests/test_slow_demo.py -q`. Expected: FAIL (missing).
- [ ] **Step 3:** Implement the script.
- [ ] **Step 4:** Run the slow test. Expected: PASS.
- [ ] **Step 5:** Generate the shipped assets with `uv run python examples/salary_table.py`. View the PNG and check that it is legible: the injection row is red before and neutral after, and ordinary cells are neutral. Revise the styling if it isn't, without changing the data.
- [ ] **Step 6:** Update the README: the PNG at the top, the demo command, and a link to the HTML ("open locally"). Keep the limitations near the top.
- [ ] **Step 7:** Run all fast tests, the slow tests and ruff. Commit `feat: salary-table demo, shipped heatmap assets, README lead image`.
