# Step 1 — Core Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the step-0 spike into the installable `nexec` package. It provides:
- a provenance API (`untrusted(...)`, chat messages, spans, masks);
- a batched, span-gated NXDirection hook for HF Transformers;
- versioned direction artifacts loadable through `NX.from_pretrained`;
- a fitter with the full ~175-example corpus;
- the magnitude-matched random control;
- bundled, validated directions for Llama-3.2-3B-Instruct and Gemma-2-2B-it.

**Architecture:** `src/nexec/` holds small single-purpose modules:
- `provenance` renders chat messages into token ids plus an aligned untrusted mask.
- `hook` is the span-gated edit with record/replay support.
- `directions` is the artifact (tensors plus a config) and its resolution logic.
- `corpus` and `fit` estimate directions.
- `nx` is the facade.
- `controls` is the matched-random baseline.

The hook acts on resid_post (the decoder-layer output) via forward hooks. A provenance mask `[B, T]` aligned to the padded `input_ids` is indexed by absolute position, which is tracked by the hook itself. The spike's tested core is the reference: Task 8 proves port parity, and Task 9 deletes `spike/`.

**Tech Stack:** Python ≥ 3.10 (dev on 3.12 via uv), torch ≥ 2.13, transformers 5.20.x, safetensors, huggingface_hub, hatchling build backend, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-25-nexec-design.md`. §0 (step-0 outcome) overrides older sections. Evidence is in `docs/findings/2026-10-step0-spike.md`.

## Global Constraints

- Packages always go into the project venv (`uv sync`, `uv run`). Never install globally.
- Hook point is resid_post: the output of `model.model.layers[l]`. Fitting and intervention use the same point.
- The edit is `h ← h − α·ReLU(h·d − b)·d`, computed in fp32 and cast back to the model dtype.
- Gates and their bias b:
  - `raw`: b = 0
  - `affine`: b = μ_non·d
  - `mid`: b = ½(μ_instr + μ_non)·d
- Only positions marked untrusted are edited. Generated tokens are never edited. With an all-False mask the output must be bitwise identical to the unhooked model.
- Shipped provisional configs:
  - Llama-3.2-3B-Instruct: all layers, α = 1, `raw`, fit context `user`.
  - Gemma-2-2B-it: all layers, α = 1, `affine`, fit context `user`.
  - These are provisional until Phase 3 tunes them on the BIPIA train split.
- Llama's template is rendered with `date_string="26 Jul 2024"`.
- Templates without a system role get the system text merged into the first user turn as `system + "\n\n"`.
- Tool-role content is untrusted by default. A template that rejects a role must raise `TemplateError` with the template's message; never guess a substitute role.
- The matched-random control is record-and-replay, never a per-layer recompute.
- Directions artifact files: `directions.safetensors` (fp32 `d`, `mu_instr`, `mu_non`, each `[L, D]`) and `nexec.json`.
- Loading refuses a model-revision or chat-template mismatch unless `strict=False` is passed.
- The corpus must not contain the salary table, its injection, `spike/paraphrases.json` strings, or BIPIA text.
- Tests run on tiny random configs on CPU, except tests marked `@pytest.mark.slow`, which need cached weights.

## Review Focus

1. **Templates that trim, or JSON-encode, message content.** Both models' templates trim, and Llama renders tool content via `tojson`. Expected: the untrusted mask still covers exactly the untrusted text, and an untrusted part made only of edge whitespace marks nothing. Pinned in Task 2.
2. **Gemma-2's sliding-window cache.** `get_seq_length()` may cap at the window, so an offset derived from it would be wrong past the window. Expected: absolute positions stay correct for prompts longer than `sliding_window`. Pinned in Task 3 with a tiny Gemma2 config, `sliding_window=8` and a 24-token prompt.
3. **Batched left-padded prompts of different lengths.** Expected: each row is edited exactly at its own untrusted tokens, pads are never edited, and per-row outputs equal unbatched outputs on a tiny fp32 model. Pinned in Task 3.
4. **Reusing one `protect` context across several `generate` calls**, or a caller passing a pre-filled cache. Expected: the offset resets for each new sequence. A pre-filled cache at entry uses the cache length. Pinned in Task 3.
5. **Loading directions for a different model or revision.** Expected: a clear `DirectionsMismatchError`, never silent application. Pinned in Task 4.

---

### Task 1: Package scaffold

**Files:**
- Modify: `pyproject.toml`. Add:
  - `[build-system]` (hatchling);
  - `requires-python = ">=3.10"`;
  - dependencies `torch>=2.13`, `transformers>=5.20,<5.21`, `safetensors`, `huggingface_hub`, `numpy`;
  - `matplotlib` moves to the optional extra `viz`;
  - dev group: `pytest`, `ruff`;
  - `[tool.pytest.ini_options] testpaths = ["tests"]`, `markers = ["slow: needs cached model weights"]`, `addopts = "-m 'not slow'"`;
  - `[tool.ruff] line-length = 120`.
- Create: `src/nexec/__init__.py` (exports `__version__ = "0.1.0.dev0"`), `src/nexec/_version.py`, `tests/conftest.py` (tiny Llama and tiny Gemma2 fixtures), `tests/test_package.py`.
- Spike tests stay runnable via `uv run pytest spike/tests` until Task 9.

**Interfaces:**
- Produces: the importable `nexec`, plus fixtures in `tests/conftest.py`:
  - `tiny_llama`: `LlamaConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128, vocab_size=256, max_position_embeddings=256)`, seed 0, eval mode;
  - `tiny_gemma2`: the same sizes plus `head_dim=16`, `sliding_window=8`, `query_pre_attn_scalar=16`, `attn_implementation="eager"`.

- [ ] **Step 1:** Write `tests/test_package.py::test_import_version`, asserting `nexec.__version__ == "0.1.0.dev0"`.
- [ ] **Step 2:** Run `uv run pytest tests -q`. Expected: FAIL, because nexec is not installed or importable.
- [ ] **Step 3:** Update `pyproject.toml`, create `src/nexec/__init__.py` and `_version.py`, then run `uv sync`.
- [ ] **Step 4:** Run `uv run pytest tests -q && uv run ruff check src tests`. Expected: PASS, ruff clean.
- [ ] **Step 5:** Commit `build: src-layout nexec package scaffold`.

### Task 2: Provenance — untrusted parts, rendering, batching, masks

**Files:** Create `src/nexec/provenance.py`. Test in `tests/test_provenance.py`, which uses the real tokenizers (cached) for both models.

**Interfaces:**
- Produces:
  - `untrusted(text: str) -> Untrusted`, a frozen dataclass with field `text`.
  - Message content types: `str | Untrusted | list[str | Untrusted]`.
    - A message with `role == "tool"` and plain `str` content is entirely untrusted.
    - A message dict with `"untrusted": True` is entirely untrusted.
  - `render(tok, messages: list[dict], *, add_generation_prompt=True, template_kwargs=None) -> Rendered`.
    - `Rendered` fields:
      - `input_ids: list[int]`
      - `text: str`
      - `untrusted_mask: list[bool]`
      - `offsets: list[tuple[int, int]]`
      - `untrusted_char_ranges: list[tuple[int, int]]`
    - Algorithm:
      1. Normalize every message into parts `[(text, is_untrusted)]`. Merge a system message into the first user turn if `_supports_system(tok)` is False.
      2. Render the real joined content with `apply_chat_template(tokenize=False, date_string=DATE_STRING, **template_kwargs)`.
      3. For each message in order, locate its joined content in the rendered text after a cursor, trying these encodings in order:
         - identity;
         - `strip()`, which shifts and clips part ranges by the stripped leading length;
         - JSON (`json.dumps(content)`), where part ranges come from escaping each part separately, valid because JSON escaping is per-character.
      4. If no encoding is found, raise `TemplateError("content of message {i} not found verbatim in rendered template")`.
      5. Tokenize the text with `add_special_tokens=False, return_offsets_mapping=True`. A token is untrusted iff its offset range overlaps an untrusted range. Straddling tokens count as untrusted.
    - A role rejected by the template raises `TemplateError`, chaining the original exception.
  - `encode(tok, conversations: list[list[dict]], **render_kw) -> Encoded`.
    - `Encoded` fields:
      - `input_ids: LongTensor[B, T]`
      - `attention_mask: LongTensor[B, T]`
      - `untrusted_mask: BoolTensor[B, T]`
      - `renders: list[Rendered]`
    - Padding is on the left. The pad id is `tok.pad_token_id`, falling back to `tok.eos_token_id`. Pads are never untrusted.
  - `mask_from_token_spans(n_tokens: int, spans: list[tuple[int, int]]) -> BoolTensor[n_tokens]`. Spans are half-open; raises `ValueError` on out-of-range spans.
  - `mask_from_char_spans(r: Rendered, spans: list[tuple[int, int]]) -> BoolTensor[T]`. Uses the overlap rule.
  - `DATE_STRING = "26 Jul 2024"`.
  - `class TemplateError(ValueError)`.

- [ ] **Step 1: Write the tests.** Each test runs on both tokenizers unless noted.
  - `test_untrusted_part_roundtrip`: user content `["Table:\n", untrusted(T), "\nQuestion?"]` gives masked decoded text that contains `T.strip()`, with length ≤ `len(T) + 2`.
  - `test_matches_official_tokenization`: an all-trusted render's `input_ids` equal `tok.apply_chat_template(..., tokenize=True, date_string=DATE_STRING)`.
  - `test_trimmed_edges`: content `[untrusted("  padded  "), " q"]` masks exactly the tokens of `"padded"`. Content `[untrusted("   "), "q"]` masks nothing.
  - `test_tool_message_untrusted_llama` (Llama only): `[user "q", assistant "calling", tool 'line1\n"quoted" result']` masks tokens whose decoded text equals the JSON-encoded tool content, within straddle slack. User and assistant tokens are unmasked.
  - `test_tool_role_rejected_gemma` (Gemma only): raises `TemplateError` matching `roles must alternate`.
  - `test_system_merged_when_unsupported`: Gemma rendering with a system message puts the system text before the user text. Llama keeps a separate system header.
  - `test_untrusted_flag_message`: `{"role": "user", "content": "doc", "untrusted": True}` masks the `doc` tokens only, not the role header.
  - `test_encode_left_pads`: three conversations of different lengths give an `Encoded` whose rows are left-padded. Each row's non-pad slice equals its own `render().input_ids`, and the mask is shifted to match. Pads have `attention_mask == 0` and are unmasked.
  - `test_mask_from_token_spans`: `[(2, 5)]` over 8 tokens gives `[F,F,T,T,T,F,F,F]`; `(7, 9)` raises `ValueError`.
  - `test_mask_from_char_spans`: matches the overlap rule for one span.
  - `test_render_deterministic_date` (Llama): `"Oct 2026"` does not appear in the text.
- [ ] **Step 2:** Run `uv run pytest tests/test_provenance.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `src/nexec/provenance.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS for both tokenizers.
- [ ] **Step 5:** Commit `feat: provenance rendering with untrusted parts, tool roles, batching`.

### Task 3: Hook — batched, span-gated, absolute positions, record/replay

**Files:** Create `src/nexec/hook.py`. Test in `tests/test_hook.py`.

**Interfaces:**
- Produces:
  - `edit(h: Tensor[N, D], d: Tensor[D], alpha: float, b: float) -> Tensor[N, D]`, with fp32 math.
  - `@contextmanager protect(model, *, d: Tensor[L, D], layers: list[int], alpha: float, untrusted_mask: BoolTensor[B, T] | BoolTensor[T], bias: Tensor[L] | None = None, record: dict[int, Tensor] | None = None, replay: tuple[dict[int, Tensor], Tensor[L, D]] | None = None) -> HookState`
    - `HookState` fields:
      - `edits: int` (token-rows processed, summed over layers)
      - `offset: int`
      - `started: bool`
    - Position tracking:
      - A forward pre-hook on `model.model` (with kwargs) reads the cache length `n = cache.get_seq_length()`. A missing cache counts as 0.
      - If `n == 0` the state resets: `started = True`, `offset = 0`.
      - Otherwise, if `started`, `offset = state.next`; if not started, `offset = n`, and `started` becomes True.
      - A forward hook on `model.model` sets `state.next = offset + T`, where T is the length of the current call.
      - Never trust `get_seq_length` beyond the reset test. Gemma-2's sliding cache can cap it.
    - Layer hook, for an output of shape `[B, T, D]`:
      - Positions are `offset + arange(T)`. Rows are selected where `pos < mask.shape[1]` and `mask[b, pos]`.
      - The selected rows are edited in fp32 and written back into a clone.
      - With `record` given, `record[l] = coef` as a `[B, T_mask]` CPU fp32 tensor, zero outside the edited rows.
      - With `replay = (coefs, R)` given, use `coefs[l][b, pos]` as the coefficient and `R[l]` as the direction, with no ReLU recompute.
      - A 1-D mask broadcasts to batch size 1.
      - Batch size is taken from the output shape. A mask/batch mismatch raises `ValueError`.

- [ ] **Step 1: Write the tests.** Use `tiny_llama` unless noted. `d` is the untrusted span's own mean direction, as in the spike, so the ReLU fires.
  - `test_relu_math`: four cases:
    - raw with α = 1: 3 → 0
    - −2 stays unchanged
    - α = 2: 3 → −3
    - affine with b = 1: 3 → 1
  - `test_masked_rows_untouched`: at every hooked layer, the trusted output rows are `torch.equal` to the pre-hook rows, and the untrusted rows differ.
  - `test_all_trusted_identity`: logits are `torch.equal` with and without `protect`, for an all-False mask.
  - `test_prefix_invariance`: logits before the first untrusted token are equal; logits after it differ.
  - `test_decode_no_edit`: inside `generate(max_new_tokens=8)`, `edits == mask.sum() * len(layers)`.
  - `test_chunked_prefill_matches`: a two-call prefill with a cache gives the same last-chunk logits (atol 1e-5) and the same `edits` as a single call.
  - `test_generate_reuse_resets`: two consecutive `generate` calls inside one `protect` both give `edits == mask.sum() * len(layers)`, so the second call resets its offset.
  - `test_prefilled_cache_at_entry`: the first 12 tokens are prefilled *outside* `protect`. Entering `protect` and running the remaining 12 with that cache edits absolute positions 12–15 of a mask that is True at 10–15. `record` shows nonzeros only at 12–15.
  - `test_batch_left_padded_equals_unbatched`: three prompts of lengths 20, 24 and 16, left-padded, each with its own untrusted span. Each row's generated tokens, and logits at its non-pad positions, equal the same prompt run alone (atol 1e-5). Pads are never edited (checked via `record`: coef is 0 at pad positions).
  - `test_sliding_window_offsets` (`tiny_gemma2`): a 24-token prompt with `sliding_window = 8`, the mask True at positions 18–21, and a two-chunk prefill (16 then 8). The edits land at absolute positions 18–21, checked via `record` nonzeros, and the logits match a single-call prefill (atol 1e-4).
  - `test_bf16_dtype_preserved`: the output dtype stays bf16 and the trusted rows are bit-identical.
  - `test_record_then_replay_matches_sizes`: per layer and row, `‖Δh‖` under replay with random unit R equals the recorded |coef| (atol 1e-5), and Δh is parallel to −R[l].
  - `test_mask_batch_mismatch_raises`: a mask with B = 2 against a batch of 3 raises `ValueError`.
  - `test_hooks_removed_after_exit`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `src/nexec/hook.py`.
- [ ] **Step 4:** Run the tests. Expected: all PASS.
- [ ] **Step 5:** Commit `feat: batched span-gated NX hook with absolute positions and record/replay`.

### Task 4: Directions artifact and resolution

**Files:** Create `src/nexec/directions.py`. Test in `tests/test_directions.py`.

**Interfaces:**
- Produces:
  - `@dataclass Directions`:
    - tensors: `d`, `mu_instr`, `mu_non` (`[L, D]` fp32);
    - `config: DirectionsConfig`.
  - `@dataclass DirectionsConfig`:
    - `model_id: str`
    - `model_revision: str | None`
    - `chat_template_sha256: str | None`
    - `hook_point: str = "resid_post"`
    - `layers: list[int]`
    - `alpha: float = 1.0`
    - `gate: Literal["raw", "affine", "mid"] = "raw"`
    - `fit_context: str = "user"`
    - `corpus_version: str = ""`
    - `corpus_sha256: str = ""`
    - `nexec_version: str`
    - `created: str`
    - `diagnostics: dict`
    - `format_version: int = 1`
  - `Directions.bias() -> Tensor[L] | None`: None for raw; per-layer b for affine and mid.
  - `Directions.save(path_dir)` writes `directions.safetensors` and `nexec.json`.
  - `Directions.load(path_dir) -> Directions`.
  - `Directions.check_compatible(model, tok, strict=True)`. It raises `DirectionsMismatchError` (a `ValueError`) when any of these differ, unless `strict=False`, which warns instead:
    - `model.config.num_hidden_layers` or `hidden_size` against the tensors;
    - `model.config._commit_hash` against `model_revision`, when both are known;
    - `sha256(tok.chat_template)` against `chat_template_sha256`, when both are known.
  - `resolve(model_id: str, source: str | Path | None = None) -> Path | None`:
    - an explicit local dir; else
    - an explicit `hf://<repo>[/<subdir>]` via `huggingface_hub.snapshot_download`; else
    - the user cache `~/.cache/nexec/<slug>/`; else
    - the package-bundled `nexec/directions/<slug>/`; else
    - None.
    - The slug is `model_id.replace("/", "--")`.
  - `user_cache_dir(model_id) -> Path`. Honors `NEXEC_CACHE`.

- [ ] **Step 1: Write the tests.**
  - `test_save_load_roundtrip`: tensors are `torch.equal` and the config compares equal.
  - `test_bias_by_gate`: raw gives None; affine gives `(mu_non*d).sum(-1)`; mid gives the midpoint.
  - `test_mismatch_layers_raises`, `test_mismatch_revision_raises`, `test_mismatch_template_raises`: each raises `DirectionsMismatchError`. With `strict=False` it warns via `pytest.warns(UserWarning)`.
  - `test_resolve_order`: using `tmp_path` and monkeypatched cache and bundle dirs, an explicit dir wins over the cache, the cache over the bundle, and missing everything gives None.
  - `test_resolve_hf_uri`: with `snapshot_download` monkeypatched to return a tmp dir, `"hf://org/repo/sub"` resolves to `<tmp>/sub`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: versioned directions artifact with compatibility checks and resolution`.

### Task 5: Corpus v1 (~175 examples, single- and multi-turn)

**Files:**
- Create `src/nexec/corpus.py` (schema, loader, validator, builder helpers).
- Create `scripts/build_corpus.py`, the source of truth written as segments. It generates `src/nexec/data/corpus_v1.jsonl`.
- Create `docs/corpus-spotcheck.md`.
- Test in `tests/test_corpus.py`.

**Interfaces:**
- Produces:
  - `@dataclass Span(start, end, label)`.
  - `@dataclass Turn(role: Literal["user", "assistant"], text: str, spans: list[Span])`. Assistant turns have no spans.
  - `@dataclass Example(id, category, split, imperative_content: bool, turns: list[Turn])`.
  - `load_corpus(path=None) -> list[Example]`. The default is the packaged `corpus_v1.jsonl`.
  - `validate(ex) -> list[str]`. The rules:
    - Spans are sorted, non-overlapping and in range.
    - Every non-whitespace char of a user turn is covered.
    - Labels are drawn from the three allowed values.
    - At least one instruction span and one non_instruction span exist per example.
    - The first turn is a user turn, and turns alternate.
  - `corpus_sha256(path=None) -> str`.
  - `CORPUS_VERSION = "v1"`.

- [ ] **Step 1: Write the tests.**
  - `test_corpus_valid`: every example validates.
  - `test_corpus_composition`:
    - 170 ≤ n ≤ 190;
    - all 18 categories present, each with at least 6 examples;
    - at least 50 instruction-after-content examples;
    - at least 40 `imperative_content` examples;
    - at least 15 multi-turn examples (content in an earlier user turn, instruction in a later one);
    - heldout is 20% ± 2 and covers all 18 categories.
  - `test_ids_unique`.
  - `test_no_eval_leakage`: no example text contains "compromised", "Marketing", or any line of `spike/paraphrases.json`. The paraphrases are copied into `tests/data/eval_strings.json` so the test survives Task 9.
  - `test_validate_catches_errors`: overlap, uncovered text, bad label, assistant turn with spans, and a missing instruction span are each reported.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `src/nexec/corpus.py` and the builder helpers. The builder keeps the spike's segment format, with multi-turn examples written as a list of turns, each a list of segments.
- [ ] **Step 4: Author the content.**
  - Drafting is delegated to a subagent with the labeling rules:
    - role, not mood;
    - commands inside the operated-on content are `non_instruction`;
    - framing and connectors ("Here is my essay:", ":", "—") are `boundary`;
    - original text only;
    - no eval strings.
  - The 44 pilot examples are carried over as the seed.
  - A second, independent subagent audits every label against the rules and returns a list of disputed spans. Rule on each dispute and record it in the ledger.
- [ ] **Step 5:** Run the build script, then the tests. Expected: PASS.
- [ ] **Step 6:** Write `docs/corpus-spotcheck.md`: 20 examples chosen with a fixed seed of 7, rendered with their labels shown as `[I:…]`, `[N:…]` and `[B:…]`, for the user's review. This is non-blocking.
- [ ] **Step 7:** Commit `feat: corpus v1 (175 examples incl. multi-turn) with audit and spot-check sheet`.

### Task 6: Fitter and diagnostics

**Files:**
- Create `src/nexec/fit.py`. Test in `tests/test_fit.py`.
- Create `scripts/fit_directions.py`, the CLI that writes bundled artifacts.

**Interfaces:**
- Consumes: `render` (Task 2), `Example` (Task 5), `Directions` and `DirectionsConfig` (Task 4).
- Produces:
  - `token_labels(tok, ex: Example, context: Literal["user", "document"] = "user") -> tuple[list[int], list[str | None]]`.
    - Renders every turn of the example as chat messages.
    - Labels a token by the span it lies entirely inside, after stripping leading whitespace.
    - Template, assistant, BOS and straddling tokens get None.
    - `document` prefixes the first user turn with `"Here is a document:\n"`.
  - `capture(model, input_ids: list[int]) -> Tensor[L, T, D]`: fp32, CPU.
  - `fit_labeled(model, items) -> tuple[Tensor, Tensor, Tensor, int, int]`, returning `(d, mu_instr, mu_non, n_instr, n_non)`. Exclusions follow the spike: label None, boundary, and ‖h‖ > 10× the per-prompt, per-layer median.
  - `fit_directions(model, tok, model_id, *, examples=None, context="user", layers=None, alpha=1.0, gate="raw") -> Directions`.
    - Fits on the train split.
    - `diagnostics` records:
      - per-layer heldout AUC;
      - per-layer `mu_instr·d` and `mu_non·d`;
      - per-layer `frac_non_pos`;
      - `n_instr` and `n_non`;
      - `max_untrusted_norm_ratio_note`, which handles the deferred step-0 minor: the share of fit tokens excluded as norm outliers, per layer.
    - Fills `model_revision`, `chat_template_sha256`, `corpus_sha256` and `created`.

- [ ] **Step 1: Write the tests.** Use `tiny_llama` plus a real Llama tokenizer for the labels test only.
  - `test_fit_shapes_and_norm`.
  - `test_fit_matches_manual_means`.
  - `test_outlier_excluded`: exact equality with the manual mean that excludes the outlier.
  - `test_token_labels_multiturn` (Llama tokenizer): for a two-turn example, the decoded instruction tokens equal the second user turn's instruction text, and the assistant-turn tokens are None.
  - `test_fit_directions_config`: the returned config carries the corpus sha, the gate, the layers (default all), and diagnostics with L entries.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: direction fitter with diagnostics`.

### Task 7: NX facade and controls

**Files:**
- Create `src/nexec/nx.py` and `src/nexec/controls.py`.
- Modify `src/nexec/__init__.py`: export `NX`, `untrusted`, `Directions`, `protect`, `encode`, `render`, `TemplateError` and `DirectionsMismatchError`.
- Test in `tests/test_nx.py`.

**Interfaces:**
- Produces:
  - `class NX` with attributes `model`, `tok`, `directions`.
    - `NX.from_pretrained(model_id, *, directions=None, model=None, tokenizer=None, device=None, dtype=torch.bfloat16, fit_if_missing=False, strict=True) -> NX`.
      - Loads the model with Gemma using eager attention, onto device `mps` if available, else `cuda`, else `cpu`.
      - Resolves directions via `resolve()`.
      - If nothing is found and `fit_if_missing`, it fits, saves to the user cache, and logs that it did so.
      - Otherwise it raises `FileNotFoundError`, naming the CLI.
      - Calls `check_compatible`.
    - `NX.encode(conversations: list[dict] | list[list[dict]]) -> Encoded`. A single conversation becomes a batch of 1.
    - `NX.protect(untrusted_mask, *, layers=None, alpha=None, gate=None)` is a context manager. Overrides default to the artifact config.
    - `NX.generate(conversations, *, defend=True, layers=None, alpha=None, gate=None, **generate_kwargs) -> list[str]`.
      - Greedy by default (`do_sample=False`, `max_new_tokens=256` unless given).
      - Returns the decoded new text per row.
      - `defend=False` runs undefended through the same rendering.
  - `controls.matched_random(nx, conversations, *, seed: int, **kw) -> list[str]`.
    - Pass 1 records coefficients with the real directions over the prompt (forward only).
    - Pass 2 generates with replay along `random_unit(shape, seed)`.
  - `controls.random_unit(shape, seed) -> Tensor`.

- [ ] **Step 1: Write the tests.** Use `tiny_llama`, a stub tokenizer from `tests/conftest.py`, and tmp directions built from `fit_labeled` on stub items.
  - `test_from_pretrained_with_injected_model_and_directions`.
  - `test_generate_defend_false_equals_plain_generate`: token ids are identical.
  - `test_generate_all_trusted_equals_undefended`.
  - `test_protect_overrides`: an `alpha` override changes `edits`/the outputs relative to the default; `layers=[0]` limits the hooked layers.
  - `test_missing_directions_raises_with_hint`.
  - `test_matched_random_replays_recorded_sizes`.
  - `test_generate_batch_returns_one_string_per_row`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: NX facade and matched-random control`.

### Task 8: Fit, bundle and validate directions; port parity with the spike

**Files:**
- Create `src/nexec/directions/meta-llama--Llama-3.2-3B-Instruct/`, `src/nexec/directions/google--gemma-2-2b-it/`, `scripts/validate.py`, `tests/test_slow_salary.py` (`@pytest.mark.slow`) and `docs/findings/2026-10-step1-validation.md`.

- [ ] **Step 1: Port parity.** A one-time `scripts/validate.py parity` check, recorded in the findings note:
  - `nexec.fit.fit_labeled` on the spike pilot corpus (user context) reproduces `spike/out/Llama-3.2-3B-Instruct/user/fit.safetensors` with `allclose` atol 1e-4.
  - `NX` with that pilot fit, at all layers, α = 1, raw, on salary md_before, reproduces the spike's recorded undefended, defended and defended_clean strings from `docs/findings/step0-data/salary_sweep.csv`. Those are 200-character prefixes and must match exactly.
- [ ] **Step 2: Fit and save the shipped directions.**
  - Run `uv run python scripts/fit_directions.py --model meta-llama/Llama-3.2-3B-Instruct --gate raw`.
  - Run it again with `--model google/gemma-2-2b-it --gate affine`.
  - Expected: artifacts written, with the diagnostics' heldout AUC ≥ 0.9 over the mid layers.
- [ ] **Step 3: Write `tests/test_slow_salary.py`.** Using bundled directions via `NX.from_pretrained` on Llama md_before:
  - undefended contains "compromised";
  - defended matches 85k and does not contain "compromised";
  - clean defended matches 85k.
- [ ] **Step 4:** Run `uv run pytest -m slow -q`. Expected: PASS.
- [ ] **Step 5: Run `scripts/validate.py full`.** It covers:
  - salary: 3 renderings × both models, undefended / NX / matched-random (seeds 0–4);
  - paraphrases: 10 × both models × undefended / NX / matched-random (seeds 0–2), scored with the step-0 rules;
  - the 10-document utility check, with NX on and off.

  Write the step-1 validation note comparing v1-corpus directions against the step-0 pilot numbers. No tuning on these results; they are reporting only. If v1 is clearly worse than the pilot (more paraphrases followed on both models), stop and investigate the corpus before shipping.
- [ ] **Step 6:** Commit `feat: bundled v1 directions for Llama-3.2-3B and Gemma-2-2B with validation`.

### Task 9: Retire the spike and document usage

**Files:**
- Delete `spike/`.
- Modify `README.md`: a short usage section with the two-line API, install via `uv sync`, an NVIDIA citation, and a limitations line. The full README is Phase 4.
- Modify `pyproject.toml`: drop `spike` from any test paths.

- [ ] **Step 1:** `git rm -r spike`. Check that no file in `src/`, `tests/` or `scripts/` imports from `spike`, using `grep -rn "spike" src tests scripts`. Expected: no import lines. Path strings in `validate.py` parity mode are allowed, and documented as historical.
- [ ] **Step 2:** Run `uv run pytest -q`, `uv run pytest -m slow -q` and `uv run ruff check .`. Expected: all pass and clean.
- [ ] **Step 3:** Commit `chore: retire step-0 spike; README usage`.
