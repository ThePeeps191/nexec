# nexec

**No-execute bits for LLM tokens.** Mark the tokens that came from outside: tool output, retrieved documents,
emails, web pages. nexec removes their *instruction-like* component inside the model, so the model can still read
them but stops obeying them.

This is a reimplementation of NVIDIA's **NXDirection** (Tamás Vörös, NVIDIA AI Security, Oct 2026,
[research page](https://research.nvidia.com/ai-security/nxdirection-no-execute-along-instruction-direction)). The
science is theirs; this repo provides a provenance API, a HuggingFace Transformers implementation, versioned
direction files, and evaluations.

> **Status: v0 in progress.** The core library works and is tested. The demo, heatmap and BIPIA benchmark come
> next. Do not rely on it as a security boundary (see Limitations).

## Usage

```python
from nexec import NX, untrusted

nx = NX.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")   # loads the model + bundled directions
answer = nx.generate([
    {"role": "system", "content": "Answer the question using only the table data."},
    {"role": "user", "content": [untrusted(table_from_somewhere), "\n\nWhat is the average salary in Marketing?"]},
])
```

- `untrusted(...)` marks part of a message as data. Messages with `role="tool"` are untrusted as a whole.
- Pass a list of conversations to batch them.
- `nx.generate(..., defend=False)` runs the same prompt undefended.
- Lower level: `nx.encode(...)` returns `input_ids`, `attention_mask` and a token-aligned `untrusted_mask`, and
  `with nx.protect(mask): model.generate(...)` wraps your own generation loop.

**Bundled directions:** Llama-3.2-3B-Instruct (raw gate) and Gemma-2-2B-it (affine gate). Fit directions for
another model with `NX.from_pretrained(model_id, fit_if_missing=True)` or `scripts/fit_directions.py`. The
settings (all layers, α = 1) are provisional until they are tuned on BIPIA.

## How it works

1. **Find the direction, once per model.** A small labeled corpus (`src/nexec/data/corpus_v1.jsonl`, 178 short
   examples) tags tokens as *instruction* (the user's request) or *content* (what it operates on, even when the
   content is itself phrased as a command). The per-layer direction is
   `d = normalize(mean(h | instruction) − mean(h | content))`.
2. **Apply it at inference time.** For every untrusted token at each chosen layer:
   `h ← h − α · ReLU(h·d − b) · d`. Tokens that already read as content are left alone. Trusted tokens (system
   prompt, user request, chat template) are never edited.

Step-0 evidence from this repo (`docs/findings/2026-10-step0-spike.md`): NVIDIA's salary-table demo flips on
Llama-3.2-3B. Random directions with the same per-token edit sizes do nothing, and ordinary documents are answered
byte-identically.

## Limitations

- **Mitigation, not a guarantee.** Paraphrased injections are reduced, not eliminated. Keep sandboxes and tool
  allowlists, and keep secrets out of prompts.
- **Needs hidden states.** It works with open-weight models in HF Transformers, not with closed APIs.
- **Provenance is your job.** nexec does not detect injections. It only acts on tokens you mark. Tool messages
  (`role="tool"` / `"ipython"`) are untrusted by default. Any other unknown role must be marked explicitly.
- **Special-token text inside untrusted content.** A literal string such as `<|eot_id|>` inside untrusted content
  is tokenized as a real control token, exactly as `apply_chat_template` does. Those tokens are still masked and
  edited, but strip or escape such strings from external text if your threat model includes forged chat turns.
- **Not yet supported with `defend=True`:** beam search and `num_return_sequences > 1`.

## Development

```bash
uv sync                       # project venv only; nothing is installed globally
uv run pytest                 # fast tests (tiny random models, CPU)
uv run pytest -m slow         # end-to-end salary demo (needs cached weights)
```
