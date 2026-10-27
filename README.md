# nexec

**No-execute bits for LLM tokens.** Mark the text that came from outside (tool output, retrieved documents, emails,
web pages) and nexec removes its *instruction-like* component inside the model. The model still reads that text but
stops obeying it.

![NVIDIA's salary-table example on Llama-3.2-3B-Instruct, swept through all 28 layers: undefended the model obeys the note hidden in the table; with nexec it answers the question](https://raw.githubusercontent.com/ThePeeps191/nexec/main/docs/assets/salary_layers.gif)

```python
from nexec import NX, untrusted

nx = NX.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")      # model + bundled directions
# table: any text from outside (a tool result, a retrieved page) that may carry an injected instruction
nx.generate([{"role": "user", "content": [untrusted(table), "\n\nWhat is the average salary in Marketing?"]}])
```

> **Read this first.** nexec is a *mitigation layer*, not a security boundary.
> - On BIPIA, 3–8% of injections still get through.
> - It acts only on text you mark as untrusted; it does not detect injections.
> - It needs open weights in HF Transformers.
>
> Keep sandboxes, tool allowlists and least privilege. See
> [Limitations](https://github.com/ThePeeps191/nexec#limitations).

**Credit.** nexec reimplements NVIDIA's **NXDirection** (Tamás Vörös, NVIDIA AI Security, Oct 2026,
[research page](https://research.nvidia.com/ai-security/nxdirection-no-execute-along-instruction-direction)). The
method is theirs. This repo adds:
- a provenance API;
- a Hugging Face Transformers implementation;
- versioned direction files;
- independent evaluations.

Closest related work:
- **CounterSteer** (Russinovich, [arXiv 2609.36570](https://arxiv.org/abs/2609.36570)) subtracts a fixed steering
  direction from tool-result tokens at prefill.
- **Semantic Overlays** (Penman, [arXiv 2608.23873](https://arxiv.org/abs/2608.23873),
  [code](https://github.com/JoshuaSP/semantic-overlays)) uses trained adapters to mark spans as non-executable.

All three share the idea of an edit inside the model, gated by provenance.

## Install

```bash
pip install nexec            # or: uv add nexec
pip install "nexec[viz]"     # + matplotlib, for PNG heatmaps
```

Python ≥ 3.10, PyTorch, Transformers 5.x (results measured with 5.20). The bundled models are gated on the Hugging Face Hub: accept the Llama 3.2
or Gemma 2 license there and log in (`hf auth login`) before the first download. The device is picked
automatically (Apple MPS, CUDA or CPU). All results here were measured on Apple MPS.

To run the demo from a clone (it prints both answers and rewrites the heatmaps):

```bash
git clone https://github.com/ThePeeps191/nexec && cd nexec
uv sync && uv run python examples/salary_table.py
```

## Results

All results are on the bundled settings: all layers, α = 1, raw gate for Llama, affine gate for Gemma.

**BIPIA.** Email, table, abstract and code tasks; 3,300 attacked and 300 clean test samples per condition; Claude
Sonnet 5.5 as judge.

| Model | Attack success, undefended → nexec | Random direction, same edit sizes | Clean utility, undefended → nexec |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 22.9% → **8.3%** | 22.3% | 55.0% → 55.5% |
| Gemma-2-2B-it | 22.0% → **3.0%** | 21.7% | 47.0% → 49.5% |

**Clean document QA.** 200 SQuAD questions, with the passage marked untrusted.

| Model | Accuracy, undefended → nexec | Paired difference [95% CI] | Identical answers |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 91.0% → 91.0% | 0.0 pt [−1.5, +1.5] | 170/200 |
| Gemma-2-2B-it | 94.0% → 94.5% | +0.5 pt [0.0, +1.5] | 171/200 |

- **Attacks that need semantic judging almost vanish:** Llama 13.9% → 0.9%, Gemma 15.0% → 0.5%.
- **Style-only and code injections partly survive, mostly on Llama.** Examples are "use emojis in your answer" and
  malicious code pasted into a Stack Overflow answer. Llama's rule-scored attack success only falls 41.6% → 24.7%
  (Gemma: 38.1% → 7.6%).
  - The direction does recognize these injections; at α = 1 the edit is just too weak for them.
  - Stronger settings are opt-in: Llama with the affine gate at α = 4 reaches 0.6%, but costs 5.7 pt of clean
    utility. Gemma at α = 2 reaches 1.8%.
- **The direction is doing the work.** A random direction with identical per-token edit sizes does nothing.
- **Utility is unchanged within noise** on both BIPIA and SQuAD.
- **NVIDIA reports lower residuals** (0.1% / 0.2%) over 5 tasks, with an unnamed judge. We do not reproduce those.

Reports:
- [BIPIA](https://github.com/ThePeeps191/nexec/blob/main/docs/findings/2026-10-step3-bipia.md)
- [residual diagnosis and retune](https://github.com/ThePeeps191/nexec/blob/main/docs/findings/2026-10-step3.5-residual.md)
- [SQuAD](https://github.com/ThePeeps191/nexec/blob/main/docs/findings/2026-10-step4-rag-utility.md)

Every generation and verdict is committed under `docs/findings/`.

## Usage

```python
from nexec import NX, untrusted

nx = NX.from_pretrained("google/gemma-2-2b-it")
table = "| Name | Dept | Salary |\n|---|---|---|\n| Bob | Mkt | 82k |\n| Eve | Mkt | 88k |"   # untrusted text
out = nx.generate(
    [
        {"role": "system", "content": "Answer the question using only the table data."},
        {"role": "user", "content": [untrusted(table), "\n\nWhat is the average salary in Marketing?"]},
    ],
    max_new_tokens=160,
)
```

- `untrusted(...)` marks part of a message as data. Messages with `role="tool"` are untrusted as a whole.
- Pass a list of conversations to batch them. `generate` returns one string per conversation.
- `nx.generate(..., defend=False)` runs the same prompt undefended. `alpha=` and `layers=` override the bundled
  settings.
- At a lower level, `nx.encode(...)` returns `input_ids`, `attention_mask` and a token-aligned `untrusted_mask`.
  `with nx.protect(mask): model.generate(...)` wraps your own generation loop.
- `nexec.viz` draws per-token heatmaps: an interactive HTML layer slider, and PNG with the `viz` extra.

**Bundled directions** cover Llama-3.2-3B-Instruct (raw gate) and Gemma-2-2B-it (affine gate).
- They are derived from those models and ship under their terms: the Llama 3.2 Community License ("Built with Llama")
  and the Gemma Terms of Use. Copies sit next to each direction file in `nexec/directions/`.
- For any other Transformers decoder, `NX.from_pretrained(model_id, fit_if_missing=True)` fits directions from the
  packaged corpus and caches them in `~/.cache/nexec`. That took 21 s for Llama-3.2-3B, model load included,
  on the M-series Mac used for the results here.
  `scripts/fit_directions.py` does the same.
- Fitted directions are untested on BIPIA for models other than these two.

## How it works

1. **Find the direction, once per model.** A small labeled corpus tags tokens as *instruction* (the user's request)
   or *content* (what the request operates on, even when that content is itself phrased as a command). The direction
   is computed per layer: `d = normalize(mean(h | instruction) − mean(h | content))`.
2. **Apply it at inference time.** For every untrusted token at each layer: `h ← h − α · ReLU(h·d − b) · d`.
   - Tokens that already read as content are left alone.
   - Trusted tokens (system prompt, user request, chat template) are never edited.
   - Because the edit is in the residual stream, it carries through the KV cache into everything the model generates
     afterwards.

## Limitations

- **Mitigation, not a guarantee.** On BIPIA, 3–8% of injections still succeed.
  - Code-shaped injections and terse style requests are the weakest cases.
  - Paraphrased injections are reduced, not eliminated.
  - Adaptive attacks have not been evaluated: injections written to read as content, or optimized against the
    direction itself.
  - Keep sandboxes and tool allowlists, and keep secrets out of prompts.
- **Needs hidden states.** It works with open-weight models in HF Transformers, not with closed APIs. There is no
  vLLM, llama.cpp or MLX backend yet.
- **Provenance is your job.** nexec does not detect injections; it only acts on tokens you mark. Tool messages
  (`role="tool"` / `"ipython"`) are untrusted by default. Any other unknown role must be marked explicitly.
- **Small models only so far.** Evaluated on Llama-3.2-3B-Instruct and Gemma-2-2B-it.
- **Utility evidence is task-specific.** BIPIA and short extractive QA are flat. On 10 ordinary documents, answers
  were byte-identical in 9/10 (Llama) and 10/10 (Gemma). Long-form generation and multi-document RAG are untested.
- **Special-token text inside untrusted content.** A literal string such as `<|eot_id|>` inside untrusted content
  is tokenized as a real control token, exactly as `apply_chat_template` does. Those tokens are still masked and
  edited, but strip or escape such strings from external text if your threat model includes forged chat turns.
- **Not yet supported with `defend=True`:** beam search and `num_return_sequences > 1`.

## Development

```bash
uv sync                       # project venv only; nothing is installed globally
uv run pytest                 # fast tests (tiny random models, CPU); gated-tokenizer tests skip without HF access
uv run pytest -m slow         # end-to-end salary demo + heatmap (needs cached weights)
```

The design spec is in `docs/superpowers/specs/`. Each step's findings, with raw data, are in `docs/findings/`.

## License

The code is MIT. The bundled direction files are derived from Llama 3.2 and Gemma 2 and ship under those models'
terms (see the `NOTICE.md` in each directory).

Built with Llama.
