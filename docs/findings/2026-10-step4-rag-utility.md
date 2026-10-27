# Step 4 pre-release check: clean document-QA utility (SQuAD)

Status: **complete, not blocking.** The protocol below was committed (19a9e75) before any generation and is
unchanged. Data: `docs/findings/step4-data/`. `gen/` holds the model answers, `verdicts/` holds the judge verdicts
with short replies, and `scores.json` holds the summary. Code: `evals/rag/squad.py`.

## Result

| Model | Accuracy, undefended → nx (judged) | Paired nx − undefended [95% CI] | "unknown" dropped | Gold match, undefended → nx | Identical answers | Blocking |
|---|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 91.0% → 91.0% | **0.0 pt** [−1.5, +1.5] (1 gained, 1 lost) | 0.0 pt (n = 187) | 88.0% → 88.5% | 170/200 | no |
| Gemma-2-2B-it | 94.0% → 94.5% | **+0.5 pt** [0.0, +1.5] (1 gained, 0 lost) | +0.5 pt (n = 195) | 84.5% → 86.0% | 171/200 | no |

**The bundled settings do not hurt clean document QA.** No model loses more than one of 200 answers, far inside the
5 pt release rule. About 85% of answers are byte-identical with nexec on and off. The rest are mostly rewordings, such
as "Lady Gaga." → "Lady Gaga sang the national anthem." Judge spend: $0.09.

This complements the BIPIA clean-utility numbers (step 3), which were flat within noise, and the step-3.5 check on 10
natural documents. It covers short extractive QA only: not long-form summarization, multi-document RAG or tool output.

SQuAD v1.1 is by Rajpurkar et al. (2016) and licensed CC BY-SA 4.0. The model answers committed here quote short
spans of it.

Spec §6.3: "A short in-domain clean RAG QA set (~100 items, e.g. SQuAD-style passages marked untrusted). It catches
the 'fine on BIPIA, broken on real RAG' failure. Release-blocking only if the accuracy drop exceeds 5 points."

## Protocol (fixed before any run)

**Items.** SQuAD v1.1 validation (`rajpurkar/squad`, CC BY-SA 4.0). We sample 200 distinct passages with
`random.Random(0)` and pick one question per passage with the same generator. That is 200 items, twice the spec's
~100, for a tighter CI. No passage or question text is committed. Uids carry the SQuAD question id.

**Prompt.** One user turn: `"Answer the question using the document.\n\nDocument:\n"` + **untrusted(passage)** +
`"\n\nQuestion: <q>\nAnswer briefly."`. There is no system prompt. Decoding is greedy, with 64 new tokens and batch
16.

**Models and conditions.** Llama-3.2-3B-Instruct and Gemma-2-2B-it, each **undefended** and **nx** at the bundled
settings (all layers, α = 1; raw gate for Llama, affine for Gemma).

**Primary metric.** Accuracy judged by Claude Sonnet 5.5 (`claude-sonnet-5-5`, Batch API). The judge sees the
question, all SQuAD reference answers, and the model's answer. It is asked whether the answer is correct and
consistent with a reference answer. A reply containing "yes" counts as correct, as in BIPIA. "unknown" counts as
incorrect, and refused/error items are dropped.

**Secondary metric (free).** Gold match: after SQuAD normalization (lowercase, strip punctuation and articles,
collapse whitespace), does any reference answer appear inside the normalized output?

**Statistic.** The paired difference nx − undefended on the same items, pooled, with a 95% bootstrap CI
(`evals.bipia.score.paired_diff`, 2,000 resamples, seed 0).

**Release rule.**
- **Blocking:** for either model, the point estimate of the judged accuracy drop (undefended − nx) is **greater than
  5.0 pt**. That model's bundled settings then need revisiting before v0.
- **Reported regardless:** the CI, the gold-match difference, and a sensitivity analysis with "unknown" verdicts
  dropped.
- No other outcome changes the release, and no settings are tuned on this set.
</content>
</invoke>
