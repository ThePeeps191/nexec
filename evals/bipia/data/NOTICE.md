# Third-party data: BIPIA

The files in this directory come from **BIPIA** (Benchmark for Indirect Prompt Injection Attacks), Microsoft,
<https://github.com/microsoft/BIPIA>, commit `a004b69`. They are MIT-licensed; the license text is in `LICENSE`.

Copied **unmodified** from that commit (BIPIA path → path here):

| BIPIA | here |
|---|---|
| `benchmark/email/{train,test}.jsonl` | `email/{train,test}.jsonl` |
| `benchmark/table/{train,test}.jsonl` | `table/{train,test}.jsonl` |
| `benchmark/code/{train,test}.jsonl` | `code/{train,test}.jsonl` |
| `benchmark/abstract/index.json` | `abstract/index.json` |
| `benchmark/abstract/md5.txt` | `abstract/md5.txt` |
| `benchmark/text_attack_{train,test}.json` | `text_attack_{train,test}.json` |
| `benchmark/code_attack_{train,test}.json` | `code_attack_{train,test}.json` |
| `LICENSE` | `LICENSE` |

**XSum text is not redistributed.** The abstract task's contexts and summaries come from XSum
(`EdinburghNLP/xsum` on the Hugging Face Hub, under its own terms). `abstract/index.json` holds only row indices;
`evals/bipia/data.py` downloads XSum and selects those rows the way BIPIA's `benchmark/abstract/process.py` does,
checks the result against `abstract/md5.txt`, and caches it under `evals/bipia/cache/` (gitignored).

BIPIA's prompt templates, insertion functions and attack-evaluator registry are ported (not copied as files) in
`evals/bipia/data.py` and `evals/bipia/asr.py`, which credit BIPIA in their headers. No BIPIA code is executed.
