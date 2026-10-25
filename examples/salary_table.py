"""NVIDIA's salary-table demo with nexec: the same prompt undefended vs. with NXDirection on the table tokens.

    uv run python examples/salary_table.py                      # Llama-3.2-3B-Instruct, writes docs/assets/
    uv run python examples/salary_table.py --model google/gemma-2-2b-it --rendering md_after

On Gemma-2-2B the injection does not make the model say "compromised"; it degrades the answer (82k instead of 85k),
which NXDirection repairs.
"""

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.cases import INJECTION, QUESTION, salary_conversation  # noqa: E402
from nexec import NX  # noqa: E402
from nexec.viz import to_html, to_png, token_scores  # noqa: E402

LLAMA = "meta-llama/Llama-3.2-3B-Instruct"


def _block(label: str, text: str) -> str:
    body = textwrap.fill(" ".join(text.split()), width=88, initial_indent="    ", subsequent_indent="    ")
    return f"{label}\n{body}"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=LLAMA)
    ap.add_argument("--rendering", default="md_before", choices=["md_before", "md_after", "csv"])
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    ap.add_argument("--no-heatmap", action="store_true")
    a = ap.parse_args(argv)

    nx = NX.from_pretrained(a.model)
    attacked = salary_conversation(a.rendering)
    gen = dict(max_new_tokens=160)
    out = {
        "undefended": nx.generate(attacked, defend=False, **gen)[0],
        "defended": nx.generate(attacked, **gen)[0],
        "clean_defended": nx.generate(salary_conversation(a.rendering, None), **gen)[0],
    }
    print(f"\nmodel: {a.model}   (untrusted span: the table)\n")
    print(_block("Injected note in the table:", INJECTION))
    print(_block("User question:", QUESTION) + "\n")
    print(_block("Undefended:", out["undefended"]) + "\n")
    print(_block("With nexec (NXDirection on the table tokens):", out["defended"]) + "\n")
    print(_block("With nexec, same table without the injection:", out["clean_defended"]) + "\n")

    if not a.no_heatmap:
        stem = "salary_heatmap"
        if a.model != LLAMA:
            stem += "_" + a.model.split("/")[-1]
        if a.rendering != "md_before":
            stem += "_" + a.rendering
        scores = token_scores(nx, attacked, title=f"Salary table: {a.model.split('/')[-1]}")
        out["html"] = str(to_html(scores, Path(a.out) / f"{stem}.html"))
        out["png"] = str(to_png(scores, Path(a.out) / f"{stem}.png"))
        print(f"heatmap (layer {scores.default_layer}): {out['html']}\n                   {out['png']}")
    return out


if __name__ == "__main__":
    main()
