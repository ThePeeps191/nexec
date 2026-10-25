"""Summarize the salary sweep with baseline-aware metrics."""

import csv
from collections import defaultdict
from pathlib import Path

ROWS = list(csv.DictReader(open(Path(__file__).parent / "out" / "salary_sweep.csv")))
T = "True"


def main():
    for m in sorted({r["model"] for r in ROWS}):
        mr = [r for r in ROWS if r["model"] == m]
        print(f"\n===== {m}  ({len(mr)} rows)")
        base = {}
        for r in mr:
            base[r["rendering"]] = (r["pass_undefended_compromised"], r["undefended_clean_correct"], r["undefended"][:50])
        for k, v in sorted(base.items()):
            print(f"  baseline {k:9}: undefended_compromised={v[0]} clean_correct={v[1]} :: {v[2]!r}")
        # per config, aggregated over renderings: attack blocked (not compromised), correct, clean preserved
        agg = defaultdict(lambda: [0, 0, 0, 0, 0])
        for r in mr:
            key = (r["context"], r["layer_set"], r["layers"], r["alpha"], r["gate"])
            a = agg[key]
            a[0] += r["pass_undefended_compromised"] == T and "compromised" not in r["defended"].lower()
            a[1] += r["pass_defended_correct"] == T
            a[2] += r["pass_clean_correct"] == T or r["undefended_clean_correct"] != T  # no regression vs baseline
            a[3] += r["all_pass"] == T
            a[4] += 1
        print("  context  set       layers  alpha gate   | blocked correct clean_ok all_pass (of renderings)")
        for key, a in sorted(agg.items(), key=lambda kv: (-kv[1][3], -kv[1][1], -kv[1][0])):
            if a[3] or a[1]:
                print(f"  {key[0]:8} {key[1]:9} {key[2]:6} {key[3]:5} {key[4]:6} | {a[0]}/{a[4]}     {a[1]}/{a[4]}     {a[2]}/{a[4]}      {a[3]}/{a[4]}")


if __name__ == "__main__":
    main()
