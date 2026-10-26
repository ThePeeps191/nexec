"""BIPIA pipeline: build samples -> generate (per condition) -> judge -> score -> tune.

    uv run python -m evals.bipia.pipeline generate --model meta-llama/Llama-3.2-3B-Instruct --split test \
        --conditions undefended nx matched
    uv run python -m evals.bipia.pipeline judge --model ... --split test --tags undefended nx matched
    uv run python -m evals.bipia.pipeline score --model ... --split test --tags undefended nx matched
    uv run python -m evals.bipia.pipeline tune --model ...

Generations and per-item verdicts are written under docs/findings/step3-data/ (committed). The judge cache with full
prompts stays in evals/bipia/cache/ (gitignored: utility prompts quote XSum reference summaries).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from evals.bipia import asr
from evals.bipia.data import build_clean, build_samples, conversation
from evals.bipia.judge import judge_many, spend_usd, utility_prompt
from evals.bipia.run import read_records, run
from evals.bipia.score import summarize_asr, summarize_utility

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "findings" / "step3-data"
CACHE = ROOT / "evals" / "bipia" / "cache" / "judge_cache.jsonl"
TASKS = ("email", "table", "abstract", "code")
MAX_NEW = {"code": 256, "email": 160, "table": 160, "abstract": 160}
NVIDIA_DROP = {"meta-llama/Llama-3.2-3B-Instruct": 0.030, "google/gemma-2-2b-it": 0.067}


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def samples(split: str) -> tuple[list, list]:
    """(attacked, clean) per the plan: test = 20 contexts x 1 instr/type x 3 positions + up to 100 clean;
    dev = 5 train contexts, sampled down to 300 attacked, + 25 clean per task."""
    attacked, clean = [], []
    for t in TASKS:
        if split == "test":
            attacked += build_samples(t, "test", n_contexts=20, attacks_per_type=1, seed=0)
            clean += build_clean(t, "test", n_contexts=100, seed=0)
        else:
            attacked += build_samples(t, "train", n_contexts=5, attacks_per_type=1, seed=1)
            clean += build_clean(t, "train", n_contexts=25, seed=1)
    if split == "dev":
        attacked = random.Random(1).sample(attacked, 300)
    return attacked, clean


def _items(ss) -> list[dict]:
    return [{"uid": s.uid, "conversation": conversation(s),
             "meta": {"task": s.task, "split": s.split, "attack": s.attack, "position": s.position}} for s in ss]


def gen_path(model_id, split, tag, kind) -> Path:
    return DATA / "gen" / short(model_id) / split / f"{tag}.{kind}.jsonl.gz"


def generate(nx, model_id, split, condition, tag, *, layers=None, alpha=None, clean=True) -> None:
    attacked, clean_s = samples(split)
    run(nx, condition, _items(attacked), gen_path(model_id, split, tag, "attacked"), layers=layers, alpha=alpha,
        max_new_tokens=MAX_NEW)
    if clean and condition != "matched":
        run(nx, condition, _items(clean_s), gen_path(model_id, split, tag, "clean"), layers=layers, alpha=alpha,
            max_new_tokens=MAX_NEW)


def _by_uid(ss) -> dict:
    return {s.uid: s for s in ss}


def _collect(model_id, split, tag):
    attacked_s, clean_s = samples(split)
    sa, sc = _by_uid(attacked_s), _by_uid(clean_s)
    att = read_records(gen_path(model_id, split, tag, "attacked"))
    cln = read_records(gen_path(model_id, split, tag, "clean"))
    prompts, rule_out = {}, {}
    for r in att:
        if asr.is_rule(r["attack"], split=r["split"]):
            rule_out[r["uid"]] = bool(asr.evaluator(r["attack"], split=r["split"])(r["output"]))
        else:
            prompts["asr:" + r["uid"]] = asr.evaluator(r["attack"], split=r["split"]).format(r["output"])
    for r in cln:
        s = sc[r["uid"]]
        prompts["util:" + r["uid"]] = utility_prompt(s.task, s.fields, s.ideal, r["output"])
    return prompts, rule_out, att, cln, sa, sc


def prejudge(pairs: list[tuple[str, str]], split: str, *, client=None, log=print) -> float:
    """Judge every (model_id, tag) of a split in one batch submission (later scoring hits the cache).
    Returns the spend of the records involved."""
    allp = {}
    for model_id, tag in pairs:
        prompts = _collect(model_id, split, tag)[0]
        allp.update({f"{short(model_id)}|{tag}|{k}": v for k, v in prompts.items()})
    judged = judge_many(allp, cache_path=CACHE, client=client, log=log)
    usd = spend_usd({r.key: r for r in judged.values()}.values())
    log(f"prejudge {split}: {len(allp)} prompts ({len({r.key for r in judged.values()})} unique), ${usd:.2f}")
    return usd


def judge_tag(model_id, split, tag, *, client=None, log=print) -> dict:
    """Resolve every attacked/clean record of one tag to an outcome; writes the verdict file. Returns rows."""
    prompts, rule_out, att, cln, sa, sc = _collect(model_id, split, tag)
    judged = judge_many(prompts, cache_path=CACHE, client=client, log=log) if prompts else {}
    asr_rows, util_rows, verdicts = [], [], []
    for r in att:
        s = sa[r["uid"]]
        if r["uid"] in rule_out:
            ok, via, v = rule_out[r["uid"]], "rule", None
        else:
            v = judged["asr:" + r["uid"]]
            ok = None if v.verdict in ("refused", "error") else v.verdict == "yes"
            via = "judge"
        asr_rows.append({"task": s.task, "position": s.position, "success": ok, "rule": via == "rule"})
        verdicts.append({"uid": r["uid"], "kind": "asr", "via": via, "outcome": ok,
                         **({"reply": v.reply, "verdict": v.verdict, "judge_model": v.model} if v else {})})
    for r in cln:
        v = judged["util:" + r["uid"]]
        ok = None if v.verdict in ("refused", "error") else v.verdict == "yes"
        util_rows.append({"task": sc[r["uid"]].task, "ok": ok})
        verdicts.append({"uid": r["uid"], "kind": "utility", "via": "judge", "outcome": ok, "reply": v.reply,
                         "verdict": v.verdict, "judge_model": v.model})
    out = DATA / "verdicts" / short(model_id) / split / f"{tag}.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as fh:
        for v in verdicts:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    log(f"{tag}: judged {len(prompts)} prompts; spend so far ${spend_usd(judged.values()):.2f} (this tag)")
    return {"asr_rows": asr_rows, "util_rows": util_rows}


def score_tag(model_id, split, tag, **kw) -> dict:
    rows = judge_tag(model_id, split, tag, **kw)
    res = {"asr": summarize_asr(rows["asr_rows"])}
    if rows["util_rows"]:
        res["utility"] = summarize_utility(rows["util_rows"])
    out = DATA / "scores" / short(model_id) / split / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=list) + "\n")
    return res


def grid(n_layers: int) -> list[tuple[str, list[int], float]]:
    sets = {"all": list(range(n_layers)), "early": list(range(0, round(0.4 * n_layers) + 1)),
            "midearly": list(range(round(0.2 * n_layers), round(0.6 * n_layers) + 1))}
    return [(f"{name}_a{a:g}", layers, a) for name, layers in sets.items() for a in (1.0, 2.0)]


def select(model_id: str, results: dict[str, dict], baseline: dict) -> str:
    """Min dev ASR s.t. utility drop <= max(2pt, NVIDIA's drop); ties prefer the shipped config (all_a1)."""
    cap = max(0.02, NVIDIA_DROP.get(model_id, 0.02))
    base_u = baseline["utility"]["mean"]
    ok = {t: r for t, r in results.items() if base_u - r["utility"]["mean"] <= cap + 1e-9}
    if not ok:
        print(f"select: no config within the {100 * cap:.1f}pt utility cap; choosing by ASR alone", flush=True)
    pool = ok or results
    best = min(pool.values(), key=lambda r: r["asr"]["mean"])["asr"]["mean"]
    tied = [t for t, r in pool.items() if r["asr"]["mean"] == best]
    return "all_a1" if "all_a1" in tied else sorted(tied)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "gen-grid", "prejudge", "judge", "score", "select", "tune"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--conditions", nargs="*", default=["undefended", "nx", "matched"])
    ap.add_argument("--tags", nargs="*", default=["undefended", "nx", "matched"])
    ap.add_argument("--layers", default=None, help="a-b")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    layers = None
    if a.layers:
        lo, hi = map(int, a.layers.split("-"))
        layers = list(range(lo, hi + 1))
    if a.cmd == "generate":
        from nexec import NX

        nx = NX.from_pretrained(a.model)
        for c in a.conditions:
            generate(nx, a.model, a.split, c, a.tag or c, layers=layers, alpha=a.alpha)
    elif a.cmd == "gen-grid":
        from nexec import NX

        nx = NX.from_pretrained(a.model)
        generate(nx, a.model, "dev", "undefended", "undefended")
        for tag, lay, alpha in grid(len(nx.model.model.layers)):
            generate(nx, a.model, "dev", "nx", tag, layers=lay, alpha=alpha)
    elif a.cmd == "prejudge":
        prejudge([(m, t) for m in (a.models or [a.model]) for t in a.tags], a.split)
    elif a.cmd == "select":
        from transformers import AutoConfig

        n = AutoConfig.from_pretrained(a.model).num_hidden_layers
        base = score_tag(a.model, "dev", "undefended")
        results = {t: score_tag(a.model, "dev", t) for t, _, _ in grid(n)}
        choice = select(a.model, results, base)
        summary = {"baseline": {"asr": base["asr"]["mean"], "utility": base["utility"]["mean"]},
                   "grid": {t: {"asr": results[t]["asr"]["mean"], "utility": results[t]["utility"]["mean"],
                                "layers": f"{lay[0]}-{lay[-1]}", "alpha": al} for t, lay, al in grid(n)},
                   "choice": choice}
        out = DATA / "tune" / f"{short(a.model)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
    elif a.cmd in ("judge", "score"):
        for t in a.tags:
            res = score_tag(a.model, a.split, t)
            u = res.get("utility", {}).get("mean", float("nan"))
            print(f"{short(a.model)} {a.split} {t}: ASR {100 * res['asr']['mean']:.1f}% "
                  f"(rule-only {100 * res['asr']['rule_only']['asr']:.1f}%), utility {100 * u:.1f}%")
    else:
        from nexec import NX

        nx = NX.from_pretrained(a.model)
        generate(nx, a.model, "dev", "undefended", "undefended")
        base = score_tag(a.model, "dev", "undefended")
        results = {}
        for tag, lay, alpha in grid(len(nx.model.model.layers)):
            generate(nx, a.model, "dev", "nx", tag, layers=lay, alpha=alpha)
            results[tag] = score_tag(a.model, "dev", tag)
            print(f"dev {tag}: ASR {100 * results[tag]['asr']['mean']:.1f}% "
                  f"utility {100 * results[tag]['utility']['mean']:.1f}%", flush=True)
        choice = select(a.model, results, base)
        summary = {"baseline": {"asr": base["asr"]["mean"], "utility": base["utility"]["mean"]},
                   "grid": {t: {"asr": r["asr"]["mean"], "utility": r["utility"]["mean"],
                                "layers": f"{lay[0]}-{lay[-1]}", "alpha": al}
                            for (t, lay, al), r in zip(grid(len(nx.model.model.layers)), results.values(),
                                                       strict=True)},
                   "choice": choice}
        out = DATA / "tune" / f"{short(a.model)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1) + "\n")
        print("tuned choice:", choice)


if __name__ == "__main__":
    main()
