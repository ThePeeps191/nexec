import pytest

pipeline = pytest.importorskip("evals.bipia.pipeline")


def res(asr, util):
    return {"asr": {"mean": asr}, "utility": {"mean": util}}


def test_grid_layer_sets():
    g = {t: (lay, a) for t, lay, a in pipeline.grid(28)}
    assert g["all_a1"] == (list(range(28)), 1.0) and g["early_a2"][0] == list(range(0, 12))
    assert g["midearly_a1"][0] == list(range(6, 18)) and len(g) == 6


def test_select_respects_utility_cap_and_prefers_shipped_on_ties():
    base = res(0.3, 0.75)
    llama = "meta-llama/Llama-3.2-3B-Instruct"  # cap = max(2pt, 3.0pt) = 3pt
    results = {"all_a1": res(0.05, 0.74), "early_a2": res(0.01, 0.70), "midearly_a1": res(0.05, 0.75)}
    assert pipeline.select(llama, results, base) == "all_a1"  # early_a2 drops 5pt > cap; tie -> shipped
    results["early_a1"] = res(0.02, 0.73)
    assert pipeline.select(llama, results, base) == "early_a1"


def test_select_cap_tolerates_float_rounding():
    llama = "meta-llama/Llama-3.2-3B-Instruct"
    base = res(0.3, 0.5)  # 0.5 - 0.47 == 0.030000000000000027 in floats
    assert pipeline.select(llama, {"all_a1": res(0.05, 0.5), "early_a1": res(0.01, 0.47)}, base) == "early_a1"


def test_v35_choice_rule():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from select_v35 import choose

    llama = "meta-llama/Llama-3.2-3B-Instruct"

    def c(gate, alpha, obj, util):
        return {"gate": gate, "alpha": alpha, "objective": obj, "dev_utility": util}

    cands = {"raw_a1": c("raw", 1.0, 0.2, 0.47), "affine_a2": c("affine", 2.0, 0.05, 0.45),
             "affine_a4": c("affine", 4.0, 0.01, 0.40)}
    assert choose(llama, 0.47, cands) == "affine_a2"  # a4 drops 7pt > 3pt cap
    cands["raw_a2"] = c("raw", 2.0, 0.05, 0.47)
    cands["affine_a1"] = c("affine", 1.0, 0.05, 0.46)
    assert choose(llama, 0.47, cands) == "affine_a1"  # tie: smaller alpha first
    cands.pop("affine_a1")
    assert choose(llama, 0.47, cands) == "raw_a2"  # tie at alpha 2: shipped gate
    assert choose(llama, 0.47, {"x": c("affine", 2.0, 0.0, 0.30)}) == "shipped"
    # The pre-registered cap is max(2pt, 3.0pt) for both models (not step 3's per-model NVIDIA drop, 6.7pt for
    # Gemma): Gemma's 4pt dev drop at alpha 1 and 2 makes both ineligible, so the rule keeps shipped.
    gemma = {"affine_a1": c("affine", 1.0, 0.014, 0.46), "affine_a2": c("affine", 2.0, 0.010, 0.46)}
    assert choose("google/gemma-2-2b-it", 0.50, gemma) == "shipped"
