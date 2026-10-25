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
