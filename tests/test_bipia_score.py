import pytest

from evals.bipia.score import summarize_asr, summarize_utility, wilson


def test_wilson_bounds():
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = wilson(25, 50)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 0.01
    assert wilson(0, 0) == (0.0, 1.0)


def rows():
    out = []
    for task, k, n in [("email", 3, 10), ("table", 0, 10), ("code", 5, 10)]:
        for i in range(n):
            pos = "end" if i % 2 else "start"
            out.append({"task": task, "position": pos, "success": i < k, "rule": task == "code"})
    out.append({"task": "email", "position": "end", "success": None, "rule": False})  # refused / error
    return out


def test_asr_denominators_exclude_unscored():
    s = summarize_asr(rows())
    assert s["per_task"]["email"]["n"] == 10 and s["per_task"]["email"]["excluded"] == 1
    assert s["per_task"]["email"]["asr"] == pytest.approx(0.3)
    assert s["mean"] == pytest.approx((0.3 + 0.0 + 0.5) / 3)


def test_rule_only_and_positions():
    s = summarize_asr(rows())
    assert s["rule_only"]["n"] == 10 and s["rule_only"]["asr"] == pytest.approx(0.5)
    assert set(s["per_position"]) == {"end", "start"}


def test_bootstrap_deterministic_and_brackets_mean():
    a, b = summarize_asr(rows()), summarize_asr(rows())
    assert a["mean_ci"] == b["mean_ci"]
    assert a["mean_ci"][0] <= a["mean"] <= a["mean_ci"][1]


def test_utility():
    s = summarize_utility([{"task": "email", "ok": True}, {"task": "email", "ok": False},
                           {"task": "table", "ok": True}, {"task": "table", "ok": None}])
    assert s["per_task"]["email"]["rate"] == pytest.approx(0.5) and s["per_task"]["table"]["excluded"] == 1
    assert s["mean"] == pytest.approx(0.75)


def test_paired_utility_difference():
    from evals.bipia.score import paired_diff

    a = {f"u{i}": i % 2 == 0 for i in range(10)}
    b = {**a, "u1": True, "u3": True, "u0": False, "x": True}
    b["u5"] = None
    d = paired_diff(a, b)
    assert d["n"] == 9 and (d["gained"], d["lost"]) == (2, 1)
    assert d["diff"] == pytest.approx(1 / 9) and d["ci"][0] <= d["diff"] <= d["ci"][1]
    assert paired_diff(a, b) == d


def test_unknown_sensitivity_rows():
    from evals.bipia.analysis import _asr_rows

    vs = [{"uid": "email/test/0001/X-0/end", "kind": "asr", "via": "judge", "outcome": False, "verdict": "unknown"},
          {"uid": "code/test/0002/Y-1/start", "kind": "asr", "via": "rule", "outcome": True},
          {"uid": "email/test/0001", "kind": "utility", "via": "judge", "outcome": True, "verdict": "yes"}]
    assert [r["success"] for r in _asr_rows(vs)] == [False, True]
    assert [r["success"] for r in _asr_rows(vs, unknown_success=True)] == [True, True]
