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
