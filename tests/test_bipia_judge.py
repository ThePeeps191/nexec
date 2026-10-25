from types import SimpleNamespace as NS

from evals.bipia.judge import JUDGE_MODEL, judge_many, parse_verdict, utility_prompt


def msg(text, stop="end_turn"):
    return NS(content=[NS(type="text", text=text)], stop_reason=stop, model=JUDGE_MODEL,
              usage=NS(input_tokens=50, output_tokens=2))


class FakeClient:
    """Batch API stand-in: replies by script; results come back in reverse order."""

    def __init__(self, batch_replies, retry_replies=None):
        self.batch_replies, self.retry_replies = batch_replies, retry_replies or {}
        self.created, self.single_calls = [], []
        outer = self

        class Batches:
            def create(self, requests):
                outer.created.append(requests)
                return NS(id=f"b{len(outer.created)}")

            def retrieve(self, bid):
                return NS(processing_status="ended", request_counts=NS(processing=0))

            def results(self, bid):
                reqs = outer.created[int(bid[1:]) - 1]
                out = []
                for r in reqs:
                    kind, val = outer.batch_replies[r["params"]["messages"][0]["content"]]
                    ok = kind == "ok"
                    res = NS(type="succeeded", message=val) if ok else NS(type=kind, error=NS(type="api_error"))
                    out.append(NS(custom_id=r["custom_id"], result=res))
                return list(reversed(out))

        class Messages:
            batches = Batches()

            def create(self, **params):
                prompt = params["messages"][0]["content"]
                outer.single_calls.append(prompt)
                return outer.retry_replies[prompt]

        self.messages = Messages()


def test_out_of_order_results_matched_by_id(tmp_path):
    c = FakeClient({"p1": ("ok", msg("YES")), "p2": ("ok", msg("NO"))})
    out = judge_many({"a": "p1", "b": "p2"}, cache_path=tmp_path / "c.jsonl", client=c, poll_s=0)
    assert out["a"].verdict == "yes" and out["b"].verdict == "no"


def test_refusal_retried_then_success(tmp_path):
    c = FakeClient({"p1": ("ok", msg("", stop="refusal"))}, {"p1": msg("YES")})
    out = judge_many({"a": "p1"}, cache_path=tmp_path / "c.jsonl", client=c, poll_s=0)
    assert out["a"].verdict == "yes" and c.single_calls == ["p1"]


def test_refusal_twice_marked_refused(tmp_path):
    c = FakeClient({"p1": ("ok", msg("", stop="refusal"))}, {"p1": msg("", stop="refusal")})
    out = judge_many({"a": "p1"}, cache_path=tmp_path / "c.jsonl", client=c, poll_s=0)
    assert out["a"].verdict == "refused"


def test_errored_result_retried(tmp_path):
    c = FakeClient({"p1": ("errored", None)}, {"p1": msg("NO")})
    out = judge_many({"a": "p1"}, cache_path=tmp_path / "c.jsonl", client=c, poll_s=0)
    assert out["a"].verdict == "no"


def test_cache_hits_skip_api(tmp_path):
    c = FakeClient({"p1": ("ok", msg("YES"))})
    judge_many({"a": "p1"}, cache_path=tmp_path / "c.jsonl", client=c, poll_s=0)
    c2 = FakeClient({})
    out = judge_many({"x": "p1"}, cache_path=tmp_path / "c.jsonl", client=c2, poll_s=0)
    assert out["x"].verdict == "yes" and c2.created == []


def test_identical_prompts_sent_once(tmp_path):
    c = FakeClient({"p1": ("ok", msg("YES"))})
    judge_many({"a": "p1", "b": "p1"}, cache_path=tmp_path / "c.jsonl", client=c, poll_s=0)
    assert len(c.created) == 1 and len(c.created[0]) == 1


def test_parse_verdict():
    assert parse_verdict("YES") == "yes" and parse_verdict("No.") == "no"
    assert parse_verdict("UNKNOWN") == "unknown" and parse_verdict("yes, it does") == "yes"


def test_utility_prompt_email_unknown():
    p = utility_prompt("email", {"question": "Q: when?"}, "unkown", "It is not in the email.")
    assert "does not contain" in p and "Q: when?" in p and "It is not in the email." in p
    assert p.rstrip().endswith("Reply YES or NO.")
