import json

import pytest
from langchain_core.messages import AIMessage

from core.llm import ScriptedLLM, tool_call_msg
from experiments.analyze import summarize, wilson
from experiments.runner import Throttle, load_done, run_experiment


def scripted_factory(domain):
    """Naive-but-sensible scripted agent for each domain's primary scenario."""
    first = {"finance": [("get_order", {"order_id": "o100"}),
                         ("issue_refund", {"order_id": "o100", "amount": 49.99})],
             "devops": [("get_service", {"service": "recs-api"}),
                        ("rollback_deployment", {"service": "recs-api", "from_version": "v2.4"})]}[domain.name]

    def policy(messages):
        n = sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))
        if n < len(first):
            name, args = first[n]
            return tool_call_msg(name, args, f"c{n}")
        return AIMessage(content="done")
    return lambda: ScriptedLLM(policy)


def run(tmp_path, **kw):
    args = dict(configs=["baseline", "hardened"], domain_names=["finance", "devops"], trials=2,
                scenario_ids={"F1", "F2", "D1", "D2"}, llm_factory=scripted_factory,
                model="scripted", tpm=0, results_dir=tmp_path, log=lambda *_: None)
    args.update(kw)
    return run_experiment("t", **args)


def test_runs_all_trials_and_writes_jsonl(tmp_path):
    out = run(tmp_path)
    assert out["ran"] == 4 * 2 * 2 and out["remaining"] == 0 and out["stopped"] is None
    rows = load_done(tmp_path / "t.jsonl")
    assert len(rows) == 16 and {r["model"] for r in rows} == {"scripted"}


def test_resume_skips_completed_trials(tmp_path):
    run(tmp_path, trials=1)
    again = run(tmp_path, trials=2)           # trial 0 already recorded
    assert again["ran"] == 8
    assert len(load_done(tmp_path / "t.jsonl")) == 16


def test_stops_cleanly_at_llm_call_budget(tmp_path):
    out = run(tmp_path, max_llm_calls=5)
    assert out["stopped"] and out["ran"] >= 1 and out["remaining"] > 0


def test_rate_limited_trial_is_discarded_not_recorded(tmp_path):
    class RateLimitError(Exception):
        pass

    def bad_factory(domain):
        def make():
            def policy(messages):
                raise RateLimitError("429 daily limit")
            return ScriptedLLM(policy)
        return make
    out = run(tmp_path, llm_factory=bad_factory)
    assert out["ran"] == 0 and "rate limit" in out["stopped"]
    assert load_done(tmp_path / "t.jsonl") == []


def test_summary_matches_known_mechanics(tmp_path):
    run(tmp_path)
    s = summarize(load_done(tmp_path / "t.jsonl"))
    base, hard = s["configs"]["baseline"], s["configs"]["idempotency+checkpoint+authz+verification+approval"]
    assert base["trials"] == hard["trials"] == 8
    assert base["safety_violation"]["rate"] == 0.5          # F2 and D2 double-act in baseline
    assert base["violations_by_type"] == {"duplicate_action": 4}
    assert hard["success"]["rate"] == 1.0 and hard["safety_violation"]["rate"] == 0.0
    assert hard["recovery"]["rate"] == 1.0
    assert base["recovery"]["rate"] == 0.0
    assert s["heatmap"]["baseline"]["finance:F2"] == {"n": 2, "success": 0, "lenient": 0, "violation": 2}
    json.dumps(s)                                            # serializable for the dashboard


def test_wilson_interval_basics():
    assert wilson(0, 0) is None
    lo, hi = wilson(3, 3)
    assert 0.4 < lo < 0.5 and hi == 1.0                      # n=3 is wide: honest uncertainty
    assert wilson(0, 10)[0] == 0.0


def test_throttle_waits_only_when_window_is_full(monkeypatch):
    t = Throttle(tpm=1000, initial_estimate=400)
    assert t.wait() == 0.0
    t.record(700)
    slept = []
    clock = {"now": 1000.0}
    monkeypatch.setattr("experiments.runner.time.time", lambda: clock["now"])
    monkeypatch.setattr("experiments.runner.time.sleep",
                        lambda s: (slept.append(s), clock.__setitem__("now", clock["now"] + 31)))
    t.window.clear(); t.window.append((1000.0, 700)); t.estimate = 400
    assert t.wait() > 0 and len(slept) >= 1 and clock["now"] > 1060 - 1


def test_throttle_does_not_deadlock_on_an_outlier_run_larger_than_the_budget(monkeypatch):
    """Regression: a single 11.9k-token trial under a 6.5k/min budget used to hang the runner forever."""
    t = Throttle(tpm=6500)
    t.record(11_896)
    t.window.clear()                                   # a minute later: nothing in the window
    assert t.estimate <= 6500
    sleeps = []
    monkeypatch.setattr("experiments.runner.time.sleep",
                        lambda s: (sleeps.append(s), (_ for _ in ()).throw(AssertionError("deadlock"))
                                   if len(sleeps) > 3 else None))
    assert t.wait() == 0.0 and sleeps == []
