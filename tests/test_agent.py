"""Harness tests with a scripted LLM: deterministic, free, and they show *why* each mechanism matters."""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from core.agent import BASELINE, HARDENED, HarnessConfig
from core.execution import init_checkpoint_tables, run_scenario
from core.llm import ScriptedLLM, tool_call_msg
from core.tracing import init_trace_schema
from domains.finance.domain import FinanceDomain

D = FinanceDomain()


@pytest.fixture(scope="module", autouse=True)
def _init():
    init_trace_schema()
    init_checkpoint_tables()


def refund_policy(order_id="o100", amount=49.99):
    """A sensible but naive agent: look up the order, refund it, report back."""
    def policy(messages):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        if not calls:
            return tool_call_msg("get_order", {"order_id": order_id}, "c1")
        if len(calls) == 1:
            return tool_call_msg("issue_refund", {"order_id": order_id, "amount": amount}, "c2")
        return AIMessage(content="Done.")
    return policy


def run(sid, cfg):
    s = next(x for x in D.scenarios() if x.id == sid)
    return run_scenario(D, s, cfg, lambda: ScriptedLLM(refund_policy()), experiment="test")


def test_f1_happy_path_passes_for_baseline_and_hardened():
    assert run("F1", BASELINE)["task_success"]
    assert run("F1", HARDENED)["task_success"]


def test_f2_timeout_after_write_baseline_double_refunds():
    r = run("F2", BASELINE)                      # blind retry after a committed write
    assert "duplicate_action" in r["safety_violations"]


def test_f2_idempotency_alone_fixes_it():
    assert run("F2", HarnessConfig(idempotency=True))["task_success"]


def test_f2_verification_alone_fixes_it():
    assert run("F2", HarnessConfig(verification=True))["task_success"]


def test_f4_crash_baseline_duplicates_and_checkpoint_alone_does_not_fix_it():
    # The crash lands between the committed effect and the checkpoint write.
    assert "duplicate_action" in run("F4", BASELINE)["safety_violations"]
    assert "duplicate_action" in run("F4", HarnessConfig(checkpoint=True))["safety_violations"]
    assert run("F4", HarnessConfig(idempotency=True))["task_success"]


def test_f5_concurrency_needs_idempotency_not_verification():
    assert run("F5", HarnessConfig(idempotency=True))["task_success"]


def test_f6_stale_read_fools_verification_but_idempotency_saves_it():
    assert "duplicate_action" in run("F6", HarnessConfig(verification=True))["safety_violations"]
    assert run("F6", HARDENED)["task_success"]


def test_f7_authz_blocks_foreign_order():
    def policy(messages):
        calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        if not calls:
            return tool_call_msg("issue_refund", {"order_id": "o200", "amount": 120.0}, "x1")
        return AIMessage(content="ok")
    s = next(x for x in D.scenarios() if x.id == "F7")
    base = run_scenario(D, s, BASELINE, lambda: ScriptedLLM(policy), experiment="test")
    hard = run_scenario(D, s, HARDENED, lambda: ScriptedLLM(policy), experiment="test")
    assert "unauthorized_action" in base["safety_violations"]
    assert hard["task_success"] and not hard["safety_violations"]


def test_f10_high_value_requires_approval_and_resumes_via_interrupt():
    def policy(messages):
        calls = [c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        if not calls:
            return tool_call_msg("issue_refund", {"order_id": "o102", "amount": 900.0}, "y1")
        return AIMessage(content="ok")
    s = next(x for x in D.scenarios() if x.id == "F10")
    base = run_scenario(D, s, BASELINE, lambda: ScriptedLLM(policy), experiment="test")
    hard = run_scenario(D, s, HARDENED, lambda: ScriptedLLM(policy), experiment="test")
    assert "unapproved_high_value_action" in base["safety_violations"]
    assert hard["task_success"]
    rejected = run_scenario(D, s, HARDENED, lambda: ScriptedLLM(policy), experiment="test",
                            approver=lambda info: False)
    assert rejected["details"]["refunds_by_order"] == {}
