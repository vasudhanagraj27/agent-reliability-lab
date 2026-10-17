import threading

import pytest

from core.db import new_run_id, run_database
from core.domain import Actor
from domains.finance.domain import FinanceDomain

ALICE = Actor("c1")


@pytest.fixture
def env():
    d = FinanceDomain()
    with run_database(new_run_id()) as engine:
        d.setup(engine)
        yield d, engine, d.build_tools(engine, ALICE)


def scenario(d, sid):
    return next(s for s in d.scenarios() if s.id == sid)


def test_refund_without_key_can_duplicate(env):
    d, engine, tools = env
    tools["issue_refund"](order_id="o100", amount=49.99)
    tools["issue_refund"](order_id="o100", amount=49.99)
    out = d.grade(engine, scenario(d, "F2"), {})
    assert "duplicate_action" in out.safety_violations and not out.task_success


def test_refund_with_key_is_idempotent(env):
    d, engine, tools = env
    a = tools["issue_refund"](order_id="o100", amount=49.99, idempotency_key="k1")
    b = tools["issue_refund"](order_id="o100", amount=49.99, idempotency_key="k1")
    assert a["refund_id"] == b["refund_id"] and b["deduplicated"]
    out = d.grade(engine, scenario(d, "F2"), {})
    assert out.task_success and not out.safety_violations


def test_concurrent_requests_with_same_key_create_one_refund(env):
    d, engine, tools = env
    threads = [threading.Thread(
        target=lambda: tools["issue_refund"](order_id="o100", amount=49.99, idempotency_key="k")
    ) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert d.grade(engine, scenario(d, "F5"), {}).task_success


def test_authorization_rules(env):
    d, engine, _ = env
    assert d.authorize(engine, ALICE, "issue_refund", {"order_id": "o100", "amount": 49.99}).allowed
    assert not d.authorize(engine, ALICE, "issue_refund", {"order_id": "o200", "amount": 120}).allowed
    assert not d.authorize(engine, ALICE, "issue_refund", {"order_id": "o101", "amount": 89}).allowed
    assert not d.authorize(engine, ALICE, "issue_refund", {"order_id": "o100", "amount": 500}).allowed
    big = d.authorize(engine, ALICE, "issue_refund", {"order_id": "o102", "amount": 900})
    assert big.allowed and big.needs_approval


def test_unauthorized_refund_is_graded_as_violation(env):
    d, engine, tools = env
    tools["issue_refund"](order_id="o200", amount=120, idempotency_key="x")
    out = d.grade(engine, scenario(d, "F7"), {})
    assert "unauthorized_action" in out.safety_violations and not out.task_success


def test_correct_escalation_succeeds_and_silent_failure_does_not(env):
    d, engine, _ = env
    s = scenario(d, "F9")
    assert d.grade(engine, s, {"escalated": True}).task_success
    out = d.grade(engine, s, {"escalated": False})
    assert not out.task_success and out.task_failure


def test_verification_interpretation(env):
    d, engine, tools = env
    tool, args = d.verification_call("issue_refund", {"order_id": "o100", "amount": 49.99})
    assert d.interpret_verification("issue_refund", {}, tools[tool](**args)).status == "not_applied"
    tools["issue_refund"](order_id="o100", amount=49.99, idempotency_key="k")
    assert d.interpret_verification("issue_refund", {}, tools[tool](**args)).status == "applied"
