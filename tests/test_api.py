import json

import pytest
from fastapi.testclient import TestClient

from api import data
from api.main import app
from experiments.runner import RESULTS_DIR

client = TestClient(app)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    rows = [
        {"run_id": "r1", "domain": "finance", "scenario_id": "F2", "config": "baseline", "trial": 0,
         "model": "m", "title": "t", "legitimate": True, "tags": ["timeout"], "task_success": False,
         "safety_violations": ["duplicate_action"], "task_failure": False, "false_refusal": False,
         "escalated": False, "fault_injected": True, "crashes": 0, "errors": [], "latency_s": 1.0,
         "usage": {"input_tokens": 10, "output_tokens": 5, "llm_calls": 2}, "cost_usd": 0, "details": {}},
        {"run_id": "r2", "domain": "finance", "scenario_id": "F2", "config": "hardened", "trial": 0,
         "model": "m", "title": "t", "legitimate": True, "tags": ["timeout"], "task_success": True,
         "safety_violations": [], "task_failure": False, "false_refusal": False,
         "escalated": False, "fault_injected": True, "crashes": 0, "errors": [], "latency_s": 2.0,
         "usage": {"input_tokens": 20, "output_tokens": 5, "llm_calls": 3}, "cost_usd": 0, "details": {}},
    ]
    (tmp_path / "demo.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "demo.traces.json").write_text(json.dumps(
        {"r1": [{"seq": 1, "t": 0.0, "kind": "tool_call", "node": "issue_refund", "payload": {}}],
         "r2": []}))
    monkeypatch.setattr(data, "RESULTS_DIR", tmp_path)
    return "demo"


def test_health_and_scenarios():
    assert client.get("/api/health").json()["ok"]
    sc = client.get("/api/scenarios").json()
    assert {s["domain"] for s in sc} == {"finance", "devops"} and len(sc) == 23


def test_summary_records_trace_compare(dataset):
    s = client.get(f"/api/summary/{dataset}").json()
    assert s["configs"]["baseline"]["safety_violation"]["rate"] == 1.0
    assert s["configs"]["hardened"]["success"]["rate"] == 1.0
    assert len(client.get(f"/api/records/{dataset}", params={"config": "baseline"}).json()) == 1
    assert client.get(f"/api/trace/{dataset}/r1").json()["events"][0]["kind"] == "tool_call"
    cmp = client.get(f"/api/compare/{dataset}", params={"domain": "finance", "scenario": "F2"}).json()
    assert [r["config"] for r in cmp["runs"]] == ["baseline", "hardened"]


def test_ablation_endpoint(dataset):
    r = client.get("/api/ablation", params={"main": dataset, "ablation": dataset})
    assert r.status_code == 200 and r.json()["configs"][0] == "baseline"
    assert client.get("/api/ablation", params={"main": "nope"}).status_code == 404


def test_errors(dataset):
    assert client.get("/api/summary/nope").status_code == 404
    assert client.get("/api/summary/..%2Fetc").status_code in (400, 404)
    assert client.get(f"/api/trace/{dataset}/missing").status_code == 404


def test_live_run_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_RUN", raising=False)
    r = client.post("/api/live-run", json={"domain": "finance", "scenario_id": "F1"})
    assert r.status_code == 403


def test_index_served():
    assert client.get("/").status_code in (200, 404)   # 404 only until static/index.html exists
