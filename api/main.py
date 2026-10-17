"""FastAPI service for the Agent Reliability Lab dashboard.

    uvicorn api.main:app --reload

Read endpoints serve recorded experiment results, so the public demo needs no database or LLM key.
`POST /api/live-run` runs one scenario against the real model; it is OFF unless ENABLE_LIVE_RUN=1,
and is serialized and rate-limited because the free LLM tier has tight limits.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import data
from experiments.runner import PRESETS, all_domains

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Agent Reliability Lab", version="0.1.0")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "live_run_enabled": os.environ.get("ENABLE_LIVE_RUN") == "1"}


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    return data.datasets()


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return [{"id": s.id, "domain": d.name, "title": s.title, "request": s.request,
             "faults": [f.kind for f in s.faults] + (["concurrent_requests"] if s.concurrent > 1 else []),
             "legitimate": s.legitimate, "tags": list(s.tags)}
            for d in all_domains().values() for s in d.scenarios()]


def _dataset_or_404(name: str) -> list[dict]:
    try:
        rows = data.records(name)
    except ValueError:
        raise HTTPException(400, "invalid dataset name")
    if not rows:
        raise HTTPException(404, f"no results for '{name}'")
    return rows


@app.get("/api/summary/{name}")
def get_summary(name: str) -> dict:
    _dataset_or_404(name)
    return data.summary(name)


@app.get("/api/records/{name}")
def get_records(name: str, domain: str | None = None, scenario: str | None = None,
                config: str | None = None) -> list[dict]:
    rows = _dataset_or_404(name)
    return [r for r in rows
            if (not domain or r["domain"] == domain) and (not scenario or r["scenario_id"] == scenario)
            and (not config or r["config"] == config)]


@app.get("/api/ablation")
def get_ablation(main: str = "main", ablation: str = "ablation") -> dict:
    """Scenario x configuration outcome matrix (single safeguards vs baseline vs fully hardened)."""
    for n in (main, ablation):
        _dataset_or_404(n)
    return data.ablation(main, ablation)


@app.get("/api/trace/{name}/{run_id}")
def get_trace(name: str, run_id: str) -> dict:
    _dataset_or_404(name)
    events = data.trace(name, run_id)
    if events is None:
        raise HTTPException(404, "trace not available (run `analyze` to export traces)")
    return {"run_id": run_id, "events": events}


@app.get("/api/compare/{name}")
def compare(name: str, domain: str, scenario: str, trial: int = Query(0, ge=0)) -> dict:
    """Records and traces for every config on one scenario, for the side-by-side viewer."""
    rows = [r for r in _dataset_or_404(name)
            if r["domain"] == domain and r["scenario_id"] == scenario and r["trial"] == trial]
    if not rows:
        raise HTTPException(404, "no such scenario/trial")
    return {"runs": [{**r, "events": data.trace(name, r["run_id"])} for r in rows]}


# ----------------------------------------------------------------------------- live runs
class LiveRun(BaseModel):
    domain: str
    scenario_id: str
    config: str = "baseline"


_live_lock = threading.Lock()
_live_state = {"last": 0.0, "count": 0}
LIVE_MIN_INTERVAL_S = 15
LIVE_MAX_PER_PROCESS = 100


@app.post("/api/live-run")
def live_run(req: LiveRun) -> dict:
    if os.environ.get("ENABLE_LIVE_RUN") != "1":
        raise HTTPException(403, "live runs are disabled on this deployment")
    domains = all_domains()
    if req.domain not in domains or req.config not in PRESETS:
        raise HTTPException(400, "unknown domain or config")
    domain = domains[req.domain]
    scenario = next((s for s in domain.scenarios() if s.id == req.scenario_id), None)
    if scenario is None:
        raise HTTPException(404, "unknown scenario")
    if not _live_lock.acquire(blocking=False):
        raise HTTPException(429, "another live run is in progress")
    try:
        wait = LIVE_MIN_INTERVAL_S - (time.time() - _live_state["last"])
        if wait > 0 or _live_state["count"] >= LIVE_MAX_PER_PROCESS:
            raise HTTPException(429, f"rate limited; retry in {max(wait, 0):.0f}s"
                                if wait > 0 else "live-run budget exhausted")
        from core.execution import run_scenario
        from core.llm import make_llm
        from core.tracing import init_trace_schema
        init_trace_schema()
        _live_state["last"], _live_state["count"] = time.time(), _live_state["count"] + 1
        return run_scenario(domain, scenario, PRESETS[req.config],
                            lambda: make_llm(domain.tool_specs()), experiment="live")
    finally:
        _live_lock.release()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
