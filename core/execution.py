"""Run one (domain, scenario, harness-config) trial end to end and grade it from the database.

Handles the parts of the experiment that live *outside* the graph:
  - building an isolated Postgres schema per run
  - simulating process crashes: with `checkpoint` the same LangGraph thread resumes from its
    Postgres checkpoint; without it the "process" restarts from scratch and loses its memory
  - concurrent identical requests and a repeated request after the first one finishes
  - auto-answering human-approval interrupts (the approver is pluggable)
  - latency, token usage and cost
"""
from __future__ import annotations

import threading
import time
import traceback
from contextlib import contextmanager
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from core.agent import AgentRuntime, HarnessConfig
from core.config import settings
from core.db import new_run_id, run_database
from core.domain import Domain, Scenario
from core.faults import FaultInjector, ProcessCrash
from core.tracing import Tracer

MAX_PROCESS_RESTARTS = 3


def init_checkpoint_tables() -> None:
    """One-time creation of LangGraph's checkpoint tables (call before running experiments)."""
    with PostgresSaver.from_conn_string(settings.psycopg_url) as saver:
        saver.setup()


@contextmanager
def _checkpointer(durable: bool):
    if durable:
        with PostgresSaver.from_conn_string(settings.psycopg_url) as saver:
            yield saver
    else:
        yield MemorySaver()          # in-process only: lost when the "process" crashes


def _cost_usd(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * settings.price_in_per_mtok
            + usage.get("output_tokens", 0) * settings.price_out_per_mtok) / 1_000_000


def run_scenario(domain: Domain, scenario: Scenario, config: HarnessConfig, llm_factory: Callable,
                 *, experiment: str = "adhoc", trial: int = 0,
                 approver: Callable[[dict], bool] = lambda info: True,
                 keep_db: bool = False, model: str = "unspecified") -> dict:
    run_id = new_run_id()
    tracer = Tracer(run_id, experiment)
    tracer.start(domain.name, scenario.id, config.to_dict(), trial)
    t0 = time.time()

    with run_database(run_id, keep=keep_db) as engine:
        domain.setup(engine)
        raw_tools = domain.build_tools(engine, scenario.actor)
        injector = FaultInjector(scenario.faults, tracer, snapshotter=domain.verification_call,
                                 raw_tools=raw_tools)
        tools = {n: injector.wrap(n, fn) for n, fn in raw_tools.items()}

        results: list[dict] = []
        lock = threading.Lock()

        def run_one(index: int, tag: str) -> None:
            thread_id = f"{run_id}-{tag}{index}"
            lg_cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
            init = {"messages": [HumanMessage(scenario.request)]}
            inp, restarts, record = init, 0, {"escalated": False, "approved": False,
                                              "usage": {}, "error": None, "crashes": 0}
            llm = llm_factory()
            while True:
                with _checkpointer(config.checkpoint) as saver:
                    graph = AgentRuntime(domain, config, scenario.actor, llm, tools,
                                         injector, tracer, engine,
                                         system_prompt=domain.prompt_for(scenario)).build(saver)
                    try:
                        out = graph.invoke(inp, lg_cfg)
                        while out and out.get("__interrupt__"):
                            info = out["__interrupt__"][0].value
                            approved = approver(info)
                            tracer.event("human_decision", node=info.get("tool"), approved=approved)
                            out = graph.invoke(Command(resume=approved), lg_cfg)
                        values = graph.get_state(lg_cfg).values
                        record.update(escalated=bool(values.get("escalated")),
                                      approved=bool(values.get("approved")),
                                      usage=values.get("usage", {}))
                        break
                    except ProcessCrash:
                        record["crashes"] += 1
                        restarts += 1
                        tracer.event("process_crash", node="runner", restart=restarts,
                                     recovery="resume_from_checkpoint" if config.checkpoint
                                     else "restart_from_scratch")
                        if restarts > MAX_PROCESS_RESTARTS:
                            record["error"] = "too_many_crashes"
                            break
                        inp = None if config.checkpoint else init
                    except GraphRecursionError:
                        record["error"] = "recursion_limit"
                        tracer.event("error", node="runner", error="recursion_limit")
                        break
                    except Exception as exc:                      # API/network errors, bugs
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        tracer.event("error", node="runner", error=record["error"],
                                     trace=traceback.format_exc(limit=3))
                        break
            with lock:
                results.append(record)

        threads = [threading.Thread(target=run_one, args=(i, "a")) for i in range(scenario.concurrent)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        if scenario.repeat_request:
            run_one(0, "repeat")

        usage = {k: sum(r["usage"].get(k, 0) for r in results)
                 for k in ("input_tokens", "output_tokens", "llm_calls")}
        run_info = {"escalated": any(r["escalated"] for r in results),
                    "approved": any(r["approved"] for r in results)}
        outcome = domain.grade(engine, scenario, run_info)

    latency = time.time() - t0
    errors = [r["error"] for r in results if r["error"]]
    record = {
        "run_id": run_id, "domain": domain.name, "scenario_id": scenario.id,
        "config": config.name, "trial": trial, "model": model,
        "title": scenario.title, "legitimate": scenario.legitimate, "tags": list(scenario.tags),
        "task_success": outcome.task_success,
        "safety_violations": outcome.safety_violations,
        "task_failure": outcome.task_failure or bool(errors) and not outcome.task_success
                        and not outcome.safety_violations,
        "false_refusal": outcome.false_refusal,
        "escalated": outcome.escalated,
        "fault_injected": bool(scenario.faults) or scenario.concurrent > 1 or scenario.repeat_request,
        "crashes": sum(r["crashes"] for r in results),
        "errors": errors,
        "latency_s": round(latency, 3),
        "usage": usage, "cost_usd": round(_cost_usd(usage), 6),
        "details": outcome.details,
    }
    tracer.finish(record)
    return record
