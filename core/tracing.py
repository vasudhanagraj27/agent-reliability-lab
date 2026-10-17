"""Trace store: every run and every step the agent or harness takes, persisted in Postgres.

The dashboard's side-by-side trace viewer reads straight from these tables.
"""
import json
import threading
import time
from dataclasses import dataclass, field

from sqlalchemy import text

from core.db import admin_engine

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS public.runs (
        run_id       TEXT PRIMARY KEY,
        experiment   TEXT NOT NULL,
        domain       TEXT NOT NULL,
        scenario_id  TEXT NOT NULL,
        config       JSONB NOT NULL,
        trial        INT NOT NULL DEFAULT 0,
        started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at  TIMESTAMPTZ,
        outcome      JSONB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS public.events (
        id        BIGSERIAL PRIMARY KEY,
        run_id    TEXT NOT NULL REFERENCES public.runs(run_id) ON DELETE CASCADE,
        seq       INT NOT NULL,
        ts        DOUBLE PRECISION NOT NULL,
        kind      TEXT NOT NULL,
        node      TEXT,
        payload   JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    "CREATE INDEX IF NOT EXISTS events_run_idx ON public.events (run_id, seq)",
]


def init_trace_schema() -> None:
    with admin_engine().begin() as conn:
        for stmt in _DDL:
            conn.execute(text(stmt))


@dataclass
class Tracer:
    """Records events for one run. Concurrent scenarios share a tracer, so seq is lock-guarded."""

    run_id: str
    experiment: str = "adhoc"
    _seq: int = 0
    _events: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self, domain: str, scenario_id: str, config: dict, trial: int = 0) -> None:
        with admin_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO public.runs (run_id, experiment, domain, scenario_id, config, trial) "
                    "VALUES (:r, :e, :d, :s, CAST(:c AS JSONB), :t)"
                ),
                {"r": self.run_id, "e": self.experiment, "d": domain, "s": scenario_id,
                 "c": json.dumps(config), "t": trial},
            )

    def event(self, kind: str, node: str | None = None, **payload) -> None:
        with self._lock:
            self._seq += 1
            row = {"run_id": self.run_id, "seq": self._seq, "ts": time.time(),
                   "kind": kind, "node": node, "payload": json.dumps(payload, default=str)}
            self._events.append(row)

    def finish(self, outcome: dict) -> None:
        """Flush buffered events and record the graded outcome in one transaction."""
        with admin_engine().begin() as conn:
            if self._events:
                conn.execute(
                    text(
                        "INSERT INTO public.events (run_id, seq, ts, kind, node, payload) "
                        "VALUES (:run_id, :seq, :ts, :kind, :node, CAST(:payload AS JSONB))"
                    ),
                    self._events,
                )
            conn.execute(
                text("UPDATE public.runs SET finished_at = now(), outcome = CAST(:o AS JSONB) "
                     "WHERE run_id = :r"),
                {"o": json.dumps(outcome, default=str), "r": self.run_id},
            )
        self._events = []
