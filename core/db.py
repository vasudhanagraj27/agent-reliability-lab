"""Postgres access.

Each experiment run gets its own schema (`run_<id>`) holding the domain's tables,
so runs are isolated and reproducible while trace data lives in `public`.
"""
import uuid
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from core.config import settings

_admin_engine: Engine | None = None


def admin_engine() -> Engine:
    global _admin_engine
    if _admin_engine is None:
        _admin_engine = create_engine(settings.database_url, pool_size=10, pool_pre_ping=True)
    return _admin_engine


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def schema_for(run_id: str) -> str:
    return f"run_{run_id}"


def create_run_schema(run_id: str) -> Engine:
    """Create an isolated schema and return an engine whose search_path points at it."""
    schema = schema_for(run_id)
    with admin_engine().begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    return create_engine(
        settings.database_url,
        connect_args={"options": f"-csearch_path={schema},public"},
        pool_size=5,
        pool_pre_ping=True,
    )


def drop_run_schema(run_id: str, engine: Engine | None = None) -> None:
    if engine is not None:
        engine.dispose()
    with admin_engine().begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_for(run_id)}" CASCADE'))


@contextmanager
def run_database(run_id: str, keep: bool = False):
    """Context manager: isolated schema for one run, dropped on exit unless keep=True."""
    engine = create_run_schema(run_id)
    try:
        yield engine
    finally:
        if keep:
            engine.dispose()
        else:
            drop_run_schema(run_id, engine)
