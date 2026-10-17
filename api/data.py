"""Read-only access to recorded experiment results.

Everything the dashboard needs lives in results/<name>.jsonl (+ .traces.json), so the app can be
deployed on a free static-ish host with no database. Postgres is only used for live runs.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from experiments.analyze import ablation_matrix, summarize
from experiments.runner import RESULTS_DIR, load_done

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def safe_name(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError("invalid dataset name")
    return name


def datasets() -> list[dict]:
    out = []
    for p in sorted(RESULTS_DIR.glob("*.jsonl")):
        rows = load_done(p)
        if rows:
            out.append({"name": p.stem, "trials": len(rows),
                        "configs": sorted({r["config"] for r in rows}),
                        "models": sorted({r["model"] for r in rows})})
    return out


def records(name: str) -> list[dict]:
    return load_done(RESULTS_DIR / f"{safe_name(name)}.jsonl")


def summary(name: str) -> dict:
    return summarize(records(name))


def ablation(main: str = "main", ablation_name: str = "ablation") -> dict:
    return ablation_matrix(records(main), records(ablation_name))


@lru_cache(maxsize=4)
def _traces(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def trace(name: str, run_id: str) -> list[dict] | None:
    path = RESULTS_DIR / f"{safe_name(name)}.traces.json"
    if not path.exists():
        return None
    return _traces(str(path), path.stat().st_mtime).get(run_id)
