"""Experiment runner: (domain x scenario x harness-config x trial), resumable and rate-limit aware.

    python -m experiments.runner --name main --configs baseline,hardened --trials 3
    python -m experiments.runner --name ablation --configs baseline,idempotency,checkpoint,\
authz,verification,hardened --trials 1 --tags duplicate_risk,authorization,injection

Each finished trial is appended to results/<name>.jsonl immediately. Re-running the same command
skips trials already recorded, so a crash or a daily-limit stop loses nothing. A trial that fails
because of a provider rate limit is NOT recorded (it would be an infrastructure artifact, not
an agent result), and the run stops cleanly so you can resume later.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

from core.agent import BASELINE, HARDENED, HarnessConfig
from core.config import settings
from core.domain import Domain, Scenario
from core.execution import init_checkpoint_tables, run_scenario
from core.llm import make_llm
from core.tracing import init_trace_schema

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Ablation presets. `hardened` is every mechanism together (approval needs authz to be meaningful).
PRESETS: dict[str, HarnessConfig] = {
    "baseline": BASELINE,
    "idempotency": HarnessConfig(idempotency=True),
    "checkpoint": HarnessConfig(checkpoint=True),
    "authz": HarnessConfig(authz=True),
    "verification": HarnessConfig(verification=True),
    "hardened": HARDENED,
}


def all_domains() -> dict[str, Domain]:
    from domains.devops.domain import DevOpsDomain
    from domains.finance.domain import FinanceDomain
    return {d.name: d for d in (FinanceDomain(), DevOpsDomain())}


class Throttle:
    """Sliding 60 s token window so we stay under the provider's tokens-per-minute limit."""

    def __init__(self, tpm: int, initial_estimate: int = 3000):
        self.tpm, self.window, self.recent = tpm, deque(), deque(maxlen=8)
        self.estimate = initial_estimate

    def _used(self, now: float) -> int:
        while self.window and now - self.window[0][0] > 60:
            self.window.popleft()
        return sum(t for _, t in self.window)

    def wait(self) -> float:
        """Block until a typical run fits in the window. Returns seconds slept."""
        slept = 0.0
        while self.tpm and self._used(time.time()) + self.estimate > self.tpm:
            nap = 2.0
            time.sleep(nap)
            slept += nap
        return slept

    def record(self, tokens: int) -> None:
        self.window.append((time.time(), tokens))
        self.recent.append(tokens)
        # Cap the estimate below the budget. One outlier run larger than the whole per-minute budget
        # would otherwise make `used + estimate > tpm` permanently true and wait() would never return.
        cap = 0.75 * self.tpm if self.tpm else float("inf")
        self.estimate = min(max(self.recent), cap)


def _key(r: dict) -> tuple:
    return (r["domain"], r["scenario_id"], r["config"], r["trial"])


def load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate_limited(record: dict) -> bool:
    return any("RateLimit" in e or "429" in e or "rate_limit" in e.lower() for e in record["errors"])


def plan(domains: dict[str, Domain], names: Iterable[str], configs: list[str], trials: int,
         scenario_ids: set[str] | None, tags: set[str] | None):
    """Yield (domain, scenario, config_name, trial). Trial is the outer loop so partial runs stay balanced."""
    for trial in range(trials):
        for dname in names:
            for s in domains[dname].scenarios():
                if scenario_ids and s.id not in scenario_ids:
                    continue
                if tags and not tags & set(s.tags):
                    continue
                for cname in configs:
                    yield domains[dname], s, cname, trial


def run_experiment(name: str, *, configs: list[str], domain_names: list[str], trials: int,
                   scenario_ids: set[str] | None = None, tags: set[str] | None = None,
                   llm_factory: Callable[[Domain], Callable] | None = None,
                   model: str | None = None, tpm: int | None = None,
                   max_llm_calls: int | None = None, results_dir: Path = RESULTS_DIR,
                   pause_between: float = 0.0, log=print) -> dict:
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{name}.jsonl"
    done = {_key(r) for r in load_done(path)}
    domains = all_domains()
    init_trace_schema()
    init_checkpoint_tables()
    throttle = Throttle(tpm if tpm is not None else 6500)
    factory = llm_factory or (lambda d: (lambda: make_llm(d.tool_specs())))
    model = model or settings.llm_model

    todo = [t for t in plan(domains, domain_names, configs, trials, scenario_ids, tags)
            if (t[0].name, t[1].id, PRESETS[t[2]].name, t[3]) not in done]
    log(f"[{name}] {len(done)} done, {len(todo)} to run, model={model}")
    calls_used, ran, stopped = 0, 0, None

    for i, (domain, scenario, cname, trial) in enumerate(todo, 1):
        if max_llm_calls is not None and calls_used >= max_llm_calls:
            stopped = f"reached --max-llm-calls ({max_llm_calls})"
            break
        throttle.wait()
        rec = run_scenario(domain, scenario, PRESETS[cname], factory(domain),
                           experiment=name, trial=trial, model=model)
        if _rate_limited(rec):
            stopped = "provider rate limit hit; trial discarded, resume later"
            break
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        tokens = rec["usage"].get("input_tokens", 0) + rec["usage"].get("output_tokens", 0)
        throttle.record(tokens)
        calls_used += rec["usage"].get("llm_calls", 0)
        ran += 1
        flag = "OK " if rec["task_success"] else ("VIOL" if rec["safety_violations"] else "FAIL")
        log(f"[{i}/{len(todo)}] {scenario.id:>3} {cname:<12} t{trial} {flag} "
            f"{rec['safety_violations'] or ''} tok={tokens} {rec['latency_s']}s")
        if pause_between:
            time.sleep(pause_between)

    return {"ran": ran, "remaining": len(todo) - ran, "stopped": stopped, "path": str(path),
            "llm_calls": calls_used}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True, help="results/<name>.jsonl")
    ap.add_argument("--configs", default="baseline,hardened")
    ap.add_argument("--domains", default="finance,devops")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--scenarios", default="", help="comma-separated ids, e.g. F2,D1")
    ap.add_argument("--tags", default="", help="comma-separated scenario tags")
    ap.add_argument("--tpm", type=int, default=6500, help="token budget per minute (0 = unthrottled)")
    ap.add_argument("--max-llm-calls", type=int, default=None, help="stop after this many LLM calls")
    args = ap.parse_args()
    unknown = [c for c in args.configs.split(",") if c not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choose from {list(PRESETS)}")
    out = run_experiment(
        args.name, configs=args.configs.split(","), domain_names=args.domains.split(","),
        trials=args.trials,
        scenario_ids=set(filter(None, args.scenarios.split(","))) or None,
        tags=set(filter(None, args.tags.split(","))) or None,
        tpm=args.tpm, max_llm_calls=args.max_llm_calls)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
