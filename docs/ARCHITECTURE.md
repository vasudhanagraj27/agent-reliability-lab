# Architecture and trade-offs

## Goal
Measure, not assert, how much of an agent's reliability comes from the engineering around the model.
Same model, same tasks, same injected failures; only the harness changes.

## Layers

```
 Scenario (request + faults + ground truth)
        │
        ▼
 LangGraph agent  ◄── HarnessConfig flags: idempotency · checkpoint · authz · verification · approval
        │ tool calls
        ▼
 FaultInjector (wraps tools, below the agent)  ── timeout/stale/deny/inject/crash
        │
        ▼
 Domain tools ──► PostgreSQL (isolated schema per run)
                          │
 Grader reads the database ◄┘   →  results/*.jsonl  →  analysis  →  API  →  dashboard
 Tracer writes every step ─────────────────────────►  Postgres `events` / traces.json
```

### The `Domain` contract (`core/domain.py`)
A domain supplies tools, an authorization policy, a verification mapping (which read confirms which
write), scenarios, and a grader. The framework knows nothing about refunds or deployments. Adding the
DevOps domain required **zero changes to `core/`**.

### One graph, five flags (`core/agent.py`)
| Flag | Where it acts | What it defends against |
|---|---|---|
| `idempotency` | key derived from (actor, tool, args), enforced by a `UNIQUE` constraint in the system of record | duplicate side effects from retries, restarts, concurrent requests |
| `checkpoint` | LangGraph `PostgresSaver`; after a crash the same thread resumes | lost progress after a process crash |
| `authz` | deterministic policy check before any write | unauthorized actions, prompt injection that reaches a tool call |
| `verification` | read-after-write via the domain's read tool; retry only if "not applied"; escalate if unknown | blind retries after an ambiguous timeout |
| `approval` | LangGraph `interrupt()` for high-impact actions | unreviewed high-value changes |

The LLM decides *what* to do. Everything that makes it safe is deterministic code, so the safeguards
cannot be talked out of their job by a prompt.

## The baseline is deliberately realistic, not a strawman
It retries blindly on timeout, as a typical HTTP client does, and returns tool errors to the model.
That retry policy, not the model's reasoning, is what duplicates a committed write. The article
should say so plainly.

## Fault model (`core/faults.py`)
Faults sit in the tool layer, where infrastructure really fails.
`timeout_after_write` is the dangerous one: the effect commits, then the caller sees an error.
`crash_after_node: execute` crashes between the committed effect and the checkpoint write, so a
checkpoint alone cannot save you.
`stale_read` models a lagging replica by snapshotting the read just before the write.

## Grading
Outcomes come from the database, never from the agent's text.
- *Hard safety violation*: duplicate action, unauthorized action, unapproved high-value action.
- *Task failure*: a valid task neither completed nor safely escalated.
- *False refusal*: a legitimate request escalated or refused.
- *Recovery rate*: success among trials where a fault was injected.
- *Tokens per successful task*: cost proxy (free tier, so no dollars).

## Decisions and trade-offs
- **Postgres over SQLite.** Real concurrency tests need real transactions, row locks and unique
  constraints. SQLite would have hidden the race the concurrency scenarios exist to expose.
- **Schema per run.** Isolation and reproducibility without a database per run.
- **Deterministic safeguards over LLM-judged ones.** An LLM verifier adds its own failure modes; a
  `UNIQUE` constraint does not.
- **Recorded results for the public demo.** Free tiers have tight limits; the dashboard serves a
  snapshot so it needs no database, key or spend.
- **Small trial counts.** Free-tier limits cap the experiment. Wilson 95% intervals are reported
  because 3 trials per cell cannot support strong claims.

## Known limitations
- One model, two simulated domains, ~10 scenarios each. Results are about this setup.
- Temperature 0 still leaves some run-to-run variance; trials measure it, but only coarsely.
- Prompt injection coverage is one pattern (instructions inside tool output).
- The "approver" is simulated (auto-approve / auto-reject); no real human in the loop.
- IAM and data-engineering domains are extension points, not implemented.
