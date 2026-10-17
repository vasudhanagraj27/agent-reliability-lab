"""The LangGraph agent, built from a HarnessConfig.

One graph definition produces every configuration in the ablation. Each reliability mechanism is a
flag that adds a node or changes a route:

    agent -> read_tool -> agent                              (reads: always allowed)
    agent -> gate -> execute -> [verify] -> respond -> agent (writes)
    agent -> escalate -> END

    gate      authz        deterministic policy check; human approval via interrupt()
    execute   idempotency  deterministic key lets the system of record dedupe repeats
    verify    verification read-after-write via a read tool before trusting the outcome
    (runner)  checkpoint   durable LangGraph checkpoints in Postgres so a crash resumes

The LLM decides *what* to do; everything that makes it safe is deterministic code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from core.domain import Actor, Domain
from core.faults import FaultInjector, PermissionDenied, ToolTimeout
from core.tracing import Tracer


@dataclass(frozen=True)
class HarnessConfig:
    idempotency: bool = False
    checkpoint: bool = False
    authz: bool = False
    verification: bool = False
    approval: bool = False
    max_retries: int = 2          # baseline: blind retries on timeout; hardened: verified retries

    @property
    def name(self) -> str:
        on = [k for k in ("idempotency", "checkpoint", "authz", "verification", "approval")
              if getattr(self, k)]
        return "baseline" if not on else "+".join(on)

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE = HarnessConfig()
HARDENED = HarnessConfig(idempotency=True, checkpoint=True, authz=True, verification=True,
                         approval=True)


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    pending: dict | None          # the tool call currently being handled
    outcome: str                  # ok | timeout | error | denied
    result: dict | None
    attempts: int
    verified: str | None          # applied | not_applied | unknown | None
    approved: bool
    escalated: bool
    usage: dict


def idempotency_key(actor: Actor, tool: str, args: dict) -> str:
    """Derived from intent, not from the attempt, so retries/duplicates/restarts collapse."""
    payload = json.dumps({"actor": actor.id, "tool": tool, "args": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class AgentRuntime:
    def __init__(self, domain: Domain, config: HarnessConfig, actor: Actor, llm,
                 tools: dict, injector: FaultInjector, tracer: Tracer, engine,
                 system_prompt: str | None = None):
        self.prompt = system_prompt or domain.system_prompt
        self.domain, self.cfg, self.actor = domain, config, actor
        self.llm, self.tools, self.injector, self.tracer, self.engine = llm, tools, injector, tracer, engine
        self.specs = {s.name: s for s in domain.tool_specs()}

    # ------------------------------------------------------------------ nodes
    def agent(self, state: State) -> dict:
        reply: AIMessage = self.llm.invoke([SystemMessage(self.prompt)]
                                           + state["messages"])
        usage = dict(state.get("usage") or {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
        usage["llm_calls"] = usage.get("llm_calls", 0) + 1
        meta = getattr(reply, "usage_metadata", None) or {}
        usage["input_tokens"] += meta.get("input_tokens", 0)
        usage["output_tokens"] += meta.get("output_tokens", 0)
        calls = reply.tool_calls
        self.tracer.event("llm", node="agent",
                          text=(reply.content if isinstance(reply.content, str) else
                                " ".join(b.get("text", "") for b in reply.content
                                         if isinstance(b, dict) and b.get("type") == "text")),
                          tool_calls=[{"name": c["name"], "args": c["args"]} for c in calls])
        return {"messages": [reply], "usage": usage,
                "pending": calls[0] if calls else None, "attempts": 0, "verified": None}

    def route_after_agent(self, state: State) -> str:
        call = state.get("pending")
        if not call:
            return END
        if call["name"] == "escalate_to_human":
            return "escalate"
        spec = self.specs.get(call["name"])
        if spec is None:
            return "respond_unknown"
        return "gate" if spec.mutating else "read_tool"

    def _tool_messages(self, state: State, content: str) -> list[ToolMessage]:
        """Answer the handled call, and politely skip any extra parallel calls."""
        last: AIMessage = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        msgs = [ToolMessage(content=content, tool_call_id=last.tool_calls[0]["id"])]
        for extra in last.tool_calls[1:]:
            msgs.append(ToolMessage(content="Skipped: call one tool at a time.",
                                    tool_call_id=extra["id"]))
        return msgs

    def respond_unknown(self, state: State) -> dict:
        return {"messages": self._tool_messages(state, json.dumps({"error": "unknown_tool"})),
                "pending": None}

    def read_tool(self, state: State) -> dict:
        call = state["pending"]
        self.tracer.event("tool_call", node=call["name"], args=call["args"])
        try:
            result = self.tools[call["name"]](**call["args"])
        except ToolTimeout as exc:
            result = {"error": "timeout", "detail": str(exc)}
        except PermissionDenied as exc:
            result = {"error": "permission_denied", "detail": str(exc)}
        except TypeError as exc:
            result = {"error": "bad_arguments", "detail": str(exc)}
        self.tracer.event("tool_result", node=call["name"], result=result)
        return {"messages": self._tool_messages(state, json.dumps(result, default=str)),
                "pending": None}

    def gate(self, state: State) -> dict:
        call = state["pending"]
        name, args = call["name"], call["args"]
        if not self.cfg.authz:
            return {"outcome": "ok", "approved": False}
        verdict = self.domain.authorize(self.engine, self.actor, name, args)
        self.tracer.event("authz", node=name, allowed=verdict.allowed,
                          needs_approval=verdict.needs_approval, reason=verdict.reason)
        if not verdict.allowed:
            return {"outcome": "denied",
                    "result": {"error": "authorization_denied", "reason": verdict.reason}}
        if verdict.needs_approval:
            if not self.cfg.approval:
                return {"outcome": "denied", "result": {
                    "error": "approval_required",
                    "reason": "this action needs human approval; escalate instead"}}
            decision = interrupt({"tool": name, "args": args, "reason": verdict.reason})
            self.tracer.event("approval", node=name, approved=bool(decision))
            if not decision:
                return {"outcome": "denied", "result": {"error": "approval_rejected"}}
            return {"outcome": "ok", "approved": True}
        return {"outcome": "ok", "approved": False}

    def route_after_gate(self, state: State) -> str:
        return "respond" if state.get("outcome") == "denied" else "execute"

    def execute(self, state: State) -> dict:
        call = state["pending"]
        name, args = call["name"], dict(call["args"])
        spec = self.specs[name]
        if self.cfg.idempotency and spec.idempotency_param:
            args[spec.idempotency_param] = idempotency_key(self.actor, name, call["args"])

        # Baseline behaves like a typical HTTP client: blind retry on timeout, no checking.
        tries = 1 if self.cfg.verification else 1 + self.cfg.max_retries
        outcome, result = "error", {"error": "unknown"}
        for attempt in range(tries):
            self.tracer.event("tool_call", node=name, args=args, attempt=attempt + 1)
            try:
                result, outcome = self.tools[name](**args), "ok"
            except ToolTimeout as exc:
                result, outcome = {"error": "timeout", "detail": str(exc)}, "timeout"
                self.tracer.event("tool_result", node=name, result=result)
                continue
            except PermissionDenied as exc:
                result, outcome = {"error": "permission_denied", "detail": str(exc)}, "error"
            self.tracer.event("tool_result", node=name, result=result)
            break

        # Process can die after the effect committed but before this node's state is saved.
        self.injector.maybe_crash("execute")
        return {"outcome": outcome, "result": result,
                "attempts": state.get("attempts", 0) + 1,
                "pending": {**call, "args": call["args"], "exec_args": args}}

    def route_after_execute(self, state: State) -> str:
        if self.cfg.verification and state["outcome"] in ("ok", "timeout"):
            return "verify"
        return "respond"

    def verify(self, state: State) -> dict:
        call = state["pending"]
        exec_args = call.get("exec_args", call["args"])
        plan = self.domain.verification_call(call["name"], exec_args)
        if plan is None:
            return {"verified": None}
        read_tool, read_args = plan
        try:
            read_result = self.tools[read_tool](**read_args)
        except (ToolTimeout, PermissionDenied) as exc:
            self.tracer.event("verify", node=call["name"], status="unknown", error=str(exc))
            return {"verified": "unknown"}
        res = self.domain.interpret_verification(call["name"], exec_args, read_result)
        self.tracer.event("verify", node=call["name"], status=res.status, evidence=res.evidence)
        return {"verified": res.status}

    def route_after_verify(self, state: State) -> str:
        status = state.get("verified")
        if status == "not_applied" and state["outcome"] == "timeout":
            # Controlled retry: only because we confirmed nothing happened (as far as we can see).
            if state.get("attempts", 0) <= self.cfg.max_retries:
                return "execute"
            return "escalate_internal"
        if status == "unknown":
            return "escalate_internal"
        return "respond"

    def respond(self, state: State) -> dict:
        outcome, result = state.get("outcome", "error"), state.get("result") or {}
        verified = state.get("verified")
        if outcome == "timeout" and verified == "applied":
            payload = {"status": "confirmed", "note": "request timed out but was verified as applied",
                       "evidence": result}
        elif outcome == "ok" and verified == "not_applied":
            payload = {"status": "unconfirmed", "note": "tool reported success but verification "
                       "found no effect", "result": result}
        else:
            payload = result
        return {"messages": self._tool_messages(state, json.dumps(payload, default=str)),
                "pending": None}

    def escalate(self, state: State) -> dict:
        reason = (state["pending"] or {}).get("args", {}).get("reason", "")
        self.tracer.event("escalate", node="escalate_to_human", reason=reason)
        return {"messages": self._tool_messages(state, "Escalated to a human agent."),
                "escalated": True, "pending": None}

    def escalate_internal(self, state: State) -> dict:
        """Harness-initiated escalation: state could not be reconciled safely."""
        self.tracer.event("escalate", node="harness", reason="could not verify outcome")
        return {"messages": self._tool_messages(state, "Action outcome could not be verified; "
                                                       "escalated to a human."),
                "escalated": True, "pending": None}

    # ------------------------------------------------------------------ graph
    def build(self, checkpointer):
        g = StateGraph(State)
        for name in ("agent", "read_tool", "gate", "execute", "verify", "respond",
                     "respond_unknown", "escalate", "escalate_internal"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self.route_after_agent,
                                ["gate", "read_tool", "escalate", "respond_unknown", END])
        g.add_edge("read_tool", "agent")
        g.add_edge("respond_unknown", "agent")
        g.add_conditional_edges("gate", self.route_after_gate, ["execute", "respond"])
        g.add_conditional_edges("execute", self.route_after_execute, ["verify", "respond"])
        g.add_conditional_edges("verify", self.route_after_verify,
                                ["execute", "respond", "escalate_internal"])
        g.add_edge("respond", "agent")
        g.add_edge("escalate", "agent")
        g.add_edge("escalate_internal", "agent")
        return g.compile(checkpointer=checkpointer)
