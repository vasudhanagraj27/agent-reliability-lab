"""The contract between the reliability framework and a concrete enterprise domain.

The framework (agent harness, fault injector, experiment runner, dashboard) knows nothing
about refunds or deployments. A domain plugs in by implementing `Domain`:

    tools        what the agent can call, and which calls mutate state
    authorize    deterministic policy check for a proposed mutating call
    verify       how to confirm, via a read tool, that a call really took effect
    scenarios    realistic situations with injected faults and ground truth
    grade        inspect the real database and label the run's outcome

Adding a new domain (IAM, data engineering, ...) means writing one subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema for the arguments the LLM supplies
    mutating: bool = False              # True if the call changes external state
    idempotency_param: str | None = None  # name of the arg the harness fills with a key


@dataclass(frozen=True)
class Actor:
    """Who the agent is acting on behalf of."""
    id: str
    role: str = "customer"


@dataclass(frozen=True)
class FaultSpec:
    """One injected fault. `kind` is interpreted by core.faults.FaultInjector."""
    kind: str                           # timeout_after_write | timeout_before_write | stale_read
                                        # | deny | crash_after_node | tool_output_injection
    tool: str | None = None
    times: int = 1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    id: str
    domain: str
    title: str
    request: str                        # what the user / alert says
    actor: Actor
    faults: tuple[FaultSpec, ...] = ()
    legitimate: bool = True             # should the agent complete the task? (false-refusal metric)
    concurrent: int = 1                 # number of simultaneous identical requests
    repeat_request: bool = False        # user sends the same request again after the first finishes
    hide_policy: bool = False           # omit the approval rule from the prompt: only the harness enforces it
    tags: tuple[str, ...] = ()
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyResult:
    status: Literal["applied", "not_applied", "unknown"]
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthzResult:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""


@dataclass
class Outcome:
    """Ground-truth label for one run, computed from the database, never from the agent's text."""
    task_success: bool                  # the right end state was reached
    safety_violations: list[str] = field(default_factory=list)  # e.g. duplicate_action, unauthorized_action
    task_failure: bool = False          # valid task not completed and not safely escalated
    false_refusal: bool = False         # legitimate request refused / escalated unnecessarily
    escalated: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_safety_violation(self) -> bool:
        return bool(self.safety_violations)


class Domain(ABC):
    name: str
    system_prompt: str

    def prompt_for(self, scenario: "Scenario") -> str:
        """System prompt for a scenario. Domains override this to support `hide_policy`."""
        return self.system_prompt

    @abstractmethod
    def setup(self, engine: Engine) -> None:
        """Create and seed this domain's tables inside the run's isolated schema."""

    @abstractmethod
    def tool_specs(self) -> list[ToolSpec]: ...

    @abstractmethod
    def build_tools(self, engine: Engine, actor: Actor) -> dict[str, Callable[..., dict]]:
        """Return the raw (fault-free) tool implementations, keyed by tool name."""

    @abstractmethod
    def authorize(self, engine: Engine, actor: Actor, tool: str, args: dict) -> AuthzResult:
        """Deterministic policy check for a proposed mutating call."""

    @abstractmethod
    def verification_call(self, tool: str, args: dict) -> tuple[str, dict] | None:
        """Which read tool + args confirms that `tool(args)` took effect (None if not verifiable)."""

    @abstractmethod
    def interpret_verification(self, tool: str, args: dict, read_result: dict) -> VerifyResult: ...

    @abstractmethod
    def scenarios(self) -> list[Scenario]: ...

    @abstractmethod
    def grade(self, engine: Engine, scenario: Scenario, run: dict) -> Outcome:
        """`run` carries agent-side facts that are not in the DB (escalated, final_status, ...)."""
