"""Configurable fault injection around a domain's tools.

Faults are applied to the *tool layer*, below the agent, exactly where real infrastructure fails:

    timeout_before_write   raises ToolTimeout before the tool runs (nothing happened)
    timeout_after_write    runs the tool, commits the effect, then raises ToolTimeout
                           (the dangerous case: the caller cannot tell it succeeded)
    stale_read             a read tool returns its previous (pre-write) result
    deny                   the tool raises PermissionDenied
    tool_output_injection  appends attacker-controlled text to a tool's output (prompt injection)
    crash_after_node       raises ProcessCrash after a named graph node (handled by the agent)
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from core.domain import FaultSpec
from core.tracing import Tracer


class ToolTimeout(Exception):
    """The call timed out; the effect on the remote system is unknown to the caller."""


class PermissionDenied(Exception):
    pass


class ProcessCrash(Exception):
    """Simulates the agent process dying mid-run."""


class FaultInjector:
    def __init__(self, faults: tuple[FaultSpec, ...] | list[FaultSpec], tracer: Tracer | None = None,
                 snapshotter: Callable[[str, dict], tuple[str, dict] | None] | None = None,
                 raw_tools: dict[str, Callable[..., dict]] | None = None):
        """`snapshotter(tool, args) -> (read_tool, read_args)` names the read that observes a write
        (the domain's verification_call). With `raw_tools`, it lets stale_read model a lagging
        replica: the read is captured just before the write, so it can later serve pre-write state
        even if the agent never read before writing."""
        self._faults = list(faults)
        self._remaining = {i: f.times for i, f in enumerate(self._faults)}
        self._last_result: dict[str, Any] = {}
        self._tracer = tracer
        self._lock = threading.Lock()
        self._snapshotter, self._raw_tools = snapshotter, raw_tools or {}

    # -- helpers ---------------------------------------------------------------------------
    def _take(self, kind: str, tool: str | None = None) -> FaultSpec | None:
        """Consume one use of a matching fault, if any remain."""
        with self._lock:
            for i, f in enumerate(self._faults):
                if f.kind == kind and (f.tool is None or f.tool == tool) and self._remaining[i] > 0:
                    self._remaining[i] -= 1
                    return f
        return None

    def _log(self, kind: str, tool: str | None, **extra) -> None:
        if self._tracer:
            self._tracer.event("fault_injected", node=tool, fault=kind, **extra)

    def _snapshot_before_write(self, name: str, kwargs: dict) -> None:
        if not self._snapshotter:
            return
        plan = self._snapshotter(name, kwargs)
        if not plan:
            return
        read_tool, read_args = plan
        stale_targets = [f for f in self._faults if f.kind == "stale_read" and f.tool == read_tool]
        if stale_targets and read_tool in self._raw_tools:
            self._last_result[read_tool] = self._raw_tools[read_tool](**read_args)

    # -- tool wrapping ---------------------------------------------------------------------
    def wrap(self, name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
        def wrapped(**kwargs) -> dict:
            if self._take("deny", name):
                self._log("deny", name)
                raise PermissionDenied(f"{name}: access denied")

            if self._take("timeout_before_write", name):
                self._log("timeout_before_write", name)
                raise ToolTimeout(f"{name}: timed out before the request was processed")

            # Only consume the fault once there is a previous result to serve stale.
            if name in self._last_result and self._take("stale_read", name):
                self._log("stale_read", name)
                return self._last_result[name]

            self._snapshot_before_write(name, kwargs)
            result = fn(**kwargs)
            self._last_result[name] = result

            if self._take("timeout_after_write", name):
                self._log("timeout_after_write", name, effect_committed=True)
                raise ToolTimeout(f"{name}: timed out waiting for response")

            injection = self._take("tool_output_injection", name)
            if injection:
                self._log("tool_output_injection", name)
                result = {**result, "note": injection.params.get("text", "")}

            return result

        wrapped.__name__ = name
        return wrapped

    # -- crash points ----------------------------------------------------------------------
    def maybe_crash(self, node: str) -> None:
        with self._lock:
            for i, f in enumerate(self._faults):
                if (f.kind == "crash_after_node" and f.params.get("node") == node
                        and self._remaining[i] > 0):
                    self._remaining[i] -= 1
                    if self._tracer:
                        self._tracer.event("fault_injected", node=node, fault="crash_after_node")
                    raise ProcessCrash(f"process crashed after node '{node}'")
