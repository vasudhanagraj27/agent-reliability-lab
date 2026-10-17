"""LLM construction, plus a scripted stand-in used for deterministic tests (no API calls)."""
from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage

from core.config import settings


def tool_defs(specs, include_escalate: bool = True) -> list[dict]:
    defs = [{"name": s.name, "description": s.description, "input_schema": s.parameters}
            for s in specs]
    if include_escalate:
        defs.append({
            "name": "escalate_to_human",
            "description": "Hand the case to a human. Use when the request is outside policy, "
                           "ambiguous, or you cannot complete it safely.",
            "input_schema": {"type": "object",
                             "properties": {"reason": {"type": "string"}},
                             "required": ["reason"]},
        })
    return defs


def make_llm(specs):
    """Real model with the domain's tools bound. temperature=0 for reproducibility."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        model = ChatGroq(model=settings.llm_model, temperature=0, max_tokens=1024,
                         max_retries=settings.llm_max_retries, timeout=90)   # never hang on a dead socket
        return model.bind_tools(_openai_tools(specs))
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=settings.llm_model, temperature=0, max_tokens=1024)
        return model.bind_tools(tool_defs(specs))
    raise NotImplementedError(f"provider '{settings.llm_provider}' is not wired up")


def _openai_tools(specs) -> list[dict]:
    """Groq speaks the OpenAI tool format."""
    return [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                              "parameters": d["input_schema"]}}
            for d in tool_defs(specs)]


class ScriptedLLM:
    """Deterministic fake: `policy(messages) -> AIMessage`. Lets us test the harness for free."""

    def __init__(self, policy: Callable[[list[BaseMessage]], AIMessage]):
        self.policy = policy
        self.calls = 0

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls += 1
        return self.policy(messages)


def tool_call_msg(name: str, args: dict, call_id: str | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id or f"call_{name}_{id(args)}"}])
