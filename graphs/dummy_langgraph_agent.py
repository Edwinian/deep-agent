"""Dummy production-style agent graph built with plain LangGraph.

Goal: provide a CompiledStateGraph that follows the same high-level runtime/event
contract expected by `modules/chats/invoke_service.py` and
`modules/chats/stream_service.py`:

- input: `{"messages": [HumanMessage(...)]}` (optionally other state keys)
- output: `{"messages": [...]}` and, when HITL interrupts, `{"__interrupt__": ...}`
- streaming: emits v3 projections for `messages` and `tool_calls` (tool lifecycle)

This file intentionally implements a *rule-based* "planner" instead of a real LLM
so the graph is self-contained and can run without API keys.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command, interrupt

from utils.tool_messages import text_tool_message


class DummyAgentState(TypedDict, total=False):
    """State shape compatible with invoke/stream services.

    The services assume `messages` always exists and that HITL interrupts are
    surfaced via the reserved `__interrupt__` field.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    files: dict[str, Any]
    todos: list[Any]


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


def _redact_pii(text: str) -> str:
    """Minimal PII redaction for demo purposes (emails only)."""

    if not text:
        return text
    return _EMAIL_RE.sub("[REDACTED_EMAIL]", text)


@tool(parse_docstring=True)
def dummy_sensitive_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """A tool that simulates a sensitive action which may require HITL approval."""
    # For a dummy tool, just echo back a short acknowledgement.
    return Command(
        update={
            "messages": [
                text_tool_message(f"Sensitive tool executed: {query}", tool_call_id)
            ]
        }
    )


def _get_last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content or "")
    return ""


def _repair_tool_call_args(
    tool_call: dict[str, Any],
    *,
    user_text: str,
) -> dict[str, Any]:
    """Repair missing required args before HITL checks.

    This mirrors the intent of the repo's `ToolCallArgsRepairMiddleware`: fix up
    incomplete tool calls so HITL/tool execution doesn't fail.
    """

    repaired = dict(tool_call)
    args = dict(repaired.get("args") or {})
    if repaired.get("name") == dummy_sensitive_tool.name:
        # If the model/planner forgot required args, infer them from user text.
        if not str(args.get("query") or "").strip():
            args["query"] = user_text.strip() or "N/A"
    repaired["args"] = args
    return repaired


def _interruptible_tool_call_names() -> set[str]:
    """Which tool names require HITL approval in this dummy agent."""

    # Demonstrate the interrupt-on mapping concept.
    return {dummy_sensitive_tool.name}


def _planner_node(state: DummyAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Create an AIMessage with tool_calls, optionally interrupt for HITL."""

    messages = list(state.get("messages") or [])
    user_text = _get_last_user_text(messages)
    redacted_user_text = _redact_pii(user_text)

    # Rule-based planning:
    # - if user mentions "sensitive", call dummy_sensitive_tool
    # - otherwise, directly answer
    wants_sensitive = "sensitive" in redacted_user_text.lower()

    if not wants_sensitive:
        return {
            "messages": [
                AIMessage(content="All set. (Dummy LangGraph agent.)")
            ]
        }

    tool_call_id = f"tc-{uuid.uuid4()}"
    planned_tool_calls = [
        {
            "name": dummy_sensitive_tool.name,
            "args": {},  # intentionally incomplete to demonstrate arg repair
            "id": tool_call_id,
            "type": "tool_call",
        }
    ]

    # Repair BEFORE HITL decision.
    repaired_tool_calls = [
        _repair_tool_call_args(call, user_text=redacted_user_text)
        for call in planned_tool_calls
    ]

    # HITL: if interruptible tool calls are present, pause and surface action_requests.
    if any(
        call.get("name") in _interruptible_tool_call_names()
        for call in repaired_tool_calls
    ):
        action_requests = []
        for call in repaired_tool_calls:
            if call.get("name") == dummy_sensitive_tool.name:
                action_requests.append(
                    {
                        "name": dummy_sensitive_tool.name,
                        "args": call.get("args") or {},
                        "description": "Approve execution of sensitive tool.",
                    }
                )

        # First run: raises GraphInterrupt. Second run: returns the resume payload.
        resume_payload = interrupt({"action_requests": action_requests})

        decisions = []
        if isinstance(resume_payload, dict):
            decisions = resume_payload.get("decisions") or []
        elif isinstance(resume_payload, (list, tuple)):
            decisions = list(resume_payload)

        # Apply decisions by position/order (same ordering used by build_resume_command).
        updated_tool_calls: list[dict[str, Any]] = []
        assistant_content: str | None = None
        for idx, tool_call in enumerate(repaired_tool_calls):
            decision = decisions[idx] if idx < len(decisions) else None
            decision_type = None
            if isinstance(decision, dict):
                decision_type = decision.get("type")
            else:
                decision_type = getattr(decision, "type", None)

            decision_type_str = str(decision_type or "").lower()
            if decision_type_str in {"approve", "approved"}:
                updated_tool_calls.append(tool_call)
            elif decision_type_str in {"reject", "rejected"}:
                # Drop the tool call.
                continue
            elif decision_type_str in {"respond"}:
                respond_message = None
                if isinstance(decision, dict):
                    respond_message = decision.get("message")
                else:
                    respond_message = getattr(decision, "message", None)
                assistant_content = str(respond_message or "")
                updated_tool_calls = []
            elif decision is None:
                # If resume payload is missing decisions, default to reject to be safe.
                continue
            else:
                # Unknown decision: safest is reject.
                continue

        ai = AIMessage(
            content=assistant_content or "",
            tool_calls=updated_tool_calls,
        )
        return {"messages": [ai]}

    ai = AIMessage(content="", tool_calls=repaired_tool_calls)
    return {"messages": [ai]}


def _tools_router(state: DummyAgentState) -> Literal["tools", "end"]:
    messages = list(state.get("messages") or [])
    if not messages:
        return "end"

    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        if isinstance(last.tool_calls, list) and last.tool_calls:
            return "tools"
    return "end"


def build_dummy_langgraph_agent(
    *,
    checkpointer: Any,
    tools: list[Any] | None = None,
) -> Any:
    """Build and compile the dummy agent graph.

    Args:
        checkpointer: A LangGraph BaseCheckpointSaver (or compatible) used to
            enable resumable interrupts (HITL).
        tools: Optional tool list; defaults to just `dummy_sensitive_tool`.
    """

    _tools = tools or [dummy_sensitive_tool]
    tool_node = ToolNode(_tools)

    builder = StateGraph(DummyAgentState)
    builder.add_node("plan", _planner_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _tools_router, {"tools": "tools", "end": END})
    builder.add_edge("tools", "plan")

    # Compile with a checkpointer so `interrupt(...)` supports resume.
    return builder.compile(checkpointer=checkpointer)

