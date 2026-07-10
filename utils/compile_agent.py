"""Compile a DeepAgent spec into a runnable deep agent graph."""

from __future__ import annotations

import re
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer
from typing_extensions import override

from agents.types import DeepAgent, InterruptOn, ModelConfig
from tools.default_interrupt_on import DEFAULT_INTERRUPT_ON
from utils.daytona_sandbox import filesystem_backend
from utils.get_checkpointer import CheckpointerType, get_checkpointer
from utils.resolve_model import resolve_model

PII_GUARDRAIL_SYSTEM_APPENDIX = """\
Privacy guardrails:
- User messages may contain [REDACTED_*] placeholders where sensitive data was removed.
- Never treat a placeholder as real data or repeat it as an answer.
- If asked to recall, repeat, or confirm redacted information, explain that it was \
redacted for privacy and you cannot access or disclose it."""

REDACTED_PII_REFUSAL = (
    "I cannot access or disclose that information because it was redacted "
    "for privacy protection."
)
_REDACTED_ONLY_RESPONSE = re.compile(r"^\[REDACTED_[A-Z0-9_]+\]$")


class RedactedPIIResponseMiddleware(AgentMiddleware):
    """Replace placeholder-only AI replies and drop internal reasoning metadata."""

    @override
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        messages = state["messages"]
        if not messages:
            return None

        last_ai_idx = None
        last_ai_msg = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_msg = messages[i]
                last_ai_idx = i
                break

        if last_ai_idx is None or last_ai_msg is None:
            return None

        content = str(last_ai_msg.content or "").strip()
        additional_kwargs = dict(last_ai_msg.additional_kwargs or {})
        additional_kwargs.pop("reasoning_content", None)

        if _REDACTED_ONLY_RESPONSE.fullmatch(content):
            content = REDACTED_PII_REFUSAL

        if (
            content == str(last_ai_msg.content or "").strip()
            and additional_kwargs == (last_ai_msg.additional_kwargs or {})
        ):
            return None

        updated_message = AIMessage(
            content=content,
            id=last_ai_msg.id,
            name=last_ai_msg.name,
            tool_calls=last_ai_msg.tool_calls,
            additional_kwargs=additional_kwargs,
        )
        new_messages = list(messages)
        new_messages[last_ai_idx] = updated_message
        return {"messages": new_messages}


# Built-in PII types from LangChain guardrails, excluding url.
# https://docs.langchain.com/oss/python/langchain/guardrails#built-in-pii-types-and-configuration
DEFAULT_PII_MIDDLEWARE = tuple(
    PIIMiddleware(
        pii_type,
        apply_to_input=True,
        apply_to_output=True,
        apply_to_tool_results=True,
    )
    for pii_type in ("email", "credit_card", "ip", "mac_address", "url")
) + (RedactedPIIResponseMiddleware(),)


def compile_agent(
    agent: DeepAgent,
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    checkpointer: Checkpointer | None = None,
    interrupt_on: InterruptOn | None = None,
) -> CompiledStateGraph:
    """Build a deep agent graph from a DeepAgent spec.

    Compiles nested subagents recursively, then calls create_deep_agent.
    Use this for the main orchestrator agent or any DeepAgent with subagents.
    """
    resolved_model = resolve_model(
        agent,
        model=model,
        model_config=model_config,
    )
    subagents = agent.get("subagents")
    if subagents:
        from utils.compile_subagents import compile_subagents

        compiled_subagents = compile_subagents(
            subagents,
            default_model=resolved_model,
            default_model_config=model_config,
        )
    else:
        compiled_subagents = None

    resolved_interrupt_on = (
        interrupt_on
        if interrupt_on is not None
        else agent.get("interrupt_on", DEFAULT_INTERRUPT_ON)
    )

    skill_paths = agent.get("skill_paths")

    return create_deep_agent(
        tools=agent.get("tools"),
        system_prompt=f"{agent['system_prompt']}\n\n{PII_GUARDRAIL_SYSTEM_APPENDIX}",
        subagents=compiled_subagents,
        model=resolved_model,
        middleware=DEFAULT_PII_MIDDLEWARE,
        # Pass a BackendProtocol instance (not a factory) per deepagents 0.7 guidance.
        backend=filesystem_backend(),
        checkpointer=checkpointer or get_checkpointer(CheckpointerType.ASYNC_SQLITE),
        interrupt_on=resolved_interrupt_on,
        skills=skill_paths,
    )
