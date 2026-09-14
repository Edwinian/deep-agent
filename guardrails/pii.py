"""PII detection and redaction middleware."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing_extensions import override

REDACTED_PII_REFUSAL = (
    "I cannot access or disclose that information because it was redacted "
    "for privacy protection."
)
_REDACTED_TOKEN = re.compile(r"\[REDACTED_[A-Z0-9_]+\]")
_REDACTED_ONLY_RESPONSE = re.compile(r"^\[REDACTED_[A-Z0-9_]+\]$")

# Built-in PII types from LangChain guardrails, excluding url.
# https://docs.langchain.com/oss/python/langchain/guardrails#built-in-pii-types-and-configuration
PII_MIDDLEWARE: list[AgentMiddleware] = [
    PIIMiddleware(
        pii_type,
        apply_to_input=True,
        apply_to_output=True,
        apply_to_tool_results=True,
    )
    for pii_type in ("email", "credit_card", "ip", "mac_address")
]


class RedactedPIIResponseMiddleware(AgentMiddleware):
    """Block assistant replies that treat [REDACTED_*] tokens as real data."""

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

        original_content = str(last_ai_msg.content or "").strip()
        content = original_content
        additional_kwargs = dict(last_ai_msg.additional_kwargs or {})
        additional_kwargs.pop("reasoning_content", None)

        if _REDACTED_ONLY_RESPONSE.fullmatch(content) or _REDACTED_TOKEN.search(content):
            content = REDACTED_PII_REFUSAL

        if (
            content == original_content
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
