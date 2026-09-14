"""Shared helpers for before_agent input classifiers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def message_text(content: Any) -> str:
    """Flatten message content into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return ""


def latest_user_text(messages: list[Any]) -> str | None:
    """Return the current-turn user text, or None when there is no new human turn."""
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return None
    text = message_text(last.content)
    return text or None


def jump_to_end(content: str) -> dict[str, Any]:
    """Stop the agent and return ``content`` as the assistant reply."""
    return {
        "messages": [AIMessage(content=content)],
        "jump_to": "end",
    }
