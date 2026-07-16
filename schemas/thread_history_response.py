"""Typed schema for GET /get-history/{thread_id}."""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents.middleware.human_in_the_loop import ActionRequest
from pydantic import BaseModel, Field

from schemas.invoke_response import InvokeStatus
from schemas.source import Source


class HistoryToolEvent(BaseModel):
    """Tool call shown in the chat history UI."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["running", "done", "error", "interrupt"] = "done"
    output: str | None = None
    subagent_name: str | None = None


class HistoryChatMessage(BaseModel):
    """One user/assistant/system bubble reconstructed from checkpoint messages."""

    id: str
    role: Literal["user", "assistant", "system"]
    content: str = ""
    reasoning: str | None = None
    tools: list[HistoryToolEvent] | None = None
    sources: list[Source] | None = None


class ThreadHistoryResponse(BaseModel):
    """Checkpoint conversation history for a thread."""

    thread_id: str
    agent_id: int
    status: InvokeStatus
    messages: list[HistoryChatMessage]
    action_requests: list[ActionRequest] | None = None
    interrupt_ids: list[str] | None = None
