"""Typed schemas for POST /invoke responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageType(StrEnum):
    """Serialized LangChain message role."""

    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"


class TodoStatus(StrEnum):
    """Task status for agent TODO items."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class ToolCallStatus(StrEnum):
    """Execution status for tool result messages."""

    SUCCESS = "success"
    ERROR = "error"


class InterruptDecision(StrEnum):
    """Allowed human-in-the-loop decisions for interrupted tool calls."""

    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    RESPOND = "respond"


class ToolCall(BaseModel):
    """Tool call requested by an AI message."""

    name: str
    args: dict[str, Any]
    id: str
    type: str = "tool_call"


class MessageData(BaseModel):
    """Payload for a serialized LangChain message."""

    model_config = ConfigDict(extra="allow")

    content: str
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    type: MessageType
    name: str | None = None
    id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[Any] = Field(default_factory=list)
    usage_metadata: dict[str, Any] | None = None
    tool_call_id: str | None = None
    artifact: Any | None = None
    status: ToolCallStatus | None = None


class SerializedMessage(BaseModel):
    """Message entry produced by ``messages_to_dict``."""

    type: MessageType
    data: MessageData


class TodoItem(BaseModel):
    """Task item from agent state."""

    content: str
    status: TodoStatus


class FileEntry(BaseModel):
    """Virtual filesystem file stored in agent state."""

    content: str
    encoding: str
    created_at: str
    modified_at: str


class ActionRequest(BaseModel):
    """Tool action awaiting human review."""

    name: str
    args: dict[str, Any]
    description: str
    tool_call_id: str | None = None


class ReviewConfig(BaseModel):
    """Review options for an interrupted tool action."""

    action_name: str
    allowed_decisions: list[InterruptDecision]


class InterruptValue(BaseModel):
    """Human-in-the-loop interrupt payload."""

    action_requests: list[ActionRequest]
    review_configs: list[ReviewConfig]


class Interrupt(BaseModel):
    """LangGraph interrupt entry."""

    value: InterruptValue
    id: str


class InvokeResult(BaseModel):
    """Serialized agent state returned from a single invoke turn."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    messages: list[SerializedMessage]
    todos: list[TodoItem] | None = None
    files: dict[str, FileEntry] | None = None
    interrupts: list[Interrupt] | None = Field(default=None, alias="__interrupt__")


class InvokeStatus(StrEnum):
    """High-level invoke outcome for the client."""

    COMPLETED = "completed"
    AWAITING_TOOL_PERMISSION = "awaiting_tool_permission"


class InvokeResponse(BaseModel):
    """Response body for POST /invoke."""

    thread_id: str
    agent_id: int
    status: InvokeStatus
    permission_message: str | None = None
    result: InvokeResult


class StreamMode(StrEnum):
    """LangGraph stream mode for POST /stream chunks."""

    UPDATES = "updates"
    VALUES = "values"


class StreamChunk(BaseModel):
    """One NDJSON line emitted by POST /stream."""

    thread_id: str
    agent_id: int
    graph: list[str]
    stream_mode: StreamMode
    event: dict[str, Any]
