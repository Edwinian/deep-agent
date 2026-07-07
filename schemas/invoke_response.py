"""Typed schemas for POST /invoke responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias

from langchain.agents.middleware.human_in_the_loop import ActionRequest
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


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


class ReviewConfig(BaseModel):
    """Review options for an interrupted tool action."""

    action_name: str
    allowed_decisions: list[InterruptDecision]


class InterruptValue(BaseModel):
    """Human-in-the-loop interrupt payload."""

    action_requests: list[dict[str, Any]]
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
    messages: list[SerializedMessage]
    action_requests: list[ActionRequest] | None = None


class StreamMode(StrEnum):
    """LangGraph stream mode for POST /stream chunks."""

    UPDATES = "updates"
    VALUES = "values"


class SerializedInterruptPayload(TypedDict):
    """Serialized LangGraph interrupt in a stream values snapshot."""

    value: dict[str, Any]
    id: str


class StreamSerializedMessageDict(TypedDict):
    """One LangChain message dict produced by ``messages_to_dict``."""

    type: str
    data: dict[str, Any]


class StreamNodePayload(TypedDict, total=False):
    """Per-node payload for ``stream_mode='updates'`` (e.g. model/tools nodes)."""

    messages: list[StreamSerializedMessageDict]


class StreamValuesEvent(TypedDict, total=False):
    """Full graph state snapshot emitted when ``stream_mode='values'``.

    ``messages`` holds the conversation; the last ``type='ai'`` entry's
    ``data.content`` is the user-facing agent reply. ``files`` is the virtual
    filesystem (search results, etc.). ``__interrupt__`` appears when HITL
    pauses the run.
    """

    messages: list[StreamSerializedMessageDict]
    files: dict[str, dict[str, Any]]
    __interrupt__: list[SerializedInterruptPayload]


StreamUpdateEvent: TypeAlias = dict[str, StreamNodePayload | None]
"""Incremental updates when ``stream_mode='updates'``.

Known keys include ``model`` and ``tools`` (each a :class:`StreamNodePayload`).
Middleware steps appear as ``{MiddlewareClass}.{hook}`` with value ``null``
or a nested payload.
"""

StreamEvent: TypeAlias = StreamValuesEvent | StreamUpdateEvent
"""``event`` field on a :class:`StreamChunk`."""

StreamSerializedScalar: TypeAlias = str | int | float | bool | None
StreamSerializableInput: TypeAlias = (
    StreamSerializedScalar
    | dict[str, Any]
    | list[Any]
    | tuple[Any, ...]
)
"""Nested payloads from LangGraph ``astream`` before JSON serialization.

``BaseMessage`` and ``LangGraphInterrupt`` instances are also accepted at
runtime; they are handled before the structural branches below.
"""

StreamSerializedValue: TypeAlias = (
    StreamSerializedScalar
    | SerializedInterruptPayload
    | StreamSerializedMessageDict
    | dict[str, Any]
    | list[Any]
)


class StreamChunk(BaseModel):
    """One NDJSON line emitted by POST /stream."""

    thread_id: str
    agent_id: int
    graph: list[str]
    stream_mode: StreamMode
    event: StreamValuesEvent | StreamUpdateEvent
