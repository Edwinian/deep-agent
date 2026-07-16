"""Typed schemas for POST /invoke responses."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias

from langchain.agents.middleware.human_in_the_loop import ActionRequest
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from schemas.source import Source


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
    CANCELLED = "cancelled"


class InvokeResponse(BaseModel):
    """Response body for POST /invoke."""

    thread_id: str
    agent_id: int
    status: InvokeStatus
    messages: list[SerializedMessage]
    action_requests: list[ActionRequest] | None = None


class StreamMode(StrEnum):
    """LangGraph stream mode for POST /stream raw chunks."""

    MESSAGES = "messages"
    UPDATES = "updates"
    VALUES = "values"
    TASKS = "tasks"
    LIFECYCLE = "lifecycle"
    CUSTOM = "custom"


class StreamFormat(StrEnum):
    """Response shape for POST /stream."""

    CLIENT = "client"
    RAW = "raw"


class StreamContentSource(StrEnum):
    """Whether a chunk originated from the root agent or a nested subagent."""

    AGENT = "agent"
    SUBAGENT = "subagent"


class StreamMessageKind(StrEnum):
    """Content kind for NDJSON chunks from ``stream_events(version="v3")``."""

    TEXT = "text"
    REASONING = "reasoning"
    MESSAGE_TOOL_CALL_CHUNK = "message_tool_call_chunk"
    MESSAGE_TOOL_CALLS_FINALIZED = "message_tool_calls_finalized"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_FINISHED = "tool_call_finished"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_FINISHED = "subagent_finished"
    MESSAGE_FINISHED = "message_finished"
    RUN_FINISHED = "run_finished"
    INTERRUPT = "interrupt"


class StreamTextChunk(BaseModel):
    """One SSE event payload from ``stream_events(version="v3")``."""

    thread_id: str
    agent_id: int
    kind: StreamMessageKind = StreamMessageKind.TEXT
    source: StreamContentSource = StreamContentSource.AGENT
    subagent_name: str | None = None
    subagent_cause: dict[str, Any] | None = None
    node: str | None = None
    message_id: str | None = None
    delta: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    status: InvokeStatus | None = None
    reply: str | None = None
    tool_call_chunk: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    input: dict[str, Any] | None = None
    output: Any | None = None
    content_type: str | None = None
    sources: list[Source] | None = None
    error: str | None = None
    action_requests: list[dict[str, Any]] | None = None
    interrupt_ids: list[str] | None = None


class StreamContentType(StrEnum):
    """UI-oriented content type for client stream chunks."""

    TEXT = "text"
    TOOL = "tool"
    TOOL_RESULT = "tool_result"
    INTERRUPT = "interrupt"
    STATE = "state"


class StreamToolStatus(StrEnum):
    """Lifecycle status for a streamed tool call."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class StreamTextData(BaseModel):
    """Incremental or complete AI text for client rendering."""

    delta: str
    message_id: str | None = None


class StreamToolData(BaseModel):
    """Tool call requested by the model."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: StreamToolStatus = StreamToolStatus.PENDING


class StreamToolResultData(BaseModel):
    """Result returned after a tool executes."""

    tool_call_id: str
    name: str
    content: str
    status: ToolCallStatus = ToolCallStatus.SUCCESS


class StreamInterruptData(BaseModel):
    """Human-in-the-loop interrupt requiring client approval."""

    action_requests: list[dict[str, Any]]
    interrupt_ids: list[str] = Field(default_factory=list)


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


class StreamStateData(BaseModel):
    """Optional full-state snapshot for debugging or client sync."""

    message_count: int = 0
    has_interrupts: bool = False
    messages: list[StreamSerializedMessageDict] = Field(default_factory=list)


ClientStreamData: TypeAlias = (
    StreamTextData
    | StreamToolData
    | StreamToolResultData
    | StreamInterruptData
    | StreamStateData
)
"""Typed payload union for :class:`ClientStreamChunk`.data."""


class ClientStreamChunk(BaseModel):
    """One UI-oriented NDJSON line emitted by POST /stream?format=client."""

    thread_id: str
    agent_id: int
    graph: list[str]
    content_type: StreamContentType
    data: StreamTextData | StreamToolData | StreamToolResultData | StreamInterruptData | StreamStateData
    stream_protocol_version: int = 1


class StreamChunk(BaseModel):
    """One NDJSON line emitted by POST /stream?format=raw."""

    thread_id: str
    agent_id: int
    graph: list[str]
    stream_mode: StreamMode
    event: StreamValuesEvent | StreamUpdateEvent
