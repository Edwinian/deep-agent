"""Map LangGraph ``stream_events(version="v3")`` projections to client chunks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.stream.run_stream import AsyncGraphRunStream

from schemas.invoke_response import (
    StreamContentType,
    StreamInterruptData,
    StreamTextData,
    StreamToolData,
    StreamToolResultData,
    StreamToolStatus,
    ToolCallStatus,
)
from utils.hitl import collect_action_requests

STREAM_EVENTS_VERSION = "v3"


class AdaptedStreamEvent:
    """One client-facing stream event produced by the stream_events adapter."""

    def __init__(
        self,
        *,
        content_type: StreamContentType,
        data: (
            StreamTextData
            | StreamToolData
            | StreamToolResultData
            | StreamInterruptData
        ),
    ) -> None:
        self.content_type = content_type
        self.data = data


def graph_from_namespace(namespace: list[str] | tuple[str, ...] | None) -> list[str]:
    """Convert a LangGraph namespace tuple into the graph path used on stream chunks."""
    if not namespace:
        return []
    return list(namespace)


def _content_to_text(content: Any) -> str:
    """Convert LangChain message content (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _interrupt_event(interrupts: list[Any]) -> AdaptedStreamEvent | None:
    """Build a client interrupt event from LangGraph interrupt payloads."""
    if not interrupts:
        return None

    action_requests = collect_action_requests(interrupts)
    if not action_requests:
        return None

    interrupt_ids: list[str] = []
    for interrupt in interrupts:
        if isinstance(interrupt, dict):
            interrupt_id = interrupt.get("id")
        else:
            interrupt_id = getattr(interrupt, "id", None)
        if isinstance(interrupt_id, str):
            interrupt_ids.append(interrupt_id)

    return AdaptedStreamEvent(
        content_type=StreamContentType.INTERRUPT,
        data=StreamInterruptData(
            action_requests=action_requests,
            interrupt_ids=interrupt_ids,
        ),
    )


def _tool_result_from_output(
    *,
    tool_name: str,
    tool_call_id: str,
    output: Any,
) -> StreamToolResultData | None:
    """Extract a tool-result payload from a completed tool-call stream output."""
    if output is None:
        return None

    if isinstance(output, ToolMessage):
        status = ToolCallStatus.ERROR if output.status == "error" else ToolCallStatus.SUCCESS
        return StreamToolResultData(
            tool_call_id=output.tool_call_id or tool_call_id,
            name=output.name or tool_name,
            content=_content_to_text(output.content),
            status=status,
        )

    update = getattr(output, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, ToolMessage):
                    return _tool_result_from_output(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        output=message,
                    )

    if isinstance(output, str):
        return StreamToolResultData(
            tool_call_id=tool_call_id,
            name=tool_name,
            content=output,
            status=ToolCallStatus.SUCCESS,
        )

    return StreamToolResultData(
        tool_call_id=tool_call_id,
        name=tool_name,
        content=_content_to_text(output),
        status=ToolCallStatus.SUCCESS,
    )


async def _adapt_message_stream(message_stream: Any) -> AsyncIterator[AdaptedStreamEvent]:
    """Map one ``run.messages`` stream handle to text events."""
    message_id = getattr(message_stream, "message_id", None)

    async for token in message_stream.text:
        if not token:
            continue
        yield AdaptedStreamEvent(
            content_type=StreamContentType.TEXT,
            data=StreamTextData(
                delta=str(token),
                message_id=message_id if isinstance(message_id, str) else None,
            ),
        )

    async for token in message_stream.reasoning:
        if not token:
            continue
        yield AdaptedStreamEvent(
            content_type=StreamContentType.TEXT,
            data=StreamTextData(
                delta=str(token),
                message_id=message_id if isinstance(message_id, str) else None,
            ),
        )


async def _adapt_tool_call_stream(tool_stream: Any) -> AsyncIterator[AdaptedStreamEvent]:
    """Map one ``run.tool_calls`` stream handle to tool and tool-result events."""
    tool_call_id = str(getattr(tool_stream, "tool_call_id", "") or "")
    tool_name = str(getattr(tool_stream, "tool_name", "") or "unknown")
    tool_input = getattr(tool_stream, "input", None)
    tool_args = tool_input if isinstance(tool_input, dict) else {}

    if tool_call_id and tool_name:
        yield AdaptedStreamEvent(
            content_type=StreamContentType.TOOL,
            data=StreamToolData(
                id=tool_call_id,
                name=tool_name,
                args=tool_args,
                status=StreamToolStatus.PENDING,
            ),
        )

    async for _delta in tool_stream.output_deltas:
        # Tool output is finalized on the stream handle after deltas complete.
        pass

    tool_result = _tool_result_from_output(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        output=getattr(tool_stream, "output", None),
    )
    if tool_result is not None:
        yield AdaptedStreamEvent(
            content_type=StreamContentType.TOOL_RESULT,
            data=tool_result,
        )


async def _async_merge_projections(
    run: AsyncGraphRunStream,
    *,
    feeds: dict[str, Callable[[], AsyncIterator[Any]]],
) -> AsyncIterator[tuple[str, Any]]:
    """Merge multiple v3 projection iterators in completion order via a queue."""
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
    done_count = 0
    feed_count = len(feeds)

    async def _feed(name: str, iterator: AsyncIterator[Any]) -> None:
        nonlocal done_count
        try:
            async for item in iterator:
                await queue.put((name, item))
        finally:
            done_count += 1
            if done_count == feed_count:
                await queue.put(None)

    tasks = [
        asyncio.create_task(_feed(name, feed()))
        for name, feed in feeds.items()
    ]

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def iter_adapted_client_events(
    run: AsyncGraphRunStream,
) -> AsyncIterator[tuple[list[str], AdaptedStreamEvent]]:
    """Yield UI-oriented events from a v3 ``AsyncGraphRunStream``.

    Uses typed projections from ``agent.astream_events(version="v3")``:
    ``messages`` for token streams, ``tool_calls`` for tool lifecycle, and
    ``values`` plus run-level interrupt tracking for HITL pauses.
    """
    seen_interrupt_ids: set[str] = set()

    async def _values_feed() -> AsyncIterator[Any]:
        async for state in run.values:
            yield ("state", state)
            for interrupt in run._interrupts:
                interrupt_id = getattr(interrupt, "id", None)
                if isinstance(interrupt_id, str) and interrupt_id not in seen_interrupt_ids:
                    seen_interrupt_ids.add(interrupt_id)
                    yield ("interrupt", interrupt)

    async for name, item in _async_merge_projections(
        run,
        feeds={
            "messages": run.messages.__aiter__,
            "tool_calls": run.tool_calls.__aiter__,
            "values": _values_feed,
        },
    ):
        if name == "messages":
            graph = graph_from_namespace(getattr(item, "namespace", []))
            async for adapted in _adapt_message_stream(item):
                yield graph, adapted
            continue

        if name == "tool_calls":
            async for adapted in _adapt_tool_call_stream(item):
                yield [], adapted
            continue

        if name != "values":
            continue

        if isinstance(item, tuple) and len(item) == 2:
            kind, payload = item
            if kind == "interrupt":
                interrupt_event = _interrupt_event([payload])
                if interrupt_event is not None:
                    yield [], interrupt_event
                continue
            if kind != "state":
                continue
            state = payload
        else:
            state = item

        if isinstance(state, dict):
            interrupts = state.get("__interrupt__")
            if isinstance(interrupts, list) and interrupts:
                interrupt_event = _interrupt_event(interrupts)
                if interrupt_event is not None:
                    yield [], interrupt_event


def protocol_event_to_raw_payload(event: dict[str, Any]) -> tuple[str, list[str], Any]:
    """Convert one v3 protocol event into raw stream chunk fields."""
    method = str(event.get("method", "event"))
    params = event.get("params")
    if not isinstance(params, dict):
        return method, [], event

    namespace = params.get("namespace")
    graph = graph_from_namespace(namespace if isinstance(namespace, list) else [])
    payload: dict[str, Any] = {
        "data": params.get("data"),
        "timestamp": params.get("timestamp"),
        "interrupts": params.get("interrupts", ()),
    }
    return method, graph, payload
