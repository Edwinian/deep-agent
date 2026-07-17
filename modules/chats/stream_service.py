"""Service layer for POST /stream."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream.run_stream import AsyncGraphRunStream
from langgraph.types import Command, Interrupt as LangGraphInterrupt

from .invoke_service import InvokeService
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import (
    InvokeStatus,
    StreamContentSource,
    StreamMessageKind,
    StreamTextChunk,
)
from schemas.content_type import CONTENT_TYPE_KEY
from schemas.source import SOURCES_KEY, Source
from mcp_interceptors.mcp_auth import mcp_access_token_context
from utils.content_blocks import extract_user_text, is_non_text_content_block_dump
from utils.daytona_sandbox import sync_skills_for_thread
from utils.hitl import collect_action_requests
from utils.langfuse_tracing import with_langfuse_config

STREAM_EVENTS_VERSION: Literal["v3"] = "v3"

_THINKING_STATUS = "Thinking..."
_DELEGATING_STATUS = "Delegating to {name}…"
_RUNNING_SUBAGENT_STATUS = "Running {name}…"


class StreamService:
    """Stream v3 message, tool-call, and subagent projections to clients."""

    def __init__(self, invoke_service: InvokeService) -> None:
        self._invoke_service = invoke_service
        self._runs_lock = asyncio.Lock()
        self._active_runs: dict[str, AsyncGraphRunStream] = {}
        self._cancel_requested: set[str] = set()

    async def cancel_stream(self, thread_id: str) -> bool:
        """Abort an in-flight /stream run (best effort) by `thread_id`."""
        async with self._runs_lock:
            run = self._active_runs.get(thread_id)
            if run is None:
                return False
            self._cancel_requested.add(thread_id)

        # Abort outside the lock so other tasks can observe state updates.
        try:
            await run.abort()
        except Exception:
            # If abort fails mid-flight, we still treat it as a cancellation request.
            pass
        return True

    async def has_active_stream(self, thread_id: str) -> bool:
        """Return True when a /stream run is in flight for this thread."""
        async with self._runs_lock:
            return thread_id in self._active_runs

    @staticmethod
    def emit_stream_chunk(chunk: StreamTextChunk) -> str:
        """Serialize one SSE ``data`` frame."""
        payload = json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False)
        return f"data: {payload}\n\n"

    @staticmethod
    def emit_system_status(
        *,
        content: str,
        thread_id: str,
        agent_id: int,
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> str:
        """Emit a client-visible status line (thinking, delegation, etc.)."""
        return StreamService.emit_stream_chunk(
            StreamTextChunk(
                thread_id=thread_id,
                agent_id=agent_id,
                kind=StreamMessageKind.SYSTEM,
                source=source,
                subagent_name=subagent_name,
                subagent_cause=subagent_cause,
                node=node,
                message_id=message_id,
                content=content,
            )
        )

    @staticmethod
    def emit_system_status_once(
        seen: set[str],
        key: str,
        *,
        content: str,
        thread_id: str,
        agent_id: int,
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> str | None:
        """Emit a status line at most once per dedupe key within one stream run."""
        if key in seen:
            return None
        seen.add(key)
        return StreamService.emit_system_status(
            content=content,
            thread_id=thread_id,
            agent_id=agent_id,
            source=source,
            subagent_name=subagent_name,
            subagent_cause=subagent_cause,
            node=node,
            message_id=message_id,
        )

    @staticmethod
    def _tool_message_from_output(output: Any) -> ToolMessage | None:
        """Extract a ToolMessage from nested graph tool outputs."""
        if isinstance(output, ToolMessage):
            return output

        update = getattr(output, "update", None)
        if isinstance(update, dict):
            messages = update.get("messages", [])
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, ToolMessage):
                        return message

        if isinstance(output, dict):
            messages = output.get("messages", [])
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, ToolMessage):
                        return message
        return None

    @staticmethod
    def _tool_message_stream_fields(
        tool_message: ToolMessage,
    ) -> tuple[str, str | None, list[Source] | None]:
        """Extract client-facing content, content type, and sources from a ToolMessage."""
        content = tool_message.content
        if not isinstance(content, str):
            content = InvokeService.content_to_text(content)
        content_type = tool_message.additional_kwargs.get(CONTENT_TYPE_KEY)
        if content_type is not None:
            content_type = str(content_type)
        else:
            content_type = None

        sources: list[Source] | None = None
        raw_sources = tool_message.additional_kwargs.get(SOURCES_KEY)
        if isinstance(raw_sources, list) and raw_sources:
            parsed: list[Source] = []
            for item in raw_sources:
                try:
                    if isinstance(item, Source):
                        parsed.append(item)
                    elif isinstance(item, dict):
                        parsed.append(Source.model_validate(item))
                except Exception:
                    continue
            sources = parsed or None
        return content, content_type, sources

    @staticmethod
    def serialize_tool_output(output: Any) -> Any:
        """Convert a tool output payload to a JSON-serializable value."""
        if output is None:
            return None
        if StreamService.extract_interrupts(output):
            return None

        tool_message = StreamService._tool_message_from_output(output)
        if tool_message is not None:
            content, content_type, sources = StreamService._tool_message_stream_fields(
                tool_message
            )
            payload: dict[str, Any] = {"content": content}
            if content_type is not None:
                payload["content_type"] = content_type
            if sources:
                payload["sources"] = [source.model_dump() for source in sources]
            if content_type is not None or sources:
                return payload
            return content

        if isinstance(output, ToolMessage):
            content = output.content
            if isinstance(content, str):
                return content
            return InvokeService.content_to_text(content)
        if isinstance(output, str | int | float | bool):
            return output
        if isinstance(output, dict):
            return output
        if isinstance(output, list):
            return output
        if isinstance(output, tuple):
            if StreamService.extract_interrupts(output):
                return None
            return [StreamService.serialize_tool_output(item) for item in output]
        if isinstance(output, Command):
            command_update = output.update
            if isinstance(command_update, dict):
                serialized = StreamService.serialize_tool_output(command_update)
                if serialized is not None:
                    return serialized
        serialized_repr = str(output)
        if serialized_repr.startswith("(Interrupt(") or serialized_repr.startswith(
            "Command(update="
        ):
            return None
        return serialized_repr

    @staticmethod
    def normalize_tool_call_chunk(chunk: Any) -> dict[str, Any]:
        """Normalize a ``message.tool_calls`` delta to a JSON object."""
        if isinstance(chunk, dict):
            return chunk
        if hasattr(chunk, "model_dump"):
            return chunk.model_dump(mode="json")
        return {"value": str(chunk)}

    @staticmethod
    def normalize_tool_calls(finalized: Any) -> list[dict[str, Any]]:
        """Normalize finalized ``message.tool_calls`` values."""
        if not finalized:
            return []
        if not isinstance(finalized, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in finalized:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "model_dump"):
                normalized.append(item.model_dump(mode="json"))
            else:
                normalized.append({"value": str(item)})
        return normalized

    async def merge_projection_feeds(
        self,
        feeds: dict[str, Callable[[], AsyncIterator[Any]]],
    ) -> AsyncIterator[tuple[str, Any]]:
        """Merge multiple v3 projection iterators in completion order."""
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

    @staticmethod
    def extract_interrupts(value: Any) -> list[Any]:
        """Collect LangGraph interrupt objects from a tool output or state value."""
        if value is None:
            return []

        if isinstance(value, LangGraphInterrupt):
            return [value]

        if isinstance(value, dict) and "id" in value and "value" in value:
            return [value]

        if isinstance(value, (list, tuple)):
            collected: list[Any] = []
            for item in value:
                collected.extend(StreamService.extract_interrupts(item))
            return collected

        return []

    @staticmethod
    def interrupt_ids_from_payloads(interrupts: list[Any]) -> list[str]:
        """Return stable interrupt ids from LangGraph interrupt payloads."""
        interrupt_ids: list[str] = []
        for interrupt in interrupts:
            if isinstance(interrupt, LangGraphInterrupt):
                interrupt_id = interrupt.id
            elif isinstance(interrupt, dict):
                interrupt_id = interrupt.get("id")
            else:
                interrupt_id = getattr(interrupt, "id", None)
            if isinstance(interrupt_id, str):
                interrupt_ids.append(interrupt_id)
        return interrupt_ids

    def build_interrupt_chunk(
        self,
        interrupts: list[Any],
        *,
        thread_id: str,
        agent_id: int,
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
    ) -> StreamTextChunk | None:
        """Build one interrupt chunk from LangGraph interrupt payloads."""
        if not interrupts:
            return None

        action_requests = collect_action_requests(interrupts)
        if not action_requests:
            return None

        return StreamTextChunk(
            thread_id=thread_id,
            agent_id=agent_id,
            kind=StreamMessageKind.INTERRUPT,
            source=source,
            subagent_name=subagent_name,
            subagent_cause=subagent_cause,
            action_requests=action_requests,
            interrupt_ids=self.interrupt_ids_from_payloads(interrupts),
        )

    async def _emit_interrupt_chunks(
        self,
        interrupts: list[Any],
        *,
        thread_id: str,
        agent_id: int,
        seen_interrupt_ids: set[str],
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Emit interrupt chunks once per unseen interrupt id."""
        fresh_interrupts: list[Any] = []
        for interrupt in interrupts:
            if isinstance(interrupt, LangGraphInterrupt):
                interrupt_id = interrupt.id
            elif isinstance(interrupt, dict):
                interrupt_id = interrupt.get("id")
            else:
                interrupt_id = getattr(interrupt, "id", None)
            if not isinstance(interrupt_id, str) or interrupt_id in seen_interrupt_ids:
                continue
            seen_interrupt_ids.add(interrupt_id)
            fresh_interrupts.append(interrupt)

        chunk = self.build_interrupt_chunk(
            fresh_interrupts,
            thread_id=thread_id,
            agent_id=agent_id,
            source=source,
            subagent_name=subagent_name,
            subagent_cause=subagent_cause,
        )
        if chunk is not None:
            yield self.emit_stream_chunk(chunk)

    async def values_interrupt_feed(
        self,
        run: Any,
        seen_interrupt_ids: set[str],
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield state snapshots and newly observed interrupts from ``run.values``."""
        if not hasattr(run, "values"):
            return

        async for state in run.values:
            yield ("state", state)

            for interrupt in getattr(run, "_interrupts", []):
                if isinstance(interrupt, LangGraphInterrupt):
                    interrupt_id = interrupt.id
                elif isinstance(interrupt, dict):
                    interrupt_id = interrupt.get("id")
                else:
                    interrupt_id = getattr(interrupt, "id", None)
                if isinstance(interrupt_id, str) and interrupt_id not in seen_interrupt_ids:
                    yield ("interrupt", interrupt)

            if isinstance(state, dict):
                state_interrupts = state.get("__interrupt__")
                if isinstance(state_interrupts, list):
                    for interrupt in state_interrupts:
                        if isinstance(interrupt, dict):
                            interrupt_id = interrupt.get("id")
                        else:
                            interrupt_id = getattr(interrupt, "id", None)
                        if (
                            isinstance(interrupt_id, str)
                            and interrupt_id not in seen_interrupt_ids
                        ):
                            yield ("interrupt", interrupt)

    def projection_feeds_for_run(
        self,
        run: Any,
        *,
        seen_interrupt_ids: set[str] | None = None,
    ) -> dict[str, Callable[[], AsyncIterator[Any]]]:
        """Build merge feeds for a run or subagent handle."""
        feeds: dict[str, Callable[[], AsyncIterator[Any]]] = {
            "messages": run.messages.__aiter__,
        }
        if hasattr(run, "tool_calls"):
            feeds["tool_calls"] = run.tool_calls.__aiter__
        if seen_interrupt_ids is not None and hasattr(run, "values"):
            feeds["values"] = lambda: self.values_interrupt_feed(run, seen_interrupt_ids)
        return feeds

    @staticmethod
    def resolve_subagent_metadata(
        subagent: Any,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Extract subagent name and triggering cause from a handle."""
        name = getattr(subagent, "name", None)
        if not isinstance(name, str):
            graph_name = getattr(subagent, "graph_name", None)
            name = graph_name if isinstance(graph_name, str) else None

        cause = getattr(subagent, "cause", None)
        resolved_cause = cause if isinstance(cause, dict) else None
        return name, resolved_cause

    async def merge_run_projections(
        self,
        run: AsyncGraphRunStream,
        *,
        seen_interrupt_ids: set[str],
    ) -> AsyncIterator[tuple[str, Any]]:
        """Merge root ``messages``, ``tool_calls``, ``subagents``, and ``values``."""
        feeds = self.projection_feeds_for_run(run, seen_interrupt_ids=seen_interrupt_ids)
        if hasattr(run, "subagents"):
            feeds["subagents"] = run.subagents.__aiter__
        async for item in self.merge_projection_feeds(feeds):
            yield item

    async def yield_interrupt_chunks(
        self,
        item: Any,
        *,
        thread_id: str,
        agent_id: int,
        seen_interrupt_ids: set[str],
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Map one ``run.values`` feed item to interrupt SSE chunks."""
        if isinstance(item, tuple) and len(item) == 2:
            kind, payload = item
            if kind != "interrupt":
                return
            async for line in self._emit_interrupt_chunks(
                [payload],
                thread_id=thread_id,
                agent_id=agent_id,
                seen_interrupt_ids=seen_interrupt_ids,
                source=source,
                subagent_name=subagent_name,
                subagent_cause=subagent_cause,
            ):
                yield line
            return

        async for line in self._emit_interrupt_chunks(
            self.extract_interrupts(item),
            thread_id=thread_id,
            agent_id=agent_id,
            seen_interrupt_ids=seen_interrupt_ids,
            source=source,
            subagent_name=subagent_name,
            subagent_cause=subagent_cause,
        ):
            yield line

    async def yield_message_chunks(
        self,
        message: Any,
        *,
        thread_id: str,
        agent_id: int,
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
        seen_system_status: set[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream one ``run.messages`` handle: text, reasoning, and tool-call chunks.

        ``message.text``, ``message.reasoning``, and ``message.tool_calls`` share one
        graph pump. Iterating them sequentially waits for each projection to finish
        before the next starts, so tokens only reach the client after the full LLM
        turn. Merge the three projections so deltas are yielded as they arrive.
        """
        node = getattr(message, "node", None)
        message_id = getattr(message, "message_id", None)
        resolved_node = node if isinstance(node, str) else None
        resolved_message_id = message_id if isinstance(message_id, str) else None
        common = {
            "thread_id": thread_id,
            "agent_id": agent_id,
            "source": source,
            "subagent_name": subagent_name,
            "subagent_cause": subagent_cause,
            "node": resolved_node,
            "message_id": resolved_message_id,
        }

        # Immediate status so the UI is not blank while the model starts.
        status_key = f"{resolved_message_id or 'message'}:{_THINKING_STATUS}"
        if seen_system_status is not None:
            line = self.emit_system_status_once(
                seen_system_status,
                status_key,
                content=_THINKING_STATUS,
                thread_id=thread_id,
                agent_id=agent_id,
                source=source,
                subagent_name=subagent_name,
                subagent_cause=subagent_cause,
                node=resolved_node,
                message_id=resolved_message_id,
            )
            if line:
                yield line
        else:
            yield self.emit_system_status(
                content=_THINKING_STATUS,
                thread_id=thread_id,
                agent_id=agent_id,
                source=source,
                subagent_name=subagent_name,
                subagent_cause=subagent_cause,
                node=resolved_node,
                message_id=resolved_message_id,
            )

        streamed_reasoning = ""
        streamed_answer_text = ""

        async for name, item in self.merge_projection_feeds(
            {
                "reasoning": message.reasoning.__aiter__,
                "text": message.text.__aiter__,
                "tool_calls": message.tool_calls.__aiter__,
            }
        ):
            if name == "reasoning":
                if not item:
                    continue
                delta_text = str(item)
                streamed_reasoning += delta_text
                yield self.emit_stream_chunk(
                    StreamTextChunk(
                        **common,
                        kind=StreamMessageKind.REASONING,
                        delta=delta_text,
                    )
                )
                continue

            if name == "text":
                if not item:
                    continue
                delta_text = str(item)
                # Grok may emit stringified tool_call blocks on the text channel.
                if is_non_text_content_block_dump(delta_text):
                    continue
                streamed_answer_text += delta_text
                yield self.emit_stream_chunk(
                    StreamTextChunk(
                        **common,
                        kind=StreamMessageKind.TEXT,
                        delta=delta_text,
                    )
                )
                continue

            if name == "tool_calls":
                yield self.emit_stream_chunk(
                    StreamTextChunk(
                        **common,
                        kind=StreamMessageKind.MESSAGE_TOOL_CALL_CHUNK,
                        tool_call_chunk=self.normalize_tool_call_chunk(item),
                    )
                )

        get_finalized = getattr(message.tool_calls, "get", None)
        if get_finalized is not None:
            finalized = get_finalized()
        else:
            finalized = await message.tool_calls

        normalized_tool_calls = self.normalize_tool_calls(finalized)
        if normalized_tool_calls:
            if not streamed_reasoning and not streamed_answer_text:
                tool_names = [
                    str(call.get("name") or "").strip()
                    for call in normalized_tool_calls
                    if isinstance(call, dict)
                ]
                tool_names = [name for name in tool_names if name]
                if tool_names:
                    label = ", ".join(tool_names)
                    planning_content = f"Planning tool use: {label}…"
                    if seen_system_status is not None:
                        planning_key = (
                            f"{resolved_message_id or 'message'}:planning:{label}"
                        )
                        line = self.emit_system_status_once(
                            seen_system_status,
                            planning_key,
                            content=planning_content,
                            thread_id=thread_id,
                            agent_id=agent_id,
                            source=source,
                            subagent_name=subagent_name,
                            subagent_cause=subagent_cause,
                            node=resolved_node,
                            message_id=resolved_message_id,
                        )
                        if line:
                            yield line
                    else:
                        yield self.emit_system_status(
                            content=planning_content,
                            thread_id=thread_id,
                            agent_id=agent_id,
                            source=source,
                            subagent_name=subagent_name,
                            subagent_cause=subagent_cause,
                            node=resolved_node,
                            message_id=resolved_message_id,
                        )
            yield self.emit_stream_chunk(
                StreamTextChunk(
                    **common,
                    kind=StreamMessageKind.MESSAGE_TOOL_CALLS_FINALIZED,
                    tool_calls=normalized_tool_calls,
                )
            )

        full_text_value = await message.text
        full_reasoning_value = await message.reasoning
        full_text_raw = "" if full_text_value is None else str(full_text_value)
        full_reasoning = (
            "" if full_reasoning_value is None else str(full_reasoning_value)
        )
        # Drop stringified tool_call-only dumps; unwrap text from block lists.
        full_text = extract_user_text(full_text_raw)

        if (
            streamed_answer_text
            and full_text
            and streamed_answer_text != full_text
            and full_text.startswith(streamed_answer_text)
        ):
            remainder = full_text[len(streamed_answer_text) :]
            if remainder and not is_non_text_content_block_dump(remainder):
                yield self.emit_stream_chunk(
                    StreamTextChunk(
                        **common,
                        kind=StreamMessageKind.TEXT,
                        delta=remainder,
                    )
                )
                streamed_answer_text = full_text
        elif not streamed_answer_text and full_text:
            yield self.emit_stream_chunk(
                StreamTextChunk(
                    **common,
                    kind=StreamMessageKind.TEXT,
                    delta=full_text,
                )
            )
            streamed_answer_text = full_text

        if len(streamed_reasoning) < len(full_reasoning):
            remainder = full_reasoning[len(streamed_reasoning) :]
            if remainder:
                yield self.emit_stream_chunk(
                    StreamTextChunk(
                        **common,
                        kind=StreamMessageKind.REASONING,
                        delta=remainder,
                    )
                )
                streamed_reasoning = full_reasoning

        if full_text or full_reasoning:
            yield self.emit_stream_chunk(
                StreamTextChunk(
                    **common,
                    kind=StreamMessageKind.MESSAGE_FINISHED,
                    content=full_text or None,
                    reasoning_content=full_reasoning or None,
                )
            )

    @staticmethod
    def collect_sources_from_state(raw_result: dict[str, Any]) -> list[Source]:
        """Collect Source payloads from tool messages and ``/_sources.json``."""
        from tools.web_search.web_search_tool import SOURCES_FILE, _load_sources_file

        by_url: dict[str, Source] = {}

        def _add(items: list[Any]) -> None:
            for item in items:
                try:
                    source = (
                        item
                        if isinstance(item, Source)
                        else Source.model_validate(item)
                    )
                except Exception:
                    continue
                if source.url:
                    by_url[source.url] = source

        files = raw_result.get("files")
        if isinstance(files, dict):
            _add(_load_sources_file(files))
            # Also accept either key form written by older runs.
            if SOURCES_FILE.lstrip("/") in files and SOURCES_FILE not in files:
                _add(_load_sources_file({SOURCES_FILE: files[SOURCES_FILE.lstrip("/")]}))

        messages = raw_result.get("messages") or []
        if isinstance(messages, list):
            for message in messages:
                tool_message = None
                if isinstance(message, ToolMessage):
                    tool_message = message
                elif isinstance(message, dict):
                    data = message.get("data") if message.get("type") == "tool" else None
                    if isinstance(data, dict):
                        raw_sources = (data.get("additional_kwargs") or {}).get(
                            SOURCES_KEY
                        )
                        if isinstance(raw_sources, list):
                            _add(raw_sources)
                        continue
                if tool_message is not None:
                    raw_sources = tool_message.additional_kwargs.get(SOURCES_KEY)
                    if isinstance(raw_sources, list):
                        _add(raw_sources)

        return list(by_url.values())

    @staticmethod
    def merge_sources(
        existing: list[Source] | None,
        incoming: list[Source] | None,
    ) -> list[Source] | None:
        """Dedupe sources by URL."""
        if not incoming:
            return existing or None
        by_url: dict[str, Source] = {
            source.url: source for source in (existing or []) if source.url
        }
        for source in incoming:
            if source.url:
                by_url[source.url] = source
        return list(by_url.values()) or None

    async def yield_run_finished_chunk(
        self,
        *,
        thread_id: str,
        agent_id: int,
        raw_result: dict[str, Any],
        cancelled: bool = False,
        sources: list[Source] | None = None,
    ) -> str:
        """Emit one terminal chunk matching invoke's final reply and status."""
        collected = self.merge_sources(
            sources,
            self.collect_sources_from_state(raw_result),
        )

        if cancelled:
            return self.emit_stream_chunk(
                StreamTextChunk(
                    thread_id=thread_id,
                    agent_id=agent_id,
                    kind=StreamMessageKind.RUN_FINISHED,
                    status=InvokeStatus.CANCELLED,
                    reply=InvokeService.last_ai_reply(raw_result.get("messages", [])),
                    sources=collected,
                )
            )

        if self._invoke_service.has_pending_interrupts(raw_result):
            return self.emit_stream_chunk(
                StreamTextChunk(
                    thread_id=thread_id,
                    agent_id=agent_id,
                    kind=StreamMessageKind.RUN_FINISHED,
                    status=InvokeStatus.AWAITING_TOOL_PERMISSION,
                    # Do not promote prior task output as a "final" reply while
                    # HITL is pending — that makes the UI look done + ask again.
                    reply=InvokeService.last_ai_reply(
                        raw_result.get("messages", []),
                        allow_task_fallback=False,
                    ),
                    action_requests=collect_action_requests(raw_result["__interrupt__"]),
                    interrupt_ids=self.interrupt_ids_from_payloads(
                        raw_result.get("__interrupt__", [])
                    ),
                    sources=collected,
                )
            )

        return self.emit_stream_chunk(
            StreamTextChunk(
                thread_id=thread_id,
                agent_id=agent_id,
                kind=StreamMessageKind.RUN_FINISHED,
                status=InvokeStatus.COMPLETED,
                reply=InvokeService.last_ai_reply(raw_result.get("messages", [])),
                sources=collected,
            )
        )

    async def yield_run_tool_call_chunks(
        self,
        tool_call: Any,
        *,
        thread_id: str,
        agent_id: int,
        seen_interrupt_ids: set[str],
        source: StreamContentSource = StreamContentSource.AGENT,
        subagent_name: str | None = None,
        subagent_cause: dict[str, Any] | None = None,
        collected_sources: list[Source] | None = None,
    ) -> AsyncIterator[str]:
        """Stream one ``run.tool_calls`` handle: start, output deltas, and finish."""
        tool_call_id = str(getattr(tool_call, "tool_call_id", "") or "")
        tool_name = str(getattr(tool_call, "tool_name", "") or "unknown")
        tool_input = getattr(tool_call, "input", None)
        resolved_input = tool_input if isinstance(tool_input, dict) else {}
        common = {
            "thread_id": thread_id,
            "agent_id": agent_id,
            "source": source,
            "subagent_name": subagent_name,
            "subagent_cause": subagent_cause,
        }

        yield self.emit_stream_chunk(
            StreamTextChunk(
                **common,
                kind=StreamMessageKind.TOOL_CALL_STARTED,
                tool_call_id=tool_call_id or None,
                tool_name=tool_name,
                input=resolved_input,
            )
        )

        async for delta in tool_call.output_deltas:
            if delta is None:
                continue
            delta_interrupts = self.extract_interrupts(delta)
            if delta_interrupts:
                async for line in self._emit_interrupt_chunks(
                    delta_interrupts,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    seen_interrupt_ids=seen_interrupt_ids,
                    source=source,
                    subagent_name=subagent_name,
                    subagent_cause=subagent_cause,
                ):
                    yield line
                continue
            delta_text = delta if isinstance(delta, str) else str(delta)
            if (
                not delta_text
                or delta_text.startswith("(Interrupt(")
                or delta_text.startswith("Command(update=")
            ):
                continue
            yield self.emit_stream_chunk(
                StreamTextChunk(
                    **common,
                    kind=StreamMessageKind.TOOL_CALL_OUTPUT_DELTA,
                    tool_call_id=tool_call_id or None,
                    tool_name=tool_name,
                    delta=delta_text,
                )
            )

        raw_output = getattr(tool_call, "output", None)
        async for line in self._emit_interrupt_chunks(
            self.extract_interrupts(raw_output),
            thread_id=thread_id,
            agent_id=agent_id,
            seen_interrupt_ids=seen_interrupt_ids,
            source=source,
            subagent_name=subagent_name,
            subagent_cause=subagent_cause,
        ):
            yield line

        tool_message = self._tool_message_from_output(raw_output)
        tool_content: str | None = None
        tool_content_type: str | None = None
        tool_sources: list[Source] | None = None
        if tool_message is not None:
            tool_content, tool_content_type, tool_sources = (
                self._tool_message_stream_fields(tool_message)
            )

        if tool_sources and collected_sources is not None:
            merged = self.merge_sources(collected_sources, tool_sources)
            collected_sources.clear()
            if merged:
                collected_sources.extend(merged)

        yield self.emit_stream_chunk(
            StreamTextChunk(
                **common,
                kind=StreamMessageKind.TOOL_CALL_FINISHED,
                tool_call_id=tool_call_id or None,
                tool_name=tool_name,
                input=resolved_input,
                content=tool_content,
                content_type=tool_content_type,
                sources=tool_sources,
                output=self.serialize_tool_output(raw_output),
                error=getattr(tool_call, "error", None),
            )
        )

    async def yield_subagent_chunks(
        self,
        subagent: Any,
        *,
        thread_id: str,
        agent_id: int,
        seen_interrupt_ids: set[str],
        collected_sources: list[Source] | None = None,
        seen_system_status: set[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream one ``run.subagents`` handle: nested messages and tool calls."""
        subagent_name, subagent_cause = self.resolve_subagent_metadata(subagent)

        yield self.emit_stream_chunk(
            StreamTextChunk(
                thread_id=thread_id,
                agent_id=agent_id,
                kind=StreamMessageKind.SUBAGENT_STARTED,
                source=StreamContentSource.SUBAGENT,
                subagent_name=subagent_name,
                subagent_cause=subagent_cause,
            )
        )
        if subagent_name:
            tool_call_id = ""
            if isinstance(subagent_cause, dict):
                raw_id = subagent_cause.get("tool_call_id")
                if isinstance(raw_id, str):
                    tool_call_id = raw_id
            status_scope = tool_call_id or subagent_name

            delegating_content = _DELEGATING_STATUS.format(name=subagent_name)
            running_content = _RUNNING_SUBAGENT_STATUS.format(name=subagent_name)
            status_specs = (
                (
                    f"{status_scope}:{delegating_content}",
                    delegating_content,
                    StreamContentSource.AGENT,
                    None,
                ),
                (
                    f"{status_scope}:{running_content}",
                    running_content,
                    StreamContentSource.SUBAGENT,
                    subagent_name,
                ),
            )
            for status_key, content, status_source, status_subagent in status_specs:
                if seen_system_status is not None:
                    line = self.emit_system_status_once(
                        seen_system_status,
                        status_key,
                        content=content,
                        thread_id=thread_id,
                        agent_id=agent_id,
                        source=status_source,
                        subagent_name=status_subagent,
                        subagent_cause=subagent_cause,
                    )
                    if line:
                        yield line
                else:
                    yield self.emit_system_status(
                        content=content,
                        thread_id=thread_id,
                        agent_id=agent_id,
                        source=status_source,
                        subagent_name=status_subagent,
                        subagent_cause=subagent_cause,
                    )

        async for name, item in self.merge_projection_feeds(
            self.projection_feeds_for_run(subagent, seen_interrupt_ids=seen_interrupt_ids)
        ):
            if name == "messages":
                async for line in self.yield_message_chunks(
                    item,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    source=StreamContentSource.SUBAGENT,
                    subagent_name=subagent_name,
                    subagent_cause=subagent_cause,
                    seen_system_status=seen_system_status,
                ):
                    yield line
                continue

            if name == "values":
                async for line in self.yield_interrupt_chunks(
                    item,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    seen_interrupt_ids=seen_interrupt_ids,
                    source=StreamContentSource.SUBAGENT,
                    subagent_name=subagent_name,
                    subagent_cause=subagent_cause,
                ):
                    yield line
                continue

            if name == "tool_calls":
                async for line in self.yield_run_tool_call_chunks(
                    item,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    seen_interrupt_ids=seen_interrupt_ids,
                    source=StreamContentSource.SUBAGENT,
                    subagent_name=subagent_name,
                    subagent_cause=subagent_cause,
                    collected_sources=collected_sources,
                ):
                    yield line

        yield self.emit_stream_chunk(
            StreamTextChunk(
                thread_id=thread_id,
                agent_id=agent_id,
                kind=StreamMessageKind.SUBAGENT_FINISHED,
                source=StreamContentSource.SUBAGENT,
                subagent_name=subagent_name,
                subagent_cause=subagent_cause,
                error=getattr(subagent, "error", None),
            )
        )

    async def stream_agent_content(
        self,
        agent: CompiledStateGraph,
        *,
        thread_id: str,
        agent_id: int,
        config: RunnableConfig,
        input_state: Any,
        access_token: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream message and tool-call projections via ``astream_events(version="v3")``.

        Mirrors the LangChain event-streaming guide::

            stream = agent.stream_events(input, version="v3")
            for message in stream.messages:
                for chunk in message.tool_calls:
                    print(f"tool call chunk: {chunk}")
                finalized = message.tool_calls.get()
                if finalized:
                    print(f"finalized tool calls: {finalized}")
            for call in stream.tool_calls:
                print(f"{call.tool_name}({call.input})")
                for delta in call.output_deltas:
                    print(delta, end="", flush=True)
                print(call.output, call.error)

            for subagent in stream.subagents:
                print(f"{subagent.name}: ", end="")
                for message in subagent.messages:
                    for token in message.text:
                        print(token, end="", flush=True)
                print()
        """
        sync_skills_for_thread(agent_id, thread_id)
        with mcp_access_token_context(access_token):
            run: AsyncGraphRunStream = await agent.astream_events(
                input_state,
                config=config,
                version=STREAM_EVENTS_VERSION,
            )

            seen_interrupt_ids: set[str] = set()
            seen_system_status: set[str] = set()
            collected_sources: list[Source] = []

            async with self._runs_lock:
                self._active_runs[thread_id] = run

            try:
                async with run:
                    async for name, item in self.merge_run_projections(
                        run,
                        seen_interrupt_ids=seen_interrupt_ids,
                    ):
                        if name == "messages":
                            async for line in self.yield_message_chunks(
                                item,
                                thread_id=thread_id,
                                agent_id=agent_id,
                                seen_system_status=seen_system_status,
                            ):
                                yield line
                            continue

                        if name == "values":
                            async for line in self.yield_interrupt_chunks(
                                item,
                                thread_id=thread_id,
                                agent_id=agent_id,
                                seen_interrupt_ids=seen_interrupt_ids,
                            ):
                                yield line
                            continue

                        if name == "tool_calls":
                            async for line in self.yield_run_tool_call_chunks(
                                item,
                                thread_id=thread_id,
                                agent_id=agent_id,
                                seen_interrupt_ids=seen_interrupt_ids,
                                collected_sources=collected_sources,
                            ):
                                yield line
                            continue

                        if name == "subagents":
                            async for line in self.yield_subagent_chunks(
                                item,
                                thread_id=thread_id,
                                agent_id=agent_id,
                                seen_interrupt_ids=seen_interrupt_ids,
                                collected_sources=collected_sources,
                                seen_system_status=seen_system_status,
                            ):
                                yield line

                    final_state = await run.output()
                    raw_result: dict[str, Any] = (
                        dict(final_state) if isinstance(final_state, dict) else {}
                    )
                    if isinstance(final_state, dict) and await run.interrupted():
                        run_interrupts = await run.interrupts()
                        if run_interrupts and not raw_result.get("__interrupt__"):
                            raw_result["__interrupt__"] = run_interrupts

                    async with self._runs_lock:
                        cancelled = thread_id in self._cancel_requested
                        self._cancel_requested.discard(thread_id)

                    yield await self.yield_run_finished_chunk(
                        thread_id=thread_id,
                        agent_id=agent_id,
                        raw_result=raw_result,
                        cancelled=cancelled,
                        sources=list(collected_sources) or None,
                    )
            finally:
                async with self._runs_lock:
                    self._active_runs.pop(thread_id, None)
                    self._cancel_requested.discard(thread_id)

    async def stream(
        self,
        payload: InvokeAgent,
        *,
        access_token: str | None = None,
    ) -> StreamingResponse:
        """Compile the requested agent and stream v3 message and tool-call projections."""
        model_config = payload.get("model_config")
        agent_id = payload["agent_id"]
        agent = await self._invoke_service.get_compiled_agent(agent_id, model_config)
        thread_id = payload.get("thread_id") or str(uuid.uuid4())
        config: RunnableConfig = with_langfuse_config(
            {"configurable": {"thread_id": thread_id}},
            thread_id=thread_id,
            agent_id=agent_id,
        )
        input_state = await self._invoke_service.resolve_input_state(
            agent,
            payload,
            config=config,
        )

        return StreamingResponse(
            self._flushing_sse(
                self.stream_agent_content(
                    agent,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    config=config,
                    input_state=input_state,
                    access_token=access_token,
                )
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Encoding": "none",
            },
        )

    async def _flushing_sse(self, frames: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield each SSE frame, then yield to the event loop so ASGI can flush."""
        async for frame in frames:
            yield frame
            await asyncio.sleep(0)
