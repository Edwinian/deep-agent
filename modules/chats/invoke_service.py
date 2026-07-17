"""Service layer for POST /invoke and shared agent orchestration."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    messages_to_dict,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agents.ids import GENERAL_AGENT_ID
from agents.types import DeepAgent, ModelConfig
from db.agent_store import AgentNotFoundError
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import (
    InvokeResponse,
    InvokeStatus,
    SerializedMessage,
)
from schemas.source import SOURCES_KEY, Source
from schemas.thread_history_response import (
    HistoryChatMessage,
    HistoryToolEvent,
    ThreadHistoryResponse,
)
from schemas.thread_rewind_response import ThreadRewindResponse
from schemas.thread_teardown_response import ThreadTeardownResponse
from mcp_interceptors.mcp_auth import mcp_access_token_context
from utils.compile_agent import compile_agent
from utils.content_blocks import extract_user_text, is_non_text_content_block_dump
from utils.daytona_sandbox import delete_daytona_sandbox, sync_skills_for_thread
from utils.get_checkpointer import delete_thread_checkpoints, get_sqlite_checkpointer
from utils.hitl import (
    build_resume_command,
    collect_action_requests,
    collect_pending_interrupts,
    is_resume_request,
)
from utils.langfuse_tracing import with_langfuse_config
from utils.resolve_agent import resolve_agent

_INTERRUPT_NOT_FOUND_DETAIL = (
    "No interrupted tool calls found for this thread. "
    "The thread_id may be wrong or the checkpoint database was reset."
)

_THREAD_AWAITING_PERMISSION_DETAIL = (
    "Thread is awaiting tool permission. Approve, edit, or reject the pending "
    "action before deleting the thread."
)

_NO_USER_MESSAGE_DETAIL = (
    "No user message found in this thread. Send a message before regenerating."
)

_THREAD_NOT_FOUND_DETAIL = (
    "No checkpoint history found for this thread_id."
)


class InvokeService:
    """Compile agents, run turns, and build invoke responses."""

    def __init__(self, checkpointer: Checkpointer | None = None) -> None:
        self._checkpointer = checkpointer
        self._agent_cache: dict[int, CompiledStateGraph] = {}

    def _compile_kwargs(self) -> dict[str, Checkpointer]:
        """Pass an explicit checkpointer override when one was injected."""
        if self._checkpointer is None:
            return {}
        return {"checkpointer": self._checkpointer}

    async def _resolve_agent_spec(self, agent_id: int) -> DeepAgent:
        """Load the agent spec from the database."""
        try:
            return await resolve_agent(agent_id)
        except AgentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def get_compiled_agent(
        self,
        agent_id: int,
        model_config: ModelConfig | None,
    ) -> CompiledStateGraph:
        """Return a cached compiled agent graph, compiling on first use per agent_id."""
        if model_config is not None:
            agent_spec = await self._resolve_agent_spec(agent_id)
            return compile_agent(
                agent_spec,
                model_config=model_config,
                **self._compile_kwargs(),
            )

        if agent_id not in self._agent_cache:
            agent_spec = await self._resolve_agent_spec(agent_id)
            self._agent_cache[agent_id] = compile_agent(
                agent_spec,
                **self._compile_kwargs(),
            )
        return self._agent_cache[agent_id]

    @staticmethod
    def content_to_text(content: Any) -> str:
        """Extract user-facing text blocks from LangChain message content."""
        return extract_user_text(content)

    @staticmethod
    def content_to_reasoning(content: Any) -> str:
        """Extract reasoning blocks from LangChain message content."""
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                reasoning = block.get("reasoning")
                if isinstance(reasoning, str):
                    parts.append(reasoning)
        return "".join(parts)

    @staticmethod
    def last_task_tool_output(messages: list[Any]) -> str:
        """Return the latest non-empty ``task`` tool result (subagent answer)."""
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                name = str(message.name or "")
                if name != "task":
                    continue
                text = InvokeService.content_to_text(message.content).strip()
                if text:
                    return text
                continue
            if isinstance(message, dict) and message.get("type") == "tool":
                data = message.get("data", {})
                if not isinstance(data, dict):
                    continue
                name = str(data.get("name") or message.get("name") or "")
                if name != "task":
                    continue
                text = InvokeService.content_to_text(data.get("content")).strip()
                if text:
                    return text
        return ""

    @staticmethod
    def last_ai_reply(
        messages: list[Any],
        *,
        allow_task_fallback: bool = True,
    ) -> str:
        """Return the invoke-equivalent final assistant reply from graph messages.

        Use only the chronologically last AI message's user-facing text.
        Skip stringified tool_call-only dumps. If that turn has no answer
        (common when Grok stalls on file tools), optionally fall back to the
        latest ``task`` subagent output — do not reuse earlier mid-run AI
        chatter. Disable task fallback on HITL pauses so a prior research
        answer is not shown as a finished reply while tools still need approval.
        """
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = InvokeService.content_to_text(message.content).strip()
                if text and not is_non_text_content_block_dump(text):
                    return text
                if allow_task_fallback:
                    return InvokeService.last_task_tool_output(messages)
                return ""
            if isinstance(message, dict) and message.get("type") == "ai":
                data = message.get("data", {})
                if not isinstance(data, dict):
                    continue
                text = InvokeService.content_to_text(data.get("content")).strip()
                if text and not is_non_text_content_block_dump(text):
                    return text
                if allow_task_fallback:
                    return InvokeService.last_task_tool_output(messages)
                return ""
        if allow_task_fallback:
            return InvokeService.last_task_tool_output(messages)
        return ""

    @staticmethod
    def _normalize_message_dict(message: dict[str, Any]) -> dict[str, Any]:
        """Coerce message content to a string for API serialization."""
        data = message.get("data")
        if not isinstance(data, dict):
            return message

        content = data.get("content")
        if not isinstance(content, str):
            data["content"] = InvokeService.content_to_text(content)

        if message.get("type") == "ai":
            additional_kwargs = data.get("additional_kwargs")
            if isinstance(additional_kwargs, dict) and "reasoning_content" in additional_kwargs:
                data["additional_kwargs"] = {
                    key: value
                    for key, value in additional_kwargs.items()
                    if key != "reasoning_content"
                }

        return message

    @staticmethod
    def serialize_messages(messages: list[Any]) -> list[SerializedMessage]:
        """Serialize LangChain messages for the client-facing invoke response."""
        if not messages:
            return []
        if isinstance(messages[0], BaseMessage):
            message_dicts = messages_to_dict(messages)
        else:
            message_dicts = messages
        return [
            SerializedMessage.model_validate(
                InvokeService._normalize_message_dict(message)
            )
            for message in message_dicts
        ]

    @staticmethod
    def has_pending_interrupts(result: dict[str, Any]) -> bool:
        """Return True when the agent state contains unresolved HITL interrupts."""
        interrupts = result.get("__interrupt__")
        return bool(interrupts)

    def build_invoke_response(
        self,
        *,
        thread_id: str,
        agent_id: int,
        raw_result: dict[str, Any],
    ) -> InvokeResponse:
        """Build the client-facing invoke response from raw agent state."""
        serialized_messages = self.serialize_messages(raw_result.get("messages", []))

        if self.has_pending_interrupts(raw_result):
            return InvokeResponse(
                thread_id=thread_id,
                agent_id=agent_id,
                status=InvokeStatus.AWAITING_TOOL_PERMISSION,
                messages=serialized_messages,
                action_requests=collect_action_requests(raw_result["__interrupt__"]),
            )

        return InvokeResponse(
            thread_id=thread_id,
            agent_id=agent_id,
            status=InvokeStatus.COMPLETED,
            messages=serialized_messages,
        )

    async def resolve_input_state(
        self,
        agent: CompiledStateGraph,
        payload: InvokeAgent,
        *,
        config: RunnableConfig,
    ) -> Any:
        """Build graph input for a new turn or a HITL resume."""
        permissions = payload.get("permissions")

        if is_resume_request(
            thread_id=payload.get("thread_id"),
            permissions=permissions,
        ):
            snapshot = await agent.aget_state(config, subgraphs=True)
            pending_interrupts = collect_pending_interrupts(snapshot)
            if not pending_interrupts:
                raise HTTPException(
                    status_code=400,
                    detail=_INTERRUPT_NOT_FOUND_DETAIL,
                )
            return build_resume_command(
                pending_interrupts,
                permissions or [],
            )

        message = payload.get("message")
        if not message:
            raise HTTPException(
                status_code=400,
                detail="message is required unless resuming with permissions.",
            )
        return {"messages": [HumanMessage(content=message)]}

    async def run_agent(
        self,
        agent: CompiledStateGraph,
        payload: InvokeAgent,
        *,
        config: RunnableConfig,
        access_token: str | None = None,
    ) -> dict[str, Any]:
        """Run or resume one agent turn and return the final graph state."""
        thread_id = (
            payload.get("thread_id")
            or config.get("configurable", {}).get("thread_id")
            or ""
        )
        config = with_langfuse_config(
            config,
            thread_id=str(thread_id),
            agent_id=payload["agent_id"],
        )
        input_state = await self.resolve_input_state(
            agent,
            payload,
            config=config,
        )
        if thread_id:
            sync_skills_for_thread(payload["agent_id"], thread_id)
        with mcp_access_token_context(access_token):
            return await agent.ainvoke(input_state, config=config)

    async def invoke(
        self,
        payload: InvokeAgent,
        *,
        access_token: str | None = None,
    ) -> InvokeResponse:
        """Compile the requested agent and run or resume one turn."""
        model_config = payload.get("model_config")
        agent_id = payload["agent_id"]
        agent = await self.get_compiled_agent(agent_id, model_config)
        thread_id = payload.get("thread_id") or str(uuid.uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        result = await self.run_agent(
            agent,
            payload,
            config=config,
            access_token=access_token,
        )

        return self.build_invoke_response(
            thread_id=thread_id,
            agent_id=agent_id,
            raw_result=result,
        )

    async def _agent_ids_for_hitl_check(self, agent_id: int | None) -> list[int]:
        if agent_id is not None:
            return [agent_id]
        if self._agent_cache:
            return list(self._agent_cache.keys())
        return [GENERAL_AGENT_ID]

    async def thread_awaiting_tool_permission(
        self,
        thread_id: str,
        *,
        agent_id: int | None = None,
    ) -> bool:
        """Return True when the thread has unresolved human-in-the-loop interrupts."""
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for candidate_agent_id in await self._agent_ids_for_hitl_check(agent_id):
            agent = await self.get_compiled_agent(candidate_agent_id, None)
            snapshot = await agent.aget_state(config, subgraphs=True)
            if collect_pending_interrupts(snapshot):
                return True
        return False

    @staticmethod
    def _is_human_message(message: Any) -> bool:
        if isinstance(message, HumanMessage):
            return True
        if isinstance(message, dict) and message.get("type") == "human":
            return True
        return False

    @staticmethod
    def _message_text(message: Any) -> str:
        if isinstance(message, BaseMessage):
            return InvokeService.content_to_text(message.content).strip()
        if isinstance(message, dict):
            data = message.get("data")
            if isinstance(data, dict):
                return InvokeService.content_to_text(data.get("content")).strip()
            return InvokeService.content_to_text(message.get("content")).strip()
        return ""

    @staticmethod
    def _message_reasoning(message: Any) -> str:
        if isinstance(message, AIMessage):
            from_content = InvokeService.content_to_reasoning(message.content)
            if from_content:
                return from_content.strip()
            extra = message.additional_kwargs or {}
            reasoning = extra.get("reasoning_content")
            return str(reasoning).strip() if reasoning else ""
        if isinstance(message, dict) and message.get("type") == "ai":
            data = message.get("data")
            if not isinstance(data, dict):
                return ""
            from_content = InvokeService.content_to_reasoning(data.get("content"))
            if from_content:
                return from_content.strip()
            extra = data.get("additional_kwargs") or {}
            if isinstance(extra, dict):
                reasoning = extra.get("reasoning_content")
                return str(reasoning).strip() if reasoning else ""
        return ""

    @staticmethod
    def _tool_sources(message: Any) -> list[Source]:
        raw: Any = None
        if isinstance(message, ToolMessage):
            raw = (message.additional_kwargs or {}).get(SOURCES_KEY)
        elif isinstance(message, dict) and message.get("type") == "tool":
            data = message.get("data")
            if isinstance(data, dict):
                raw = (data.get("additional_kwargs") or {}).get(SOURCES_KEY)
        if not isinstance(raw, list):
            return []
        sources: list[Source] = []
        for item in raw:
            parsed = InvokeService._parse_source(item)
            if parsed is not None:
                sources.append(parsed)
        return sources

    @staticmethod
    def _parse_source(item: Any) -> Source | None:
        """Normalize a Source model or raw dict from checkpoint state."""
        if isinstance(item, Source):
            return item
        if isinstance(item, dict):
            try:
                return Source.model_validate(item)
            except Exception:
                return None
        return None

    @classmethod
    def _merge_history_sources(
        cls,
        existing: list[Source] | None,
        incoming: Any,
    ) -> list[Source] | None:
        """Dedupe sources by URL, accepting Source models or dict payloads."""
        merged: dict[str, Source] = {}
        for source in existing or []:
            parsed = cls._parse_source(source)
            if parsed is not None and parsed.url:
                merged[parsed.url] = parsed
        if isinstance(incoming, list):
            items = incoming
        else:
            items = [incoming]
        for item in items:
            parsed = cls._parse_source(item)
            if parsed is not None and parsed.url:
                merged[parsed.url] = parsed
        return list(merged.values()) or None

    @classmethod
    def build_history_messages(
        cls,
        messages: list[Any],
        *,
        state_values: dict[str, Any] | None = None,
    ) -> list[HistoryChatMessage]:
        """Collapse LangChain checkpoint messages into chat UI bubbles."""
        from tools.web_search.web_search_tool import _load_sources_file

        history: list[HistoryChatMessage] = []
        current_assistant: HistoryChatMessage | None = None
        tools_by_id: dict[str, HistoryToolEvent] = {}

        for message in messages:
            if isinstance(message, HumanMessage) or (
                isinstance(message, dict) and message.get("type") == "human"
            ):
                current_assistant = None
                tools_by_id = {}
                msg_id = getattr(message, "id", None) or (
                    (message.get("data") or {}).get("id")
                    if isinstance(message, dict)
                    else None
                )
                history.append(
                    HistoryChatMessage(
                        id=str(msg_id or uuid.uuid4()),
                        role="user",
                        content=cls._message_text(message),
                    )
                )
                continue

            if isinstance(message, AIMessage) or (
                isinstance(message, dict) and message.get("type") == "ai"
            ):
                tool_calls: list[Any] = []
                msg_id = None
                if isinstance(message, AIMessage):
                    msg_id = message.id
                    tool_calls = list(message.tool_calls or [])
                elif isinstance(message, dict):
                    data = message.get("data") or {}
                    msg_id = data.get("id") if isinstance(data, dict) else None
                    if isinstance(data, dict):
                        tool_calls = list(data.get("tool_calls") or [])

                tools: list[HistoryToolEvent] = []
                tools_by_id = {}
                for call in tool_calls:
                    if isinstance(call, dict):
                        tool_id = str(call.get("id") or uuid.uuid4())
                        tool = HistoryToolEvent(
                            id=tool_id,
                            name=str(call.get("name") or "tool"),
                            args=dict(call.get("args") or {}),
                            status="done",
                        )
                    else:
                        tool_id = str(getattr(call, "id", None) or uuid.uuid4())
                        tool = HistoryToolEvent(
                            id=tool_id,
                            name=str(getattr(call, "name", None) or "tool"),
                            args=dict(getattr(call, "args", None) or {}),
                            status="done",
                        )
                    tools.append(tool)
                    tools_by_id[tool_id] = tool

                reasoning = cls._message_reasoning(message) or None
                current_assistant = HistoryChatMessage(
                    id=str(msg_id or uuid.uuid4()),
                    role="assistant",
                    content=cls._message_text(message),
                    reasoning=reasoning,
                    tools=tools or None,
                )
                history.append(current_assistant)
                continue

            if isinstance(message, ToolMessage) or (
                isinstance(message, dict) and message.get("type") == "tool"
            ):
                tool_call_id = None
                tool_name = "tool"
                output = cls._message_text(message)
                status = "done"
                if isinstance(message, ToolMessage):
                    tool_call_id = message.tool_call_id
                    tool_name = message.name or tool_name
                    if getattr(message, "status", None) == "error":
                        status = "error"
                elif isinstance(message, dict):
                    data = message.get("data") or {}
                    if isinstance(data, dict):
                        tool_call_id = data.get("tool_call_id")
                        tool_name = data.get("name") or tool_name
                        if data.get("status") == "error":
                            status = "error"

                tool = tools_by_id.get(str(tool_call_id)) if tool_call_id else None
                if tool is None and current_assistant is not None:
                    tool = HistoryToolEvent(
                        id=str(tool_call_id or uuid.uuid4()),
                        name=str(tool_name),
                        status=status,  # type: ignore[arg-type]
                        output=output or None,
                    )
                    current_tools = list(current_assistant.tools or [])
                    current_tools.append(tool)
                    current_assistant.tools = current_tools
                    tools_by_id[tool.id] = tool
                elif tool is not None:
                    tool.output = output or tool.output
                    tool.status = status  # type: ignore[assignment]
                    if tool_name and tool.name == "tool":
                        tool.name = str(tool_name)

                if current_assistant is not None:
                    tool_sources = cls._tool_sources(message)
                    if tool_sources:
                        current_assistant.sources = cls._merge_history_sources(
                            current_assistant.sources,
                            tool_sources,
                        )
                continue

        if history and state_values:
            files = state_values.get("files")
            file_sources = _load_sources_file(files) if isinstance(files, dict) else []
            if file_sources:
                for item in reversed(history):
                    if item.role == "assistant":
                        item.sources = cls._merge_history_sources(
                            item.sources,
                            file_sources,
                        )
                        break

        return history

    async def get_history(
        self,
        thread_id: str,
        *,
        agent_id: int | None = None,
    ) -> ThreadHistoryResponse:
        """Load checkpoint messages for a thread as chat history bubbles."""
        resolved_agent_id = agent_id if agent_id is not None else GENERAL_AGENT_ID
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        checkpointer = await get_sqlite_checkpointer()
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            raise HTTPException(status_code=404, detail=_THREAD_NOT_FOUND_DETAIL)

        agent = await self.get_compiled_agent(resolved_agent_id, None)
        snapshot = await agent.aget_state(config, subgraphs=True)
        values = dict(snapshot.values or {})
        messages = list(values.get("messages") or [])
        history = self.build_history_messages(messages, state_values=values)

        pending = collect_pending_interrupts(snapshot)
        if pending:
            return ThreadHistoryResponse(
                thread_id=thread_id,
                agent_id=resolved_agent_id,
                status=InvokeStatus.AWAITING_TOOL_PERMISSION,
                messages=history,
                action_requests=collect_action_requests(pending),
                interrupt_ids=[interrupt.id for interrupt in pending],
            )

        return ThreadHistoryResponse(
            thread_id=thread_id,
            agent_id=resolved_agent_id,
            status=InvokeStatus.COMPLETED,
            messages=history,
        )

    async def clear_from_last_user(
        self,
        thread_id: str,
        *,
        agent_id: int | None = None,
    ) -> ThreadRewindResponse:
        """Remove the last user turn and everything after it from checkpoint state.

        Keeps prior turns intact so ``POST /stream`` can re-append the same user
        prompt for regenerate. Also clears pending HITL interrupts by rewriting
        checkpoint messages.
        """
        resolved_agent_id = agent_id if agent_id is not None else GENERAL_AGENT_ID
        agent = await self.get_compiled_agent(resolved_agent_id, None)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        snapshot = await agent.aget_state(config, subgraphs=True)
        messages = list((snapshot.values or {}).get("messages") or [])
        if not messages:
            raise HTTPException(status_code=404, detail=_NO_USER_MESSAGE_DETAIL)

        last_human_idx = None
        for index in range(len(messages) - 1, -1, -1):
            if self._is_human_message(messages[index]):
                last_human_idx = index
                break

        if last_human_idx is None:
            raise HTTPException(status_code=404, detail=_NO_USER_MESSAGE_DETAIL)

        last_user_text = self._message_text(messages[last_human_idx])
        if not last_user_text:
            raise HTTPException(status_code=404, detail=_NO_USER_MESSAGE_DETAIL)

        kept = messages[:last_human_idx]
        removed_count = len(messages) - len(kept)
        await agent.aupdate_state(
            config,
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]},
        )
        return ThreadRewindResponse(
            thread_id=thread_id,
            message=last_user_text,
            removed_count=removed_count,
            remaining_count=len(kept),
        )

    async def teardown_thread(
        self,
        thread_id: str,
        *,
        agent_id: int | None = None,
    ) -> ThreadTeardownResponse:
        """Delete LangGraph checkpoints and the Daytona sandbox for a thread."""
        if await self.thread_awaiting_tool_permission(thread_id, agent_id=agent_id):
            raise HTTPException(
                status_code=409,
                detail=_THREAD_AWAITING_PERMISSION_DETAIL,
            )

        await delete_thread_checkpoints(thread_id)
        sandbox_deleted = delete_daytona_sandbox(thread_id)
        return ThreadTeardownResponse(
            thread_id=thread_id,
            checkpoint_deleted=True,
            sandbox_deleted=sandbox_deleted,
        )
