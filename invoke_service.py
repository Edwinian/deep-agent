"""Service layer for POST /invoke and shared agent orchestration."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_to_dict
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agents.agent_registry import AGENT_REGISTRY
from agents.general_agent import GENERAL_AGENT_ID, resolve_general_agent
from agents.types import DeepAgent, ModelConfig
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import (
    InvokeResponse,
    InvokeStatus,
    SerializedMessage,
)
from utils.compile_agent import compile_agent
from utils.hitl import (
    build_resume_command,
    collect_action_requests,
    collect_pending_interrupts,
    is_resume_request,
)
from utils.langfuse_tracing import with_langfuse_config

_INTERRUPT_NOT_FOUND_DETAIL = (
    "No interrupted tool calls found for this thread. "
    "The thread_id may be wrong or the checkpoint database was reset."
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
        """Load the agent spec, resolving dynamic tools when needed."""
        if agent_id == GENERAL_AGENT_ID:
            return await resolve_general_agent()

        agent_spec = AGENT_REGISTRY.get(agent_id)
        if agent_spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown agent_id: {agent_id}")
        return agent_spec

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
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

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
    def last_ai_reply(messages: list[Any]) -> str:
        """Return the invoke-equivalent final assistant reply from graph messages."""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = InvokeService.content_to_text(message.content)
                if text:
                    return text
                reasoning = InvokeService.content_to_reasoning(message.content)
                if reasoning:
                    return reasoning
                continue
            if isinstance(message, dict) and message.get("type") == "ai":
                data = message.get("data", {})
                if not isinstance(data, dict):
                    continue
                content = data.get("content")
                text = InvokeService.content_to_text(content)
                if text:
                    return text
                reasoning = InvokeService.content_to_reasoning(content)
                if reasoning:
                    return reasoning
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
        return await agent.ainvoke(input_state, config=config)

    async def invoke(self, payload: InvokeAgent) -> InvokeResponse:
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
        )

        return self.build_invoke_response(
            thread_id=thread_id,
            agent_id=agent_id,
            raw_result=result,
        )
