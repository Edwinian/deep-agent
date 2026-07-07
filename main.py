"""FastAPI entrypoint for agent invocation."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage, messages_to_dict
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Interrupt as LangGraphInterrupt

from agents.agent_registry import AGENT_REGISTRY
from agents.types import ModelConfig
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import (
    InvokeResponse,
    InvokeStatus,
    SerializedInterruptPayload,
    SerializedMessage,
    StreamChunk,
    StreamEvent,
    StreamMode,
    StreamSerializableInput,
    StreamSerializedMessageDict,
    StreamSerializedValue,
)
from utils.compile_agent import compile_agent
from utils.get_checkpointer import CheckpointerType, get_checkpointer
from utils.hitl import (
    build_resume_command,
    collect_action_requests,
    collect_pending_interrupts,
    is_resume_request,
)
from utils.langfuse_tracing import with_langfuse_config

app = FastAPI(title="Deep Agents API")

_agent_cache: dict[int, CompiledStateGraph] = {}
_shared_checkpointer = get_checkpointer(CheckpointerType.IN_MEMORY)


def _get_compiled_agent(agent_id: int, model_config: ModelConfig | None) -> CompiledStateGraph:
    agent_spec = AGENT_REGISTRY.get(agent_id)
    if agent_spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent_id: {agent_id}")

    if model_config is not None:
        return compile_agent(
            agent_spec,
            model_config=model_config,
            checkpointer=_shared_checkpointer,
        )

    if agent_id not in _agent_cache:
        _agent_cache[agent_id] = compile_agent(
            agent_spec,
            checkpointer=_shared_checkpointer,
        )
    return _agent_cache[agent_id]


def _serialize_stream_value(
    event: StreamSerializableInput | LangGraphInterrupt | BaseMessage,
) -> StreamSerializedValue:
    if isinstance(event, LangGraphInterrupt):
        interrupt: SerializedInterruptPayload = {
            "value": _serialize_stream_value(event.value),
            "id": event.id,
        }
        return interrupt
    if isinstance(event, BaseMessage):
        message: StreamSerializedMessageDict = messages_to_dict([event])[0]
        return message
    if isinstance(event, list):
        if event and isinstance(event[0], BaseMessage):
            return messages_to_dict(event)
        return [_serialize_stream_value(item) for item in event]
    if isinstance(event, dict):
        return {key: _serialize_stream_value(item) for key, item in event.items()}
    if isinstance(event, tuple):
        return [_serialize_stream_value(item) for item in event]
    return event


def _serialize_stream_event(
    event: StreamSerializableInput | LangGraphInterrupt | BaseMessage,
) -> StreamEvent:
    """Serialize a LangGraph ``astream`` event payload for :class:`StreamChunk`."""
    return cast(StreamEvent, _serialize_stream_value(event))


def _serialize_messages(messages: list[Any]) -> list[SerializedMessage]:
    """Serialize LangChain messages for the client-facing invoke response."""
    if not messages:
        return []
    if isinstance(messages[0], BaseMessage):
        message_dicts = messages_to_dict(messages)
    else:
        message_dicts = messages
    return [SerializedMessage.model_validate(message) for message in message_dicts]


def _has_pending_interrupts(result: dict[str, Any]) -> bool:
    interrupts = result.get("__interrupt__")
    return bool(interrupts)


def _build_invoke_response(
    *,
    thread_id: str,
    agent_id: int,
    raw_result: dict[str, Any],
) -> InvokeResponse:
    serialized_messages = _serialize_messages(raw_result.get("messages", []))

    if _has_pending_interrupts(raw_result):
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


async def _run_agent(
    agent: CompiledStateGraph,
    payload: InvokeAgent,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:
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
                detail=(
                    "No interrupted tool calls found for this thread. "
                    "The server may have restarted (in-memory checkpoints are cleared on reload) "
                    "or the thread_id is wrong."
                ),
            )

        resume_input = build_resume_command(
            pending_interrupts,
            permissions or [],
        )
        return await agent.ainvoke(resume_input, config=config)

    message = payload.get("message")
    if not message:
        raise HTTPException(
            status_code=400,
            detail="message is required unless resuming with permissions.",
        )

    return await agent.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )


async def _stream_agent_events(
    agent: CompiledStateGraph,
    *,
    thread_id: str,
    agent_id: int,
    config: RunnableConfig,
    input_state: Any,
) -> AsyncIterator[str]:
    async for graph_name, stream_mode, event in agent.astream(
        input_state,
        stream_mode=["updates", "values"],
        subgraphs=True,
        config=config,
    ):
        chunk = StreamChunk(
            thread_id=thread_id,
            agent_id=agent_id,
            graph=list(graph_name),
            stream_mode=StreamMode(stream_mode),
            event=_serialize_stream_event(event),
        )
        yield json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n"


@app.post("/invoke")
async def invoke(payload: InvokeAgent) -> InvokeResponse:
    """Compile the requested agent and run or resume one turn."""
    model_config = payload.get("model_config")
    agent_id = payload["agent_id"]
    agent = _get_compiled_agent(agent_id, model_config)
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    result = await _run_agent(
        agent,
        payload,
        config=config,
    )

    return _build_invoke_response(
        thread_id=thread_id,
        agent_id=agent_id,
        raw_result=result,
    )


@app.post("/stream")
async def stream(payload: InvokeAgent) -> StreamingResponse:
    """Compile the requested agent and stream graph updates to the client."""
    model_config = payload.get("model_config")
    agent_id = payload["agent_id"]
    agent = _get_compiled_agent(agent_id, model_config)
    thread_id = payload.get("thread_id") or str(uuid.uuid4())
    config: RunnableConfig = with_langfuse_config(
        {"configurable": {"thread_id": thread_id}},
        thread_id=thread_id,
        agent_id=agent_id,
    )

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
                detail=(
                    "No interrupted tool calls found for this thread. "
                    "The server may have restarted (in-memory checkpoints are cleared on reload) "
                    "or the thread_id is wrong."
                ),
            )
        input_state = build_resume_command(
            pending_interrupts,
            permissions or [],
        )
    else:
        message = payload.get("message")
        if not message:
            raise HTTPException(
                status_code=400,
                detail="message is required unless resuming with permissions.",
            )
        input_state = {"messages": [HumanMessage(content=message)]}

    return StreamingResponse(
        _stream_agent_events(
            agent,
            thread_id=thread_id,
            agent_id=agent_id,
            config=config,
            input_state=input_state,
        ),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
