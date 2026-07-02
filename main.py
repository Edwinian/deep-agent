"""FastAPI entrypoint for agent invocation."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from dotenv import load_dotenv
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
    InvokeResult,
    InvokeStatus,
    StreamChunk,
    StreamMode,
)
from utils.compile_agent import compile_agent
from utils.hitl import (
    build_permission_message,
    build_resume_command,
    enrich_interrupt_tool_call_ids,
    is_resume_request,
)

load_dotenv()

app = FastAPI(title="Deep Agents API")

_agent_cache: dict[int, CompiledStateGraph] = {}


def _get_compiled_agent(agent_id: int, model_config: ModelConfig | None) -> CompiledStateGraph:
    agent_spec = AGENT_REGISTRY.get(agent_id)
    if agent_spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent_id: {agent_id}")

    if model_config is not None:
        return compile_agent(agent_spec, model_config=model_config)

    if agent_id not in _agent_cache:
        _agent_cache[agent_id] = compile_agent(agent_spec)
    return _agent_cache[agent_id]


def _serialize_stream_value(value: Any) -> Any:
    if isinstance(value, LangGraphInterrupt):
        return {
            "value": _serialize_stream_value(value.value),
            "id": value.id,
        }
    if isinstance(value, BaseMessage):
        return messages_to_dict([value])[0]
    if isinstance(value, list):
        if value and isinstance(value[0], BaseMessage):
            return messages_to_dict(value)
        return [_serialize_stream_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_stream_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_stream_value(item) for item in value]
    return value


def _serialize_invoke_result(
    result: dict[str, Any],
    *,
    messages: list[Any] | None = None,
) -> InvokeResult:
    serialized = _serialize_stream_value(result)
    enrich_interrupt_tool_call_ids(
        serialized,
        messages if messages is not None else result.get("messages", []),
    )
    return InvokeResult.model_validate(serialized)


def _has_pending_interrupts(result: dict[str, Any]) -> bool:
    interrupts = result.get("__interrupt__")
    return bool(interrupts)


def _build_invoke_response(
    *,
    thread_id: str,
    agent_id: int,
    raw_result: dict[str, Any],
) -> InvokeResponse:
    messages = raw_result.get("messages", [])
    invoke_result = _serialize_invoke_result(raw_result, messages=messages)

    if _has_pending_interrupts(raw_result):
        return InvokeResponse(
            thread_id=thread_id,
            agent_id=agent_id,
            status=InvokeStatus.AWAITING_TOOL_PERMISSION,
            permission_message=build_permission_message(raw_result["__interrupt__"]),
            result=invoke_result,
        )

    return InvokeResponse(
        thread_id=thread_id,
        agent_id=agent_id,
        status=InvokeStatus.COMPLETED,
        result=invoke_result,
    )


async def _run_agent(
    agent: CompiledStateGraph,
    payload: InvokeAgent,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:
    permissions = payload.get("permissions")

    if is_resume_request(
        thread_id=payload.get("thread_id"),
        permissions=permissions,
    ):
        snapshot = await agent.aget_state(config)
        if not snapshot.interrupts:
            raise HTTPException(
                status_code=400,
                detail="No interrupted tool calls found for this thread.",
            )

        resume_input = build_resume_command(
            snapshot.interrupts,
            permissions or [],
            snapshot.values.get("messages", []),
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
            event=_serialize_stream_value(event),
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
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    permissions = payload.get("permissions")
    if is_resume_request(
        thread_id=payload.get("thread_id"),
        permissions=permissions,
    ):
        snapshot = await agent.aget_state(config)
        input_state = build_resume_command(
            snapshot.interrupts,
            permissions or [],
            snapshot.values.get("messages", []),
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
