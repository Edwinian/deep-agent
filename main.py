"""FastAPI entrypoint for agent invocation."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from invoke_service import InvokeService
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import InvokeResponse
from schemas.thread_teardown_response import ThreadTeardownResponse
from stream_service import StreamService
from db.agent_store import init_agent_db
from tools.mcp_auth import parse_authorization_header
from utils.get_checkpointer import close_sqlite_checkpointer, init_sqlite_checkpointer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize persistent SQLite checkpoints for HITL thread continuity."""
    init_agent_db()
    await init_sqlite_checkpointer()
    yield
    await close_sqlite_checkpointer()


app = FastAPI(title="Deep Agents API", lifespan=lifespan)

invoke_service = InvokeService()
stream_service = StreamService(invoke_service)


@app.post("/invoke")
async def invoke(
    payload: InvokeAgent,
    authorization: str | None = Header(default=None),
) -> InvokeResponse:
    """Compile the requested agent and run or resume one turn."""
    access_token = parse_authorization_header(authorization)
    return await invoke_service.invoke(payload, access_token=access_token)


@app.post("/stream")
async def stream(
    payload: InvokeAgent,
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """Compile the requested agent and stream v3 projections over SSE."""
    access_token = parse_authorization_header(authorization)
    return await stream_service.stream(payload, access_token=access_token)


@app.post("/cancel-stream/{thread_id}")
async def cancel_stream(thread_id: str) -> dict[str, bool | str]:
    """Abort an in-flight /stream run for this thread_id."""
    cancelled = await stream_service.cancel_stream(thread_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"No active stream found for thread_id {thread_id!r}",
        )
    return {"thread_id": thread_id, "cancelled": True}


@app.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    agent_id: int | None = Query(
        default=None,
        description=(
            "Agent used to detect awaiting_tool_permission before teardown. "
            "Defaults to the general agent when omitted."
        ),
    ),
) -> ThreadTeardownResponse:
    """Delete checkpoints and the Daytona sandbox for a completed thread.

    The client owns ``thread_id`` per user session and calls this when the
    session ends. Returns 409 if the thread is awaiting HITL approval or has
    an active /stream run.
    """
    if await stream_service.has_active_stream(thread_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Thread has an active stream. Cancel it with "
                f"POST /cancel-stream/{thread_id} before deleting."
            ),
        )
    return await invoke_service.teardown_thread(thread_id, agent_id=agent_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
