"""FastAPI entrypoint for agent invocation."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from invoke_service import InvokeService
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import InvokeResponse
from stream_service import StreamService
from db.agent_store import init_agent_db
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
async def invoke(payload: InvokeAgent) -> InvokeResponse:
    """Compile the requested agent and run or resume one turn."""
    return await invoke_service.invoke(payload)


@app.post("/stream")
async def stream(payload: InvokeAgent) -> StreamingResponse:
    """Compile the requested agent and stream v3 projections over SSE."""
    return await stream_service.stream(payload)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
