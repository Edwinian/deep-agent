"""FastAPI entrypoint for agent invocation."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from invoke_service import InvokeService
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import InvokeResponse
from stream_service import StreamService

app = FastAPI(title="Deep Agents API")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
