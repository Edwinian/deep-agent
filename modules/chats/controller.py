"""Chat / thread endpoints grouped under the ``/chats`` router."""

from __future__ import annotations

from fastapi import File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from .invoke_service import InvokeService
from mcp_interceptors.mcp_auth import parse_authorization_header
from schemas.invoke_request import InvokeAgent
from schemas.invoke_response import InvokeResponse
from schemas.thread_history_response import ThreadHistoryResponse
from schemas.thread_rewind_response import ThreadRewindResponse
from schemas.thread_teardown_response import ThreadTeardownResponse
from stt.assemblyai_stt import AssemblyAISTT, TranscriptionResult
from .stream_service import StreamService
from modules.base_controller import BaseController


class ChatsController(BaseController):
    """Owns every chat / thread endpoint, exposed via :attr:`router`.

    The router is mounted at ``/chats`` so every route below is served as
    ``/chats/<path>``. The module owns its own service instances so
    :mod:`main` does not need to know about chat internals.
    """

    PREFIX = "/chats"

    def __init__(self) -> None:
        self.invoke_service = InvokeService()
        self.stream_service = StreamService(self.invoke_service)
        super().__init__()

    def _register_routes(self) -> None:
        router = self.router

        @router.post("/speech-to-text", response_model=TranscriptionResult)
        async def speech_to_text(
            audio: UploadFile = File(..., description="Prerecorded audio file to transcribe"),
        ) -> TranscriptionResult:
            """Transcribe an uploaded audio file with AssemblyAI prerecorded STT."""
            if not audio.filename:
                raise HTTPException(status_code=400, detail="Audio filename is required")

            content = await audio.read()
            if not content:
                raise HTTPException(status_code=400, detail="Audio file is empty")

            try:
                stt = AssemblyAISTT()
                return stt.transcribe(content)
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        @router.post("/invoke", response_model=InvokeResponse)
        async def invoke(
            payload: InvokeAgent,
            authorization: str | None = Header(default=None),
        ) -> InvokeResponse:
            """Compile the requested agent and run or resume one turn."""
            access_token = parse_authorization_header(authorization)
            return await self.invoke_service.invoke(payload, access_token=access_token)

        @router.post("/stream")
        async def stream(
            payload: InvokeAgent,
            authorization: str | None = Header(default=None),
        ) -> StreamingResponse:
            """Compile the requested agent and stream v3 projections over SSE."""
            access_token = parse_authorization_header(authorization)
            return await self.stream_service.stream(payload, access_token=access_token)

        @router.post("/cancel-stream/{thread_id}")
        async def cancel_stream(thread_id: str) -> dict[str, bool | str]:
            """Abort an in-flight /chats/stream run for this thread_id."""
            cancelled = await self.stream_service.cancel_stream(thread_id)
            if not cancelled:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active stream found for thread_id {thread_id!r}",
                )
            return {"thread_id": thread_id, "cancelled": True}

        @router.get("/get-history/{thread_id}", response_model=ThreadHistoryResponse)
        async def get_history(
            thread_id: str,
            agent_id: int | None = Query(
                default=None,
                description=(
                    "Agent whose checkpoint history to load. Defaults to the general "
                    "agent when omitted."
                ),
            ),
        ) -> ThreadHistoryResponse:
            """Return checkpoint chat history for a thread."""
            return await self.invoke_service.get_history(thread_id, agent_id=agent_id)

        @router.post(
            "/threads/{thread_id}/clear-from-last-user",
            response_model=ThreadRewindResponse,
        )
        async def clear_from_last_user(
            thread_id: str,
            agent_id: int | None = Query(
                default=None,
                description=(
                    "Agent whose checkpoint to rewind. Defaults to the general agent "
                    "when omitted."
                ),
            ),
        ) -> ThreadRewindResponse:
            """Clear checkpoint messages from the last user turn onward.

            Used by regenerate: the client keeps the latest user bubble, calls this
            endpoint, then ``POST /chats/stream`` with the same prompt. Cancels any
            active stream for the thread first.
            """
            if await self.stream_service.has_active_stream(thread_id):
                await self.stream_service.cancel_stream(thread_id)
            return await self.invoke_service.clear_from_last_user(thread_id, agent_id=agent_id)

        @router.delete("/threads/{thread_id}")
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
            an active /chats/stream run.
            """
            if await self.stream_service.has_active_stream(thread_id):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Thread has an active stream. Cancel it with "
                        f"POST /chats/cancel-stream/{thread_id} before deleting."
                    ),
                )
            return await self.invoke_service.teardown_thread(thread_id, agent_id=agent_id)
