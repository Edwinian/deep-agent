"""Typed schema for POST /threads/{thread_id}/clear-from-last-user."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ThreadRewindResponse(BaseModel):
    """Result of clearing checkpoint messages from the last user turn onward."""

    thread_id: str
    message: str = Field(
        description="Text of the most recent user message (for regenerate/stream).",
    )
    removed_count: int = Field(
        description="Number of checkpoint messages removed (last user + after).",
    )
    remaining_count: int = Field(
        description="Number of checkpoint messages kept before the last user turn.",
    )
