"""Typed schema for DELETE /threads/{thread_id}."""

from __future__ import annotations

from pydantic import BaseModel


class ThreadTeardownResponse(BaseModel):
    """Result of deleting a conversation thread and its resources."""

    thread_id: str
    checkpoint_deleted: bool
    sandbox_deleted: bool
