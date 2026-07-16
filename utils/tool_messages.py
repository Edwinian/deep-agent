"""Helpers for building tool result messages."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import ToolMessage

from schemas.content_type import CONTENT_TYPE_KEY, ContentType
from schemas.source import SOURCES_KEY, Source


def text_tool_message(
    content: str,
    tool_call_id: str,
    *,
    status: Literal["success", "error"] | None = None,
    sources: list[Source] | None = None,
) -> ToolMessage:
    """Build a text ToolMessage tagged for client stream rendering."""
    additional_kwargs: dict[str, Any] = {CONTENT_TYPE_KEY: ContentType.TEXT}
    if sources:
        additional_kwargs[SOURCES_KEY] = [source.model_dump() for source in sources]
    kwargs: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "additional_kwargs": additional_kwargs,
    }
    if status is not None:
        kwargs["status"] = status
    return ToolMessage(content, **kwargs)
