"""Content types for tool result messages in stream SSE payloads."""

from enum import StrEnum

CONTENT_TYPE_KEY = "content_type"


class ContentType(StrEnum):
    """Content format for a tool result streamed to clients."""

    TEXT = "text"
