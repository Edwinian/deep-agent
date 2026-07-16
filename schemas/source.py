"""Search source payloads for tool results and stream SSE chunks."""

from __future__ import annotations

from pydantic import BaseModel, Field

SOURCES_KEY = "sources"


class Source(BaseModel):
    """One web search source (Tavily-compatible)."""

    title: str
    url: str
    content: str = Field(description="Query-relevant snippet")
    score: float = Field(description="Relevance score from 0 to 1")
    raw_content: str | None = Field(
        default=None,
        description="Full cleaned page when include_raw_content is enabled",
    )
    published_date: str | None = Field(
        default=None,
        description="Publication date; typically present for topic=news",
    )
    favicon: str | None = Field(
        default=None,
        description="Favicon URL when include_favicon is enabled",
    )
