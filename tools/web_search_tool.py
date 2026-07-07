"""Web search tool and content processing utilities for research agents."""

import base64
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from typing_extensions import Annotated, Literal

from deepagents.backends.utils import create_file_data
from schemas.content_type import CONTENT_TYPE_KEY, ContentType
from tools.summarize_tool import Summary, summarize_content

logger = logging.getLogger(__name__)


def _text_tool_message(
    content: str,
    tool_call_id: str,
    *,
    status: Literal["success", "error"] | None = None,
) -> ToolMessage:
    """Build a text ToolMessage tagged for client stream rendering."""
    kwargs: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "additional_kwargs": {CONTENT_TYPE_KEY: ContentType.TEXT},
    }
    if status is not None:
        kwargs["status"] = status
    return ToolMessage(content, **kwargs)

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH, override=True)

_tavily_client: TavilyClient | None = None

_TAVILY_API_ERRORS = (
    MissingAPIKeyError,
    InvalidAPIKeyError,
    BadRequestError,
    ForbiddenError,
    UsageLimitExceededError,
)


def _tavily_api_key() -> str | None:
    """Return a trimmed Tavily API key from the environment."""
    key = os.getenv("TAVILY_API_KEY")
    if key is None:
        return None
    key = key.strip()
    return key or None


def _get_tavily_client() -> TavilyClient:
    """Create or reuse a Tavily client after validating configuration."""
    global _tavily_client
    api_key = _tavily_api_key()
    if not api_key:
        raise MissingAPIKeyError()
    if _tavily_client is None or _tavily_client.api_key != api_key:
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _format_tavily_error(exc: Exception) -> str:
    """Turn Tavily failures into actionable tool error text."""
    if isinstance(exc, MissingAPIKeyError):
        return (
            "Tavily search is not configured: TAVILY_API_KEY is missing. "
            f"Set it in {_ENV_PATH} and restart the server."
        )
    if isinstance(exc, InvalidAPIKeyError):
        key = _tavily_api_key() or ""
        hint = ""
        if key.startswith("tvly-") and len(key) < 45:
            hint = (
                " The key looks truncated (too short). Copy the full key "
                "from https://app.tavily.com/ and paste it on one line in .env."
            )
        return f"Tavily rejected TAVILY_API_KEY as invalid or unauthorized.{hint}"
    if isinstance(exc, _TAVILY_API_ERRORS):
        return f"Tavily API error: {exc}"
    return f"{type(exc).__name__}: {exc}"

def _is_deepagents_file_entry(file_entry: Any) -> bool:
    """Return True if a state files entry uses deepagents FileData format."""
    return isinstance(file_entry, dict) and "content" in file_entry


def _state_uses_deepagents_files(files: dict[str, Any]) -> bool:
    """Detect whether agent state stores files as deepagents FileData objects."""
    return not files or any(_is_deepagents_file_entry(entry) for entry in files.values())


def _normalize_virtual_path(filename: str) -> str:
    """Ensure paths match deepagents virtual FS convention (absolute, leading `/`)."""
    path = filename.strip().replace("\\", "/")
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _write_file_to_state(files: dict[str, Any], filename: str, content: str) -> None:
    """Write a file entry compatible with scratch and create_deep_agent state."""
    path = _normalize_virtual_path(filename)
    if _state_uses_deepagents_files(files):
        files[path] = create_file_data(content)
    else:
        files[path] = content

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def run_tavily_search(
    search_query: str, 
    max_results: int = 1, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary

    Raises:
        MissingAPIKeyError: If TAVILY_API_KEY is unset.
        InvalidAPIKeyError: If Tavily rejects the API key.
        BadRequestError, ForbiddenError, UsageLimitExceededError: Other Tavily failures.
    """
    client = _get_tavily_client()
    try:
        return client.search(
            search_query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except _TAVILY_API_ERRORS:
        logger.exception("Tavily search failed for query=%r", search_query)
        raise
    except Exception:
        logger.exception("Unexpected error during Tavily search for query=%r", search_query)
        raise

def _filename_from_title(title: str) -> str:
    """Derive a safe markdown filename from a page title."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{slug[:60] or 'search_result'}.md"


def summarize_webpage_content(webpage_content: str, *, title: str = "") -> Summary:
    """Summarize webpage content using summarize_tool."""
    try:
        result = summarize_content(webpage_content)
        return Summary(
            filename=_normalize_virtual_path(result.filename),
            summary=result.summary,
        )
    except Exception:
        logger.exception("Summarization failed for webpage content")
        return Summary(
            filename=_normalize_virtual_path(_filename_from_title(title)),
            summary=webpage_content,
        )


_JS_ERROR_MARKERS = (
    "javascript is disabled",
    "enable javascript",
    "javascript required",
    "requires javascript",
)


def _tavily_result_content(result: dict) -> str:
    """Return the best text Tavily already extracted for a hit."""
    return (result.get("raw_content") or result.get("content") or "").strip()


def _is_low_value_content(text: str) -> bool:
    """True when page text is empty or a known client-side error stub."""
    if not text or len(text) < 80:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _JS_ERROR_MARKERS)


def process_search_results(results: dict) -> list[dict]:
    """Process search results into file-ready records with raw content.

    Prefers Tavily-extracted content (raw_content/content) over re-fetching URLs.
    Many sites return JS-disabled stubs to bare httpx clients even when Tavily
    already has usable text.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with filenames and raw content
    """
    processed_results = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for result in results.get("results", []):
            url = result["url"]
            tavily_content = _tavily_result_content(result)
            raw_content = ""

            if tavily_content and not _is_low_value_content(tavily_content):
                raw_content = tavily_content
            else:
                try:
                    response = client.get(url)
                    if response.status_code == 200:
                        fetched = markdownify(response.text)
                        if not _is_low_value_content(fetched):
                            raw_content = fetched
                        elif tavily_content:
                            raw_content = tavily_content
                    elif tavily_content:
                        raw_content = tavily_content
                except (httpx.TimeoutException, httpx.RequestError):
                    if tavily_content:
                        raw_content = tavily_content

            if not raw_content:
                raw_content = result.get("content", "No content available.")

            summary_obj = summarize_webpage_content(
                raw_content,
                title=result.get("title", ""),
            )

            # uniquify file names
            uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
            name, ext = os.path.splitext(summary_obj.filename)
            summary_obj.filename = _normalize_virtual_path(f"{name}_{uid}{ext}")

            processed_results.append({
                "url": result["url"],
                "title": result["title"],
                "summary": summary_obj.summary,
                "filename": summary_obj.filename,
                "raw_content": raw_content,
            })

    return processed_results


@tool(parse_docstring=True)
def web_search_tool(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Search query to execute
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    try:
        search_results = run_tavily_search(
            query,
            max_results=max_results,
            topic=topic,
            include_raw_content=True,
        )
        processed_results = process_search_results(search_results)
    except Exception as exc:
        error_text = _format_tavily_error(exc)
        logger.error("web_search_tool failed for query=%r: %s", query, error_text)
        return Command(
            update={
                "messages": [
                    _text_tool_message(
                        error_text,
                        tool_call_id,
                        status="error",
                    )
                ],
            }
        )

    if not processed_results:
        return Command(
            update={
                "messages": [
                    _text_tool_message(
                        f"No search results returned for '{query}'. Try a different query.",
                        tool_call_id,
                        status="error",
                    )
                ],
            }
        )

    # Save each result to a file and prepare summary
    files = state.get("files", {})
    saved_files = []
    summaries = []

    for result in processed_results:
        filename = result["filename"]

        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_today_str()}

## Content
{result['raw_content'] if result['raw_content'] else 'No content available'}
"""

        _write_file_to_state(files, filename, file_content)
        saved_files.append(filename)
        preview = result["summary"]
        if len(preview) > 300:
            preview = preview[:300] + "..."
        summaries.append(f"- {filename}: {preview}")

    # Create minimal summary for tool message - focus on what was collected
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

    return Command(
        update={
            "files": files,
            "messages": [
                _text_tool_message(summary_text, tool_call_id),
            ],
        }
    )
