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
from utils.summarize import Summary, summarize
from utils.tool_messages import text_tool_message

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_ID = 2006

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
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


_TASK_BRIEF_PREFIXES = (
    "research and answer",
    "return a concise",
    "with up-to-date sources",
)

_CURRENT_EVENTS_MARKERS = (
    "who won",
    "who is winning",
    "finalist",
    "finalists",
    "made it to",
    "reached the final",
    "latest",
    "today",
    "yesterday",
    "this week",
    "current",
    "live score",
    "score now",
    "world cup",
    "election result",
    "breaking",
)


def _extract_question_from_brief(text: str) -> str:
    """Prefer an explicit Question: line over a full research task brief."""
    match = re.search(
        r"(?im)^\s*question\s*:\s*(.+?)(?:\n|$)",
        text.strip(),
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_search_query(query: str) -> str:
    """Turn task briefs / verbose prompts into a concise search query."""
    cleaned = _extract_question_from_brief(query)
    lowered = cleaned.lower()
    if any(marker in lowered for marker in _TASK_BRIEF_PREFIXES) and len(cleaned) > 120:
        # Drop instructional preamble; keep the last non-empty line.
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            cleaned = lines[-1]
            cleaned = re.sub(r"(?i)^question\s*:\s*", "", cleaned).strip()
    return " ".join(cleaned.split())


def _prefers_news_topic(query: str) -> bool:
    """Heuristic: current-events / sports / live-results queries need news."""
    lowered = query.lower()
    return any(marker in lowered for marker in _CURRENT_EVENTS_MARKERS)


def run_tavily_search(
    search_query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "advanced",
    time_range: Literal["day", "week", "month", "year"] | None = None,
    include_answer: bool = True,
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content
        search_depth: Tavily depth — advanced improves relevance for factual Q&A
        time_range: Optional recency filter (day/week/month/year)
        include_answer: Ask Tavily for a grounded answer summary

    Returns:
        Search results dictionary

    Raises:
        MissingAPIKeyError: If TAVILY_API_KEY is unset.
        InvalidAPIKeyError: If Tavily rejects the API key.
        BadRequestError, ForbiddenError, UsageLimitExceededError: Other Tavily failures.
    """
    client = _get_tavily_client()
    kwargs: dict[str, Any] = {
        "max_results": max_results,
        "include_raw_content": include_raw_content,
        "topic": topic,
        "search_depth": search_depth,
        "include_answer": include_answer,
    }
    if time_range:
        kwargs["time_range"] = time_range
    elif topic == "news":
        # News topic supports days; keep results recent for live events.
        kwargs["days"] = 7

    try:
        return client.search(search_query, **kwargs)
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
    """Summarize webpage content using summarize."""
    try:
        result = summarize(webpage_content)
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
                "published_date": result.get("published_date"),
                "score": result.get("score"),
            })

    return processed_results


@tool(parse_docstring=True)
def web_search_tool(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"] | None,
        InjectedToolArg,
    ] = None,
    time_range: Annotated[
        Literal["day", "week", "month", "year"] | None,
        InjectedToolArg,
    ] = None,
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Concise search query (not a full research brief). Prefer the
            user's factual question, e.g. "FIFA World Cup 2026 finalists".
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        max_results: Maximum number of results to return (default: 5)
        topic: Topic filter - 'general', 'news', or 'finance'. Leave unset to
            auto-select 'news' for current-events / sports / live-result queries.
        time_range: Optional recency filter: 'day', 'week', 'month', or 'year'.
            Defaults to 'week' when topic is news.

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    search_query = normalize_search_query(query)
    resolved_topic = topic or (
        "news" if _prefers_news_topic(search_query) else "general"
    )
    resolved_time_range = time_range
    if resolved_time_range is None and resolved_topic == "news":
        resolved_time_range = "week"

    # Bias the engine toward today's facts without replacing the user query.
    dated_query = f"{search_query} (as of {get_today_str()})"

    try:
        search_results = run_tavily_search(
            dated_query,
            max_results=max_results,
            topic=resolved_topic,
            include_raw_content=True,
            time_range=resolved_time_range,
            include_answer=True,
        )
        processed_results = process_search_results(search_results)
    except Exception as exc:
        error_text = _format_tavily_error(exc)
        logger.error("web_search_tool failed for query=%r: %s", search_query, error_text)
        return Command(
            update={
                "messages": [
                    text_tool_message(
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
                    text_tool_message(
                        f"No search results returned for '{search_query}'. Try a different query.",
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
        published = result.get("published_date") or "unknown"

        file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {search_query}
**Published:** {published}
**Date searched:** {get_today_str()}

## Content
{result['raw_content'] if result['raw_content'] else 'No content available'}
"""

        _write_file_to_state(files, filename, file_content)
        saved_files.append(filename)
        preview = result["summary"]
        if len(preview) > 300:
            preview = preview[:300] + "..."
        summaries.append(
            f"- {filename} (published: {published}): {preview}"
        )

    tavily_answer = (search_results.get("answer") or "").strip()
    answer_block = ""
    if tavily_answer:
        answer_block = f"""
Tavily answer (grounded in sources; verify against files):
{tavily_answer}
"""

    # Create minimal summary for tool message - focus on what was collected
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{search_query}' [topic={resolved_topic}]:
{answer_block}
{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Prefer recent published dates. Use read_file() for full details when needed.
⚠️ Do not conclude an event has not happened yet if recent sources say otherwise."""

    return Command(
        update={
            "files": files,
            "messages": [
                text_tool_message(summary_text, tool_call_id),
            ],
        }
    )
