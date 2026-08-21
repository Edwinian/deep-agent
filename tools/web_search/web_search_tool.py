"""Web search tool and content processing utilities for research agents."""

import base64
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain.tools import ToolRuntime
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
from schemas.source import Source
from utils.retry import is_transient_exception, retry_with_backoff
from utils.summarize import Summary, summarize
from utils.tool_messages import text_tool_message
from utils.tool_quality_retry import run_with_quality_retry

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_ID = 2006
SOURCES_FILE = "/_sources.json"
MAX_QUALITY_REWRITES = 1

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


def _merge_source_dicts(
    existing: list[dict[str, Any]],
    incoming: list[Source] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedupe sources by URL, preferring newer entries."""
    by_url: dict[str, dict[str, Any]] = {}
    for item in existing:
        url = str(item.get("url") or "")
        if url:
            by_url[url] = item
    for item in incoming:
        payload = item.model_dump() if isinstance(item, Source) else dict(item)
        url = str(payload.get("url") or "")
        if url:
            by_url[url] = payload
    return list(by_url.values())


def _load_sources_file(files: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ``/_sources.json`` from virtual files if present."""
    entry = files.get(SOURCES_FILE) or files.get(SOURCES_FILE.lstrip("/"))
    if entry is None:
        return []
    raw: Any = entry
    if isinstance(entry, dict) and "content" in entry:
        content = entry.get("content")
        if isinstance(content, list):
            raw = "\n".join(str(line) for line in content)
        else:
            raw = content
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _write_sources_file(files: dict[str, Any], sources: list[Source]) -> None:
    """Persist merged sources for later run_finished / HITL turns."""
    merged = _merge_source_dicts(_load_sources_file(files), sources)
    _write_file_to_state(
        files,
        SOURCES_FILE,
        json.dumps(merged, ensure_ascii=False, indent=2),
    )

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

_COMPLEX_QUERY_MARKERS = (
    "compare",
    "versus",
    " vs ",
    "difference between",
    "pros and cons",
    "advantages and disadvantages",
    "step by step",
    "how to",
    "analyze",
    "analysis",
    "history of",
    "timeline",
    "research ",
)

SUMMARIZE_CHAR_THRESHOLD = 2500
MAX_WEB_SEARCHES_SIMPLE = 2
MAX_WEB_SEARCHES_NORMAL = 3
MAX_WEB_SEARCHES_COMPLEX = 5
_SIMPLE_QUERY_MAX_WORDS = 15


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


def _is_simple_query(query: str) -> bool:
    """True for short single-topic questions that need at most 1–2 searches."""
    normalized = normalize_search_query(query)
    if not normalized:
        return False
    if len(normalized.split()) > _SIMPLE_QUERY_MAX_WORDS:
        return False
    lowered = normalized.lower()
    if any(marker in lowered for marker in _COMPLEX_QUERY_MARKERS):
        return False
    return normalized.count("?") <= 1


def _max_web_searches_for_query(query: str) -> int:
    """Return enforced web_search_tool budget for this query."""
    normalized = normalize_search_query(query)
    if _is_simple_query(normalized):
        if _prefers_news_topic(normalized):
            return 1
        return MAX_WEB_SEARCHES_SIMPLE
    lowered = normalized.lower()
    if any(marker in lowered for marker in _COMPLEX_QUERY_MARKERS) or len(
        normalized.split()
    ) > 25:
        return MAX_WEB_SEARCHES_COMPLEX
    return MAX_WEB_SEARCHES_NORMAL


def _count_prior_web_searches(messages: list[Any] | None) -> int:
    """Count successful prior web_search_tool calls in agent message history."""
    count = 0
    for message in messages or []:
        if isinstance(message, ToolMessage) and str(message.name or "") == "web_search_tool":
            if str(message.content or "").strip():
                count += 1
    return count


def _tavily_search_options(
    topic: Literal["general", "news", "finance"],
    *,
    is_simple: bool,
) -> dict[str, Any]:
    """Pick faster Tavily settings for news and simple queries."""
    if topic == "news":
        return {
            "search_depth": "fast",
            "include_raw_content": False,
            "include_answer": True,
        }
    if is_simple:
        return {
            "search_depth": "basic",
            "include_raw_content": False,
            "include_answer": True,
        }
    return {
        "search_depth": "advanced",
        "include_raw_content": True,
        "include_answer": True,
    }


def run_tavily_search(
    search_query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "advanced",
    time_range: Literal["day", "week", "month", "year"] = "week",
    include_answer: bool = True,
    include_favicon: bool = True,
    on_retry: Callable[[BaseException, int, float], None] | None = None,
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content
        search_depth: Tavily depth — advanced improves relevance for factual Q&A
        time_range: Recency filter (day/week/month/year). Defaults to year.
        include_answer: Ask Tavily for a grounded answer summary
        include_favicon: Include favicon URLs for each source
        on_retry: Optional callback ``(exc, attempt, delay_seconds)`` before each retry

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
        "include_favicon": include_favicon,
    }
    if time_range:
        kwargs["time_range"] = time_range
    elif topic == "news":
        # News topic supports days; keep results recent for live events.
        kwargs["days"] = 7

    def _search() -> dict:
        return client.search(search_query, **kwargs)

    try:
        return retry_with_backoff(
            _search,
            retry_on=lambda exc: is_transient_exception(exc)
            and not isinstance(exc, _TAVILY_API_ERRORS),
            on_retry=on_retry,
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


def _snippet_summary(text: str, *, max_len: int = 500) -> str:
    """Return a compact preview without calling the summarizer LLM."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len].rsplit(" ", 1)[0]
    return f"{truncated}..."


def _needs_llm_summarize(content: str) -> bool:
    """Only summarize very large page bodies; snippets are enough otherwise."""
    return len(content) > SUMMARIZE_CHAR_THRESHOLD


def _summary_from_snippet(
    raw_content: str,
    *,
    title: str,
    tavily_snippet: str,
) -> Summary:
    """Build a Summary from Tavily snippet / short content (no LLM)."""
    preview_source = (tavily_snippet or raw_content).strip()
    return Summary(
        filename=_normalize_virtual_path(_filename_from_title(title)),
        summary=_snippet_summary(preview_source),
    )


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


def _fetch_result_content(result: dict, *, client: httpx.Client) -> str:
    """Resolve the best available text for one Tavily hit."""
    url = result["url"]
    tavily_content = _tavily_result_content(result)
    if tavily_content and not _is_low_value_content(tavily_content):
        return tavily_content

    try:
        response = client.get(url)
        if response.status_code == 200:
            fetched = markdownify(response.text)
            if not _is_low_value_content(fetched):
                return fetched
            if tavily_content:
                return tavily_content
        elif tavily_content:
            return tavily_content
    except (httpx.TimeoutException, httpx.RequestError):
        if tavily_content:
            return tavily_content

    return result.get("content", "No content available.")


def process_search_results(
    results: dict,
    *,
    on_status: Callable[[str], None] | None = None,
) -> list[dict]:
    """Process search results into file-ready records with raw content.

    Prefers Tavily-extracted content over re-fetching URLs. Skips LLM
    summarization for short snippets and parallelizes summarization for large
    pages only.

    Args:
        results: Tavily search results dictionary
        on_status: Optional callback for client-visible progress

    Returns:
        List of processed results with filenames and raw content
    """
    hits = results.get("results", [])
    total = len(hits)
    if total == 0:
        return []

    if on_status is not None:
        on_status(f"Processing {total} search result{'s' if total != 1 else ''}…")

    fetched: list[tuple[dict, str]] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for index, result in enumerate(hits, start=1):
            if on_status is not None:
                on_status(f"Fetching content for result {index}/{total}…")
            fetched.append((result, _fetch_result_content(result, client=client)))

    summaries: list[Summary | None] = [None] * total
    to_summarize: list[tuple[int, str, str]] = []

    for index, (result, raw_content) in enumerate(fetched):
        title = str(result.get("title") or "")
        snippet = str(result.get("content") or "")
        if _needs_llm_summarize(raw_content):
            to_summarize.append((index, raw_content, title))
            continue
        if on_status is not None:
            on_status(f"Using snippet for result {index + 1}/{total}…")
        summaries[index] = _summary_from_snippet(
            raw_content,
            title=title,
            tavily_snippet=snippet,
        )

    if to_summarize:
        summarize_total = len(to_summarize)
        if on_status is not None:
            on_status(
                f"Summarizing {summarize_total} large result"
                f"{'s' if summarize_total != 1 else ''}…"
            )
        workers = min(summarize_total, 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    summarize_webpage_content,
                    content,
                    title=title,
                ): idx
                for idx, content, title in to_summarize
            }
            completed = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                summaries[idx] = future.result()
                completed += 1
                if on_status is not None:
                    on_status(f"Summarized result {completed}/{summarize_total}…")

    processed_results: list[dict] = []
    for (result, raw_content), summary_obj in zip(fetched, summaries):
        if summary_obj is None:
            continue

        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = _normalize_virtual_path(f"{name}_{uid}{ext}")

        processed_results.append({
            "url": result["url"],
            "title": result["title"],
            "summary": summary_obj.summary,
            "filename": summary_obj.filename,
            "raw_content": raw_content,
            "content": result.get("content") or "",
            "published_date": result.get("published_date"),
            "score": result.get("score"),
            "favicon": result.get("favicon"),
        })

    return processed_results


def _sources_from_processed(processed_results: list[dict]) -> list[Source]:
    """Map processed Tavily hits into stream Source models."""
    sources: list[Source] = []
    for result in processed_results:
        score = result.get("score")
        try:
            score_value = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_value = 0.0
        snippet = (result.get("content") or result.get("summary") or "").strip()
        sources.append(
            Source(
                title=str(result.get("title") or "Untitled"),
                url=str(result.get("url") or ""),
                content=snippet,
                score=score_value,
                raw_content=result.get("raw_content"),
                published_date=result.get("published_date"),
                favicon=result.get("favicon"),
            )
        )
    return sources

@tool(parse_docstring=True)
def web_search_tool(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"] | None,
        InjectedToolArg,
    ] = None,
    time_range: Annotated[
        Literal["day", "week", "month", "year"],
        InjectedToolArg,
    ] = "week",
) -> Command:
    """Search web and save detailed results to files while returning minimal context.

    Performs web search and saves full content to files for context offloading.
    Returns only essential information to help the agent decide on next steps.

    Args:
        query: Concise search query (not a full research brief). Prefer the
            user's factual question, e.g. "FIFA World Cup 2026 finalists".
        state: Injected agent state for file storage
        tool_call_id: Injected tool call identifier
        runtime: Injected tool runtime for streaming progress to the client
        max_results: Maximum number of results to return (default: 5)
        topic: Topic filter - 'general', 'news', or 'finance'. Leave unset to
            auto-select 'news' for current-events / sports / live-result queries.
        time_range: Recency filter: 'day', 'week', 'month', or 'year' (default: year).

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    search_query = normalize_search_query(query)
    resolved_topic = topic or (
        "news" if _prefers_news_topic(search_query) else "general"
    )
    is_simple = _is_simple_query(search_query)
    search_budget = _max_web_searches_for_query(search_query)
    prior_searches = _count_prior_web_searches(state.get("messages"))
    if prior_searches >= search_budget:
        return Command(
            update={
                "messages": [
                    text_tool_message(
                        f"Search limit reached ({search_budget} call"
                        f"{'s' if search_budget != 1 else ''} for this question). "
                        "Answer from the files and sources already saved — "
                        "do not call web_search_tool again.",
                        tool_call_id,
                        status="error",
                    ),
                ],
            }
        )

    # Bias the engine toward today's facts without replacing the user query.
    tavily_options = _tavily_search_options(resolved_topic, is_simple=is_simple)

    def _status(message: str) -> None:
        runtime.emit_output_delta(message)

    def _on_search_retry(exc: BaseException, attempt: int, delay: float) -> None:
        _status(
            f"Search failed (attempt {attempt}), retrying in {delay:.1f}s…"
        )

    # Last successful search payload retained for file offloading after quality pass.
    search_bundle: dict[str, Any] = {"results": None, "processed": None, "query": search_query}

    def _execute_search(candidate_query: str) -> str:
        normalized = normalize_search_query(candidate_query)
        dated_query = f"{normalized} (as of {get_today_str()})"
        _status(f'Searching the web for "{normalized}"…')
        raw = run_tavily_search(
            dated_query,
            max_results=max_results,
            topic=resolved_topic,
            time_range=time_range,
            include_favicon=True,
            on_retry=_on_search_retry,
            **tavily_options,
        )
        processed = process_search_results(raw, on_status=_status)
        search_bundle["results"] = raw
        search_bundle["processed"] = processed
        search_bundle["query"] = normalized
        if not processed:
            return ""
        previews: list[str] = []
        for item in processed:
            preview = str(item.get("summary") or item.get("title") or "")
            if len(preview) > 300:
                preview = preview[:300] + "..."
            previews.append(f"- {item.get('title')}: {preview}")
        answer = (raw.get("answer") or "").strip()
        parts = [f"Query: {normalized}", *previews]
        if answer:
            parts.insert(1, f"Answer: {answer}")
        return "\n".join(parts)

    try:
        quality = run_with_quality_retry(
            search_query,
            _execute_search,
            tool_name="web_search_tool",
            max_retries=MAX_QUALITY_REWRITES,
            on_status=_status,
        )
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

    processed_results = search_bundle.get("processed") or []
    search_results = search_bundle.get("results") or {}
    search_query = str(search_bundle.get("query") or search_query)

    if not quality.ok or not processed_results:
        return Command(
            update={
                "messages": [
                    text_tool_message(
                        quality.error
                        or f"No search results returned for '{search_query}'. Try a different query.",
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

    sources = _sources_from_processed(processed_results)
    _write_sources_file(files, sources)

    return Command(
        update={
            "files": files,
            "messages": [
                text_tool_message(
                    summary_text,
                    tool_call_id,
                    sources=sources,
                ),
            ],
        }
    )
