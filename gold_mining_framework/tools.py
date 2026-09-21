"""Tools available to gold mining framework agents."""

import os

from langchain_core.tools import tool
from tavily import TavilyClient  # type: ignore[import-untyped]


def _format_search_results(payload: dict) -> str:
    """Turn a Tavily response into compact text for the agent.

    Args:
        payload: Raw Tavily search response.

    Returns:
        A readable summary of the answer and ranked results.
    """
    lines: list[str] = []
    answer = payload.get("answer")
    if answer:
        lines.append(f"Answer: {answer}")
        lines.append("")

    results = payload.get("results") or []
    if not results:
        lines.append("No search results found.")
        return "\n".join(lines)

    lines.append("Results:")
    for i, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or ""
        content = (result.get("content") or "").strip()
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if content:
            lines.append(f"   {content}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """Search the web for current information on a topic.

    Use this to gather market context and to check Google Trends for candidate
    categories, subcategories, niches, and sub-niches (worldwide, since 2004).

    Args:
        query: Search query. Be specific. For trend checks, use forms such as
            "<keyword> Google Trends worldwide since 2004".

    Returns:
        Search results including an optional short answer and source snippets.
    """
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError(
            "Tavily is not configured. Set TAVILY_API_KEY in the repo-root .env file."
        )
    client = TavilyClient()
    payload = client.search(
        query,
        max_results=5,
        include_answer=True,
        topic="general",
    )
    return _format_search_results(payload)
