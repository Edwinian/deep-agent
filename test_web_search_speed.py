"""Unit tests for web search latency helpers (no Tavily / LLM calls)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from langchain_core.messages import ToolMessage


def _load_module():
    path = Path(__file__).resolve().parent / "tools" / "web_search" / "web_search_tool.py"
    spec = importlib.util.spec_from_file_location("web_search_tool", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_is_simple_query_short_news() -> None:
    mod = _load_module()
    assert mod._is_simple_query("latest football news")
    assert mod._prefers_news_topic("latest football news")


def test_is_simple_query_rejects_complex() -> None:
    mod = _load_module()
    assert not mod._is_simple_query(
        "Compare OpenAI versus Anthropic approaches to AI safety in detail"
    )


def test_max_searches_simple_news_is_one() -> None:
    mod = _load_module()
    assert mod._max_web_searches_for_query("latest football news") == 1


def test_max_searches_simple_non_news_is_two() -> None:
    mod = _load_module()
    assert mod._max_web_searches_for_query("capital of France") == 2


def test_tavily_options_news_uses_fast_without_raw_content() -> None:
    mod = _load_module()
    opts = mod._tavily_search_options("news", is_simple=True)
    assert opts["search_depth"] == "fast"
    assert opts["include_raw_content"] is False


def test_needs_llm_summarize_only_for_large_content() -> None:
    mod = _load_module()
    assert not mod._needs_llm_summarize("short snippet")
    assert mod._needs_llm_summarize("x" * (mod.SUMMARIZE_CHAR_THRESHOLD + 1))


def test_count_prior_web_searches() -> None:
    mod = _load_module()
    messages = [
        ToolMessage(content="found stuff", tool_call_id="a", name="web_search_tool"),
        ToolMessage(content="", tool_call_id="b", name="web_search_tool"),
        ToolMessage(content="other", tool_call_id="c", name="think_tool"),
    ]
    assert mod._count_prior_web_searches(messages) == 1


def test_process_search_results_skips_llm_for_snippets() -> None:
    mod = _load_module()
    statuses: list[str] = []

    results = {
        "results": [
            {
                "url": "https://example.com/a",
                "title": "Example A",
                "content": "A" * 120,
            },
            {
                "url": "https://example.com/b",
                "title": "Example B",
                "content": "B" * 120,
            },
        ]
    }

    processed = mod.process_search_results(
        results,
        on_status=statuses.append,
    )

    assert len(processed) == 2
    assert any("Using snippet" in line for line in statuses)
    assert not any("Summarizing" in line for line in statuses)
