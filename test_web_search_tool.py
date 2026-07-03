"""Standalone tester for Tavily search and web_search_tool.func.

Usage:
    uv run python test_web_search_tool.py
    uv run python test_web_search_tool.py "LangGraph releases"
    uv run python test_web_search_tool.py --tavily-only "LangGraph releases"
    uv run python test_web_search_tool.py --tool-only "LangGraph releases"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from tavily import TavilyClient

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=True)

DEFAULT_QUERY = "LangGraph recent releases"


def _check_tavily_key() -> str:
    """Validate TAVILY_API_KEY before calling the API."""
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to .env at "
            f"{_ENV_PATH} (run from repo root: deep-agents-from-scratch/)."
        )
    if key.startswith("tvly-") and len(key) < 45:
        raise RuntimeError(
            f"TAVILY_API_KEY looks truncated (length={len(key)}). "
            "Copy the full key from https://app.tavily.com/ and paste it on "
            "one line in .env with no quotes or spaces."
        )
    return key


def search_tavily(
    query: str,
    *,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict[str, Any]:
    """Run a Tavily search and return the raw API response."""
    api_key = _check_tavily_key()
    client = TavilyClient(api_key=api_key)
    return client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def _load_web_search_module():
    """Import web_search_tool without triggering tools/__init__ circular imports."""
    path = Path(__file__).resolve().parent / "tools" / "web_search_tool.py"
    spec = importlib.util.spec_from_file_location("web_search_tool", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_search_results(result: dict[str, Any]) -> None:
    """Pretty-print Tavily search hits."""
    hits = result.get("results", [])
    print(f"results: {len(hits)}")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit.get('title', '')[:80]}")
        print(f"      {hit.get('url', '')}")
        content = hit.get("content") or hit.get("raw_content") or ""
        if content:
            preview = content.replace("\n", " ")[:200]
            print(f"      {preview}...")


def test_tavily(
    query: str,
    *,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
) -> None:
    """Step 1: Tavily API only."""
    key = _check_tavily_key()
    _print_header("1. Tavily search")
    print(f"TAVILY_API_KEY loaded (length={len(key)}, prefix={key[:12]}...)")
    print(f"query: {query!r}")
    result = search_tavily(query, max_results=max_results, topic=topic)
    print(f"result: {result}")


def test_web_search_tool_func(
    query: str,
    *,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
) -> None:
    """Step 2: Full web_search_tool via .func (summarize, file write, Command)."""
    mod = _load_web_search_module()
    web_search_tool = mod.web_search_tool

    _print_header("2. web_search_tool.func")
    state: dict[str, Any] = {"messages": [], "files": {}}
    out = web_search_tool.func(
        query=query,
        state=state,
        tool_call_id=f"test-{uuid.uuid4()}",
        max_results=max_results,
        topic=topic,
    )
    if not isinstance(out, Command):
        raise TypeError(f"Expected Command, got {type(out)}")

    messages = out.update.get("messages", [])
    files = out.update.get("files", {})
    tool_msg = next(
        (m for m in messages if isinstance(m, ToolMessage)),
        None,
    )
    if tool_msg is None:
        raise RuntimeError("No ToolMessage in Command update")

    status = getattr(tool_msg, "status", "success")
    print(f"status: {status}")
    print(f"files written: {len(files)}")
    for name in files:
        print(f"  - {name}")

    print("\nToolMessage preview:")
    print(tool_msg.content[:500] if tool_msg.content else "(empty)")

    if status == "error":
        raise RuntimeError(f"web_search_tool returned error: {tool_msg.content}")

    junk_markers = ("javascript_disabled", "javascript is disabled", "no actual research")
    combined = f"{list(files)} {tool_msg.content}".lower()
    if any(marker in combined for marker in junk_markers):
        raise RuntimeError(
            "web_search_tool wrote low-value content (likely JS-disabled fetch). "
            "Check process_search_results() Tavily fallback."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Tavily search and web_search_tool independently",
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument(
        "--topic",
        choices=["general", "news", "finance"],
        default="general",
    )
    parser.add_argument("--tavily-only", action="store_true", help="Only test Tavily API")
    parser.add_argument(
        "--tool-only",
        action="store_true",
        help="Only test web_search_tool.func",
    )
    args = parser.parse_args()

    if args.tavily_only and args.tool_only:
        parser.error("Use at most one of --tavily-only and --tool-only")

    try:
        if args.tavily_only:
            test_tavily(args.query, max_results=args.max_results, topic=args.topic)
        elif args.tool_only:
            _check_tavily_key()
            test_web_search_tool_func(
                args.query,
                max_results=args.max_results,
                topic=args.topic,
            )
        else:
            test_tavily(args.query, max_results=args.max_results, topic=args.topic)
            test_web_search_tool_func(
                args.query,
                max_results=args.max_results,
                topic=args.topic,
            )
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
