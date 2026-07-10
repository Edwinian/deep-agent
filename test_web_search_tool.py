"""Standalone tester for Tavily search and web_search_tool integration paths.

Usage:
    uv run python test_web_search_tool.py
    uv run python test_web_search_tool.py "LangGraph releases"
    uv run python test_web_search_tool.py --tavily-only "LangGraph releases"
    uv run python test_web_search_tool.py --func-only "LangGraph releases"
    uv run python test_web_search_tool.py --tool-node-only "LangGraph releases"
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
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from tavily import TavilyClient
from typing_extensions import Annotated, TypedDict

from deepagents.backends.utils import create_file_data

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
    path = Path(__file__).resolve().parent / "tools" / "web_search" / "web_search_tool.py"
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


def _assert_web_search_success(
    tool_msg: ToolMessage,
    files: dict[str, Any],
    *,
    label: str,
) -> None:
    """Validate a successful web_search_tool ToolMessage and file writes."""
    status = getattr(tool_msg, "status", "success")
    print(f"status: {status}")
    print(f"files written: {len(files)}")
    for name in files:
        print(f"  - {name}")

    print("\nToolMessage preview:")
    print(tool_msg.content[:500] if tool_msg.content else "(empty)")

    if status == "error":
        raise RuntimeError(f"{label} returned error: {tool_msg.content}")

    junk_markers = ("javascript_disabled", "javascript is disabled", "no actual research")
    combined = f"{list(files)} {tool_msg.content}".lower()
    if any(marker in combined for marker in junk_markers):
        raise RuntimeError(
            f"{label} wrote low-value content (likely JS-disabled fetch). "
            "Check process_search_results() Tavily fallback."
        )


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
    print_search_results(result)


def test_web_search_tool_func(
    query: str,
    *,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
) -> None:
    """Step 2: Full web_search_tool via .func (bypasses LangGraph injection)."""
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

    _assert_web_search_success(tool_msg, files, label="web_search_tool.func")


def test_tool_node_with_filedata(
    query: str,
    *,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
) -> None:
    """Step 3: ToolNode with deepagents FileData state (production agent path)."""
    mod = _load_web_search_module()
    web_search_tool = mod.web_search_tool

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        files: dict[str, Any]
        todos: list[Any]

    _print_header("3. ToolNode + FileData state (agent-like)")
    tool_call_id = f"test-{uuid.uuid4()}"
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": web_search_tool.name,
                "args": {"query": query, "max_results": max_results, "topic": topic},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )
    # Mimics state after write_file in general-agent before research subagent runs.
    state: State = {
        "messages": [ai],
        "files": {"/user_request.txt": create_file_data("prior agent file")},
        "todos": [],
    }
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    builder = StateGraph(State)
    builder.add_node("tools", ToolNode([web_search_tool]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    result = builder.compile().invoke(state, config=config)

    tool_msg = next(
        (
            m
            for m in result.get("messages", [])
            if isinstance(m, ToolMessage) and m.name == web_search_tool.name
        ),
        None,
    )
    if tool_msg is None:
        raise RuntimeError("No web_search_tool ToolMessage in ToolNode result")

    _assert_web_search_success(
        tool_msg,
        result.get("files", {}),
        label="ToolNode + FileData",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Tavily search and web_search_tool integration paths",
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument(
        "--topic",
        choices=["general", "news", "finance"],
        default="general",
    )
    parser.add_argument("--tavily-only", action="store_true", help="Only step 1")
    parser.add_argument(
        "--func-only",
        action="store_true",
        help="Only step 2 (web_search_tool.func)",
    )
    parser.add_argument(
        "--tool-only",
        action="store_true",
        help="Alias for --func-only",
    )
    parser.add_argument(
        "--tool-node-only",
        action="store_true",
        help="Only step 3 (ToolNode + FileData state)",
    )
    args = parser.parse_args()

    if args.tool_only:
        args.func_only = True

    only_flags = sum([args.tavily_only, args.func_only, args.tool_node_only])
    if only_flags > 1:
        parser.error("Use at most one of --tavily-only, --func-only, --tool-node-only")

    run_kwargs = {
        "max_results": args.max_results,
        "topic": args.topic,
    }

    try:
        if args.tavily_only:
            test_tavily(args.query, **run_kwargs)
        elif args.func_only:
            _check_tavily_key()
            test_web_search_tool_func(args.query, **run_kwargs)
        elif args.tool_node_only:
            _check_tavily_key()
            test_tool_node_with_filedata(args.query, **run_kwargs)
        else:
            test_tavily(args.query, **run_kwargs)
            test_web_search_tool_func(args.query, **run_kwargs)
            test_tool_node_with_filedata(args.query, **run_kwargs)
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
