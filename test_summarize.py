"""Standalone tester for summarize_tool.

Usage:
    uv run python test_summarize.py
    uv run python test_summarize.py "Your text to summarize here"
    uv run python test_summarize.py --func-only "Your text"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=True)

DEFAULT_CONTENT = (
    "LangGraph is a library for building stateful, multi-actor LLM applications. "
    "Latest release: June 29, 2026. Key features include checkpointing, "
    "streaming, and human-in-the-loop workflows."
)


def _check_xai_key() -> None:
    key = (os.getenv("XAI_API_KEY") or "").strip()
    if not key or key == "your_xai_api_key_here":
        raise RuntimeError(
            "XAI_API_KEY is not set. Add it to .env at "
            f"{_ENV_PATH} (summarize_tool uses xai:grok-3-mini)."
        )


def _load_summarize_module():
    """Import summarize_tool without triggering tools/__init__ circular imports."""
    path = Path(__file__).resolve().parent / "tools" / "summarize" / "summarize_tool.py"
    spec = importlib.util.spec_from_file_location("summarize_tool", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def test_summarize_content(mod, content: str) -> None:
    """Step 1: Call summarize_content directly."""
    _print_header("1. summarize_content")
    result = mod.summarize_content(content)
    print(f"filename: {result.filename}")
    print(f"summary: {result.summary}")


def test_summarize_tool_func(mod, content: str) -> None:
    """Step 2: Call summarize_tool via .func (LangChain tool path)."""
    _print_header("2. summarize_tool.func")
    result = mod.summarize_tool.func(content=content)
    print(f"type: {type(result).__name__}")
    print(f"filename: {result.filename}")
    print(f"summary: {result.summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test summarize_tool independently")
    parser.add_argument("content", nargs="?", default=DEFAULT_CONTENT)
    parser.add_argument(
        "--func-only",
        action="store_true",
        help="Only test summarize_tool.func",
    )
    args = parser.parse_args()

    try:
        _check_xai_key()
        mod = _load_summarize_module()
        print(f"content preview: {args.content[:120]}...")

        if args.func_only:
            test_summarize_tool_func(mod, args.content)
        else:
            test_summarize_content(mod, args.content)
            test_summarize_tool_func(mod, args.content)
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
