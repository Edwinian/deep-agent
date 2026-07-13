"""Standalone tester for summarize.

Usage:
    uv run python test_summarize.py
    uv run python test_summarize.py "Your text to summarize here"
"""

from __future__ import annotations

import argparse
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
            f"{_ENV_PATH} (summarize uses xai:grok-4.3)."
        )


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def test_summarize(content: str) -> None:
    """Call summarize directly."""
    from utils.summarize import summarize

    _print_header("summarize")
    result = summarize(content)
    print(f"filename: {result.filename}")
    print(f"summary: {result.summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test summarize independently")
    parser.add_argument("content", nargs="?", default=DEFAULT_CONTENT)
    args = parser.parse_args()

    try:
        _check_xai_key()
        print(f"content preview: {args.content[:120]}...")
        test_summarize(args.content)
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
