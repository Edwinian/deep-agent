"""Backward-compatible re-exports for LangGraph tracing helpers."""

from __future__ import annotations

from utils.tracing import (
    get_langfuse_handler,
    is_langfuse_enabled,
    with_tracing_config,
)

# Legacy name kept for existing imports.
with_langfuse_config = with_tracing_config

__all__ = [
    "get_langfuse_handler",
    "is_langfuse_enabled",
    "with_langfuse_config",
    "with_tracing_config",
]
