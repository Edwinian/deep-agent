"""Langfuse tracing helpers for LangGraph agent invocations."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_core.runnables.config import RunnableConfig

def is_langfuse_enabled() -> bool:
    """Return True when Langfuse credentials are configured."""
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public_key and secret_key)


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """Create a LangChain callback handler for Langfuse tracing."""
    if not is_langfuse_enabled():
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def with_langfuse_config(
    config: RunnableConfig,
    *,
    thread_id: str,
    agent_id: int,
) -> RunnableConfig:
    """Attach Langfuse callbacks and session metadata to a RunnableConfig."""
    handler = get_langfuse_handler()
    if handler is None:
        return config

    merged: dict[str, Any] = dict(config)
    existing_callbacks = list(merged.get("callbacks") or [])
    merged["callbacks"] = [*existing_callbacks, handler]

    metadata = dict(merged.get("metadata") or {})
    metadata.setdefault("langfuse_session_id", thread_id)
    metadata.setdefault("langfuse_tags", [f"agent_id:{agent_id}"])
    merged["metadata"] = metadata

    return merged
