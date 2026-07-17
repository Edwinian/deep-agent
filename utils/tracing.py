"""Shared tracing helpers for Langfuse, LangSmith, and FastAPI endpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.runnables.config import RunnableConfig
from langfuse import get_client, observe
from langsmith import traceable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from constants.function_name import FunctionName

F = TypeVar("F", bound=Callable[..., object])

logger = logging.getLogger(__name__)

_langsmith_write_access_ok: bool | None = None


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() == "true"


def _langsmith_api_url() -> str:
    return (os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")


def init_tracing() -> None:
    """Synchronize LangSmith/LangChain tracing flags after ``load_dotenv``."""
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if api_key:
        # LangChain auto-tracing reads LANGCHAIN_API_KEY.
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)

    if _env_flag("LANGSMITH_TRACING") or _env_flag("LANGCHAIN_TRACING_V2"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if not (os.getenv("LANGSMITH_PROJECT") or "").strip():
            os.environ["LANGSMITH_PROJECT"] = "deep-agents-from-scratch"
        # SSE streams exit before background callback batches flush.
        os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "false")

    validate_langsmith_connection()


def is_langfuse_enabled() -> bool:
    """Return True when Langfuse credentials are configured."""
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public_key and secret_key)


def is_langsmith_enabled() -> bool:
    """Return True when LangSmith tracing is enabled and an API key is set."""
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    return bool(api_key) and (
        _env_flag("LANGSMITH_TRACING") or _env_flag("LANGCHAIN_TRACING_V2")
    )


def validate_langsmith_connection() -> bool:
    """Probe LangSmith write access once; log a clear warning when writes are rejected."""
    global _langsmith_write_access_ok

    if not is_langsmith_enabled():
        _langsmith_write_access_ok = None
        return False

    if _langsmith_write_access_ok is not None:
        return _langsmith_write_access_ok

    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    project = (os.getenv("LANGSMITH_PROJECT") or "deep-agents-from-scratch").strip()
    try:
        import requests

        response = requests.get(
            f"{_langsmith_api_url()}/info",
            headers={"x-api-key": api_key},
            timeout=10,
        )
        if response.status_code == 403:
            logger.warning(
                "LangSmith tracing is enabled but LANGSMITH_API_KEY was rejected with "
                "HTTP 403 on run ingest. Traces will not appear in LangSmith until you "
                "use a key with write access from https://smith.langchain.com/settings "
                "(project=%s, endpoint=%s).",
                project,
                _langsmith_api_url(),
            )
            _langsmith_write_access_ok = False
            return False
        if response.status_code >= 400:
            logger.warning(
                "LangSmith tracing probe failed: HTTP %s %s",
                response.status_code,
                response.text[:200],
            )
            _langsmith_write_access_ok = False
            return False

        logger.info(
            "LangSmith tracing enabled for project '%s' (%s).",
            project,
            _langsmith_api_url(),
        )
        _langsmith_write_access_ok = True
        return True
    except Exception as exc:
        logger.warning("LangSmith tracing probe failed: %s", exc)
        _langsmith_write_access_ok = False
        return False


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """Create a LangChain callback handler for Langfuse tracing."""
    if not is_langfuse_enabled():
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def _uses_langsmith_auto_tracing() -> bool:
    """True when LangChain env auto-tracing should own LangSmith runs."""
    return is_langsmith_enabled() and _env_flag("LANGCHAIN_TRACING_V2")


def with_tracing_config(
    config: RunnableConfig,
    *,
    thread_id: str,
    agent_id: int,
) -> RunnableConfig:
    """Attach Langfuse callbacks and tracing metadata for a graph run.

    LangSmith uses ``LANGCHAIN_TRACING_V2`` auto-tracing (env-driven). Do not
    also attach ``LangChainTracer`` or wrap with ``tracing_context`` — that
    causes ``TracerException: No indexed run ID`` on streamed graphs.
    """
    merged: dict[str, Any] = dict(config)
    callbacks = list(merged.get("callbacks") or [])

    langfuse_handler = get_langfuse_handler()
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)

    if callbacks:
        merged["callbacks"] = callbacks

    metadata = dict(merged.get("metadata") or {})
    metadata.setdefault("langfuse_session_id", thread_id)
    metadata.setdefault("langfuse_tags", [f"agent_id:{agent_id}"])
    metadata.setdefault("thread_id", thread_id)
    metadata.setdefault("agent_id", agent_id)
    if is_langsmith_enabled():
        project = (os.getenv("LANGSMITH_PROJECT") or "").strip()
        if project:
            metadata.setdefault("ls_project_name", project)
        metadata.setdefault("session_id", thread_id)
        metadata.setdefault(
            "langsmith_tags",
            [f"agent_id:{agent_id}", f"thread_id:{thread_id}"],
        )
    merged["metadata"] = metadata

    return merged


def flush_langsmith_traces() -> None:
    """Best-effort flush of pending LangSmith runs (important for SSE streams)."""
    if not is_langsmith_enabled() or _langsmith_write_access_ok is False:
        return
    try:
        from langsmith import Client

        Client().flush()
    except Exception as exc:
        logger.warning("LangSmith flush failed: %s", exc)


@contextmanager
def langsmith_request_context(
    *,
    thread_id: str,
    agent_id: int,
    tags: list[str] | None = None,
):
    """No-op placeholder kept for import stability.

    LangSmith tracing is handled by ``LANGCHAIN_TRACING_V2`` auto-tracing on
    graph runs. An extra ``tracing_context`` here conflicts with streamed
  subgraph callbacks and produces ``No indexed run ID`` errors.
    """
    del thread_id, agent_id, tags
    yield


def trace(name: str | FunctionName) -> Callable[[F], F]:
    """Apply Langfuse and LangSmith tracing with a shared span/run name."""
    span_name = str(name)

    def decorator(func: F) -> F:
        return observe(name=span_name)(traceable(name=span_name)(func))  # type: ignore[return-value]

    return decorator


class TracingMiddleware(BaseHTTPMiddleware):
    """Create one Langfuse span and one LangSmith run per HTTP request.

    SSE streaming endpoints are excluded: ``BaseHTTPMiddleware`` can interfere
    with ``text/event-stream`` delivery. Those routes attach Langfuse/LangSmith
    callbacks on the graph ``RunnableConfig`` instead (see ``with_tracing_config``).
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Do not wrap SSE endpoints with BaseHTTPMiddleware tracing.
        # Streaming responses are sensitive to middleware behavior and should
        # remain a direct pass-through.
        accepts_sse = "text/event-stream" in (request.headers.get("accept", "").lower())
        if accepts_sse or request.url.path == "/chats/stream":
            return await call_next(request)

        if not is_langfuse_enabled() and not _uses_langsmith_auto_tracing():
            return await call_next(request)

        span_name = f"{request.method} {request.url.path}"
        request_input = {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else None,
        }

        async def _handle() -> Response:
            return await call_next(request)

        # LangSmith auto-tracing on graph runs — only wrap Langfuse for HTTP spans.
        if not is_langfuse_enabled():
            return await _handle()

        client = get_client()
        with client.start_as_current_observation(
            name=span_name,
            as_type="span",
            input=request_input,
        ) as observation:
            try:
                response = await _handle()
            except Exception as exc:
                observation.update(level="ERROR", status_message=str(exc))
                client.flush()
                raise

            observation.update(output={"status_code": response.status_code})
            client.flush()
            return response
