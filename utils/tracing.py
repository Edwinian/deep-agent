"""Shared tracing helpers for Langfuse, LangSmith, and FastAPI endpoints."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, TypeVar

from langchain_core.runnables.config import RunnableConfig
from langfuse import get_client, observe
from langsmith import traceable
from langsmith.run_helpers import tracing_context
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from constants.function_name import FunctionName

F = TypeVar("F", bound=Callable[..., object])


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() == "true"


def init_tracing() -> None:
    """Synchronize LangSmith/LangChain tracing flags after ``load_dotenv``."""
    if _env_flag("LANGSMITH_TRACING") or _env_flag("LANGCHAIN_TRACING_V2"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"


def is_langfuse_enabled() -> bool:
    """Return True when Langfuse credentials are configured."""
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public_key and secret_key)


def is_langsmith_enabled() -> bool:
    """Return True when LangSmith tracing is enabled and an API key is set."""
    api_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    return bool(api_key) and (
        _env_flag("LANGSMITH_TRACING") or _env_flag("LANGCHAIN_TRACING_V2")
    )


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """Create a LangChain callback handler for Langfuse tracing."""
    if not is_langfuse_enabled():
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


@lru_cache(maxsize=1)
def get_langsmith_tracer():
    """Create a LangChain tracer that sends runs to LangSmith."""
    if not is_langsmith_enabled():
        return None

    from langchain_core.tracers.langchain import LangChainTracer

    project = (os.getenv("LANGSMITH_PROJECT") or "").strip() or None
    return LangChainTracer(project_name=project)


def with_tracing_config(
    config: RunnableConfig,
    *,
    thread_id: str,
    agent_id: int,
) -> RunnableConfig:
    """Attach Langfuse and LangSmith callbacks plus session metadata."""
    merged: dict[str, Any] = dict(config)
    callbacks = list(merged.get("callbacks") or [])

    langfuse_handler = get_langfuse_handler()
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)

    langsmith_tracer = get_langsmith_tracer()
    if langsmith_tracer is not None:
        callbacks.append(langsmith_tracer)

    if callbacks:
        merged["callbacks"] = callbacks

    metadata = dict(merged.get("metadata") or {})
    metadata.setdefault("langfuse_session_id", thread_id)
    metadata.setdefault("langfuse_tags", [f"agent_id:{agent_id}"])
    metadata.setdefault("thread_id", thread_id)
    metadata.setdefault("agent_id", agent_id)
    merged["metadata"] = metadata

    return merged


def trace(name: str | FunctionName) -> Callable[[F], F]:
    """Apply Langfuse and LangSmith tracing with a shared span/run name."""
    span_name = str(name)

    def decorator(func: F) -> F:
        return observe(name=span_name)(traceable(name=span_name)(func))  # type: ignore[return-value]

    return decorator


class TracingMiddleware(BaseHTTPMiddleware):
    """Create one Langfuse span and one LangSmith run per HTTP request."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Do not wrap SSE endpoints with BaseHTTPMiddleware tracing.
        # Streaming responses are sensitive to middleware behavior and should
        # remain a direct pass-through.
        accepts_sse = "text/event-stream" in (request.headers.get("accept", "").lower())
        if accepts_sse or request.url.path == "/chats/stream":
            return await call_next(request)

        if not is_langfuse_enabled() and not is_langsmith_enabled():
            return await call_next(request)

        span_name = f"{request.method} {request.url.path}"
        request_input = {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else None,
        }

        async def _handle() -> Response:
            return await call_next(request)

        project_name = (os.getenv("LANGSMITH_PROJECT") or "").strip() or None
        with tracing_context(  # pylint: disable=not-context-manager
            project_name=project_name,
            enabled=is_langsmith_enabled(),
            tags=[request.method, request.url.path],
            metadata=request_input,
        ):
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
