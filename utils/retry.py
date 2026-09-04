"""Tool-level retry with exponential backoff.

Used by deepagents tools (via ``ToolNode`` / tool ``invoke``) and plain LangGraph
graphs that execute the same ``BaseTool`` instances. Prefer wrapping *network I/O*
call sites or MCP-backed tools; do not retry permanent client errors or empty
business results.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

T = TypeVar("T")

OnRetry = Callable[[BaseException, int, float], None]


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_MAX_ATTEMPTS = _env_int("TOOL_RETRY_MAX_ATTEMPTS", 3)
DEFAULT_INITIAL_INTERVAL = _env_float("TOOL_RETRY_INITIAL_INTERVAL", 0.5)
DEFAULT_BACKOFF_FACTOR = _env_float("TOOL_RETRY_BACKOFF_FACTOR", 2.0)
DEFAULT_MAX_INTERVAL = _env_float("TOOL_RETRY_MAX_INTERVAL", 8.0)

# Permanent / non-retryable failure markers (name-based so we avoid hard imports).
_NON_RETRYABLE_EXC_NAMES = frozenset(
    {
        "MissingAPIKeyError",
        "InvalidAPIKeyError",
        "BadRequestError",
        "ForbiddenError",
        "UsageLimitExceededError",
        "AuthenticationError",
        "PermissionError",
        "ValidationError",
        "HTTPException",
    }
)


def is_transient_exception(exc: BaseException) -> bool:
    """Return True when ``exc`` is worth retrying with backoff."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False
    if isinstance(exc, (MemoryError, RecursionError)):
        return False
    if type(exc).__name__ in _NON_RETRYABLE_EXC_NAMES:
        return False
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return False

    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status in {408, 425, 429, 500, 502, 503, 504}
        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        ):
            return True
    except ImportError:
        pass

    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError, OSError)):
        return True

    # Qdrant client wraps transport failures in ResponseHandlingException.
    if type(exc).__name__ == "ResponseHandlingException":
        return True

    # Unknown errors: do not retry by default (avoid amplifying bugs).
    return False


def _compute_delay(
    attempt: int,
    *,
    initial_interval: float,
    backoff_factor: float,
    max_interval: float,
    jitter: bool,
) -> float:
    """Delay before the attempt after a failure (``attempt`` is 1-based failure count)."""
    delay = min(initial_interval * (backoff_factor ** (attempt - 1)), max_interval)
    if jitter:
        delay += random.uniform(0.0, delay * 0.1)
    return delay


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_interval: float = DEFAULT_INITIAL_INTERVAL,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_interval: float = DEFAULT_MAX_INTERVAL,
    jitter: bool = True,
    retry_on: Callable[[BaseException], bool] = is_transient_exception,
    on_retry: OnRetry | None = None,
) -> T:
    """Run ``fn`` with exponential backoff on transient failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not retry_on(exc):
                raise
            delay = _compute_delay(
                attempt,
                initial_interval=initial_interval,
                backoff_factor=backoff_factor,
                max_interval=max_interval,
                jitter=jitter,
            )
            logger.warning(
                "Transient tool failure (attempt %s/%s): %s; retrying in %.2fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(exc, attempt, delay)
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


async def async_retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_interval: float = DEFAULT_INITIAL_INTERVAL,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_interval: float = DEFAULT_MAX_INTERVAL,
    jitter: bool = True,
    retry_on: Callable[[BaseException], bool] = is_transient_exception,
    on_retry: OnRetry | None = None,
) -> T:
    """Async variant of :func:`retry_with_backoff`."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not retry_on(exc):
                raise
            delay = _compute_delay(
                attempt,
                initial_interval=initial_interval,
                backoff_factor=backoff_factor,
                max_interval=max_interval,
                jitter=jitter,
            )
            logger.warning(
                "Transient tool failure (attempt %s/%s): %s; retrying in %.2fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(exc, attempt, delay)
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


def wrap_tool_with_retry(
    tool: BaseTool,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_interval: float = DEFAULT_INITIAL_INTERVAL,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_interval: float = DEFAULT_MAX_INTERVAL,
    jitter: bool = True,
    retry_on: Callable[[BaseException], bool] = is_transient_exception,
    on_retry: OnRetry | None = None,
) -> BaseTool:
    """Wrap a LangChain tool so ``invoke`` / ``ainvoke`` retry transient errors.

    Safe for tools used by ``create_deep_agent`` and plain LangGraph ``ToolNode``.
    """
    original_invoke = tool.invoke
    original_ainvoke = tool.ainvoke

    def invoke(tool_input: object, config: object = None, **kwargs: object) -> object:
        return retry_with_backoff(
            lambda: original_invoke(tool_input, config=config, **kwargs),
            max_attempts=max_attempts,
            initial_interval=initial_interval,
            backoff_factor=backoff_factor,
            max_interval=max_interval,
            jitter=jitter,
            retry_on=retry_on,
            on_retry=on_retry,
        )

    async def ainvoke(tool_input: object, config: object = None, **kwargs: object) -> object:
        return await async_retry_with_backoff(
            lambda: original_ainvoke(tool_input, config=config, **kwargs),
            max_attempts=max_attempts,
            initial_interval=initial_interval,
            backoff_factor=backoff_factor,
            max_interval=max_interval,
            jitter=jitter,
            retry_on=retry_on,
            on_retry=on_retry,
        )

    tool.invoke = invoke  # type: ignore[method-assign]
    tool.ainvoke = ainvoke  # type: ignore[method-assign]
    return tool
