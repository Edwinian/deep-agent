"""Shared tracing decorators for Langfuse and LangSmith."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from langfuse import observe
from langsmith import traceable

from constants.function_name import FunctionName

F = TypeVar("F", bound=Callable[..., object])


def trace(name: str | FunctionName) -> Callable[[F], F]:
    """Apply Langfuse and LangSmith tracing with a shared span/run name."""
    span_name = str(name)

    def decorator(func: F) -> F:
        return observe(name=span_name)(traceable(name=span_name)(func))  # type: ignore[return-value]

    return decorator
